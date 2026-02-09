import torch
import gc
import os
import requests
import json
from django.conf import settings

# --- CONFIGURATION ---
MODEL_1_PATH = settings.BASE_DIR / "ocrapp" / "utils" / "model" / "qwen_gujarati_14k_final" #os.path.join(settings.BASE_DIR, "qwen_gujarati_14k_final")

# Reuse existing vLLM server (NO extra VRAM)
VLLM_URL = "http://localhost:8100/v1/chat/completions"
VLLM_MODEL_NAME = "./qwen-7b-awq"

# --- Stage 1: Global model holders (loaded once) ---
_CIVIC_MODEL = None
_CIVIC_TOKENIZER = None
_CIVIC_LOADED = False

try:
    from unsloth import FastLanguageModel
except ImportError:
    FastLanguageModel = None
    print("⚠️ Warning: 'unsloth' not found. Civic Stage 1 disabled.")


def _load_civic_model():
    """Load fine-tuned model ONCE into VRAM (called from load_models_if_needed)."""
    global _CIVIC_MODEL, _CIVIC_TOKENIZER, _CIVIC_LOADED

    if _CIVIC_LOADED or not FastLanguageModel:
        return

    if not os.path.exists(MODEL_1_PATH):
        print(f"⚠️ Civic model not found at: {MODEL_1_PATH}")
        _CIVIC_LOADED = True  # Don't retry
        return

    try:
        print(f"⏳ Loading Civic Fine-Tuned Model: {MODEL_1_PATH}...")
        _CIVIC_MODEL, _CIVIC_TOKENIZER = FastLanguageModel.from_pretrained(
            model_name=MODEL_1_PATH,
            max_seq_length=2048,
            load_in_4bit=True,
            dtype=None,
        )
        FastLanguageModel.for_inference(_CIVIC_MODEL)
        _CIVIC_LOADED = True
        print("✅ Civic Fine-Tuned Model Loaded.")
    except Exception as e:
        print(f"❌ Civic Model Load Error: {e}")
        _CIVIC_LOADED = True  # Don't retry on failure


def _run_stage_1(gujarati_text):
    """
    Stage 1: Fine-tuned model inference (model already in VRAM).
    Returns "0" or "1".
    """
    if _CIVIC_MODEL is None or _CIVIC_TOKENIZER is None:
        return "0"

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Classify the following Gujarati news brief into category 0 or 1.

### Input:
{}

### Response:
"""
    input_text = str(gujarati_text)[:1200]

    try:
        inputs = _CIVIC_TOKENIZER(
            [alpaca_prompt.format(input_text)],
            return_tensors="pt"
        ).to("cuda")

        outputs = _CIVIC_MODEL.generate(
            **inputs,
            max_new_tokens=4,
            use_cache=True,
            pad_token_id=_CIVIC_TOKENIZER.eos_token_id
        )

        decoded = _CIVIC_TOKENIZER.batch_decode(outputs)[0]

        # Free tensor memory (NOT the model)
        del inputs, outputs
        torch.cuda.empty_cache()

        response_part = decoded.split("### Response:\n")[1].strip()
        return "1" if "1" in response_part[:2] else "0"

    except Exception as e:
        print(f"❌ Stage 1 Inference Error: {e}")
        return "0"


def _call_stage_2_api(text):
    """
    Stage 2: Calls EXISTING vLLM server to refine classification.
    Uses 0 extra VRAM — just an HTTP request.
    """
    system_instruction = """You are a news classifier. Classify the text into:

**Label '1' (Municipal / Civic Infrastructure):**
- Bridges, Roads, Drainage, Water supply, Metro work.
- Municipal Corporations (SMC, AMC) budgets, tenders, deadlines.
- Political inauguration of civic projects.

**Label '0' (Crime / Police / Accidents / Other):**
- Police actions, Arrests, Murders, Thefts, Alcohol, Fights.
- Accidents, Fires, Personal tragedies (Dog bites, Suicide).
- General business/trade news not related to civic infrastructure.

**Instructions:**
1. Analyze the text.
2. If it mentions Police, Crime, Arrest, or Accident -> Label is 0.
3. If it mentions Bridge, Road, Gutter, or Civic Work -> Label is 1.
4. Output ONLY the number (0 or 1)."""

    # Truncate for API safety
    text_safe = str(text)[:6000]

    payload = {
        "model": VLLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Text: {text_safe}"}
        ],
        "temperature": 0.1,
        "max_tokens": 10
    }

    try:
        response = requests.post(VLLM_URL, json=payload, timeout=20)
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            if "0" in content:
                return 0
            if "1" in content:
                return 1
        else:
            print(f"⚠️ Stage 2 API Error: {response.status_code}")
    except Exception as e:
        print(f"⚠️ Stage 2 Connection Error: {e}")

    # Default: trust Stage 1 if API fails
    return 1


def classify_civic_issue(gujarati_text):
    """
    Hybrid Pipeline:
      Stage 1: Local Fine-Tuned Model (persistent in VRAM)
      Stage 2: API call to existing vLLM server (0 extra VRAM)

    Returns: int (0 or 1)
    """
    if not gujarati_text or not str(gujarati_text).strip():
        return 0

    # Stage 1
    prediction_stage_1 = _run_stage_1(gujarati_text)
    print(f"🏗️ Civic Stage 1 Prediction: {prediction_stage_1}")

    # Stage 2 (only if Stage 1 says "1")
    if prediction_stage_1 == "1":
        print("🔍 Civic Stage 2: Confirming via API...")
        final_pred = _call_stage_2_api(gujarati_text)
        print(f"🏁 Civic Final Prediction: {final_pred}")
        return final_pred

    return 0