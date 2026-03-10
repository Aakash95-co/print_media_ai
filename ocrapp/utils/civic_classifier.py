import requests
import json
import re
from django.conf import settings
from rapidfuzz import fuzz

# --- CONFIGURATION ---
# The URL of your vLLM server
VLLM_URL = "http://localhost:8100/v1/chat/completions"

# Stage 1: The Adapter (matches --lora-modules name in systemd)
STAGE_1_MODEL_NAME = "civic-classifier"

# Stage 2: The Base Model (matches the model path in vLLM start command)
STAGE_2_MODEL_NAME = "./qwen-14b-awq" #"./qwen-7b-awq"

CIVIC_KEYWORDS = [
    "મહાનગરપાલિકા",
    # "મ્યુનિસિપાલિટી", # Commented out as per requirement
    "મ્યુનિસિપલ",
    "કોર્પોરેશન",
    "મનપા",
]

FUZZY_THRESHOLD = 90


def _run_stage_1(gujarati_text):
    """
    Stage 1: Calls vLLM using the Fine-Tuned LoRA Adapter.
    """
    if not gujarati_text: return "0"

    # Alpaca Format (Required for the adapter)
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Classify the following Gujarati news brief into category 0 or 1.

### Input:
{}

### Response:
"""
    # Stage 1 usually works best with shorter context (~1200 chars)
    input_text = str(gujarati_text)[:1200]

    payload = {
        "model": STAGE_1_MODEL_NAME,
        "messages": [
            {"role": "user", "content": alpaca_prompt.format(input_text)}
        ],
        "temperature": 0.1,
        "max_tokens": 5
    }

    try:
        response = requests.post(VLLM_URL, json=payload, timeout=20)
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            if "1" in content: return "1"
            return "0"
        else:
            print(f"⚠️ Stage 1 API Error: {response.status_code}")
            return "0"
    except Exception as e:
        print(f"⚠️ Stage 1 Connection Error: {e}")
        return "0"


def _call_stage_2_api(text):
    """
    Stage 2: Calls vLLM using the Base Model with your EXACT Few-Shot Prompt.
    """

    # 1. THE EXACT SYSTEM INSTRUCTION & EXAMPLES
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
4. Output ONLY the number (0 or 1).

Here are examples:

### Text:
ડિસેમ્બર-2021મા ખાતમુહૂર્ત થયું , બારડોલી અને માંડવી તાલુકાને જોડતા હાઈલેવલ બ્રિજનું કામ તંત્રની ઉદાસીનતાને કારણે અનંતકાળ સુધી ખેંચાઈ રહ્યું છે.
### Label:
1

### Text:
કૃષ્ણનગરમાં તિક્ષ્ણ હથિયારો સાથે દારૂડિયાએ મચાવ્યો આતંક અને ઘરમાં ધૂસી જાનથી મારી નાખવાની ધમકી આપી હતી. પોલીસે ચાર આરોપીઓને ઝડપી પાડ્યા છે.
### Label:
0

### Text:
અમદાવાદ શહેરમાં રસ્તામાં પેચવર્ક અને ડ્રેનેજલાઈનમાં સમારકામના નામે ચોક્કસસ્થળ કે પ્રશ્નના ઉલ્લેખ વગર જ કરોડો રૂપિયાના કામ મંજૂર થઈ રહ્યા છે.
### Label:
1

### Text:
રાજકોટમાં બૂટલેગરના પુત્રએ સાગરિતો સાથે પોલીસ સ્ટેશન પર સોડા બોટલના ઘા કર્યા હતા.
### Label:
0
"""

    # 2. TRUNCATE TEXT (Exactly as per your script: 1500 chars)
    text_safe = str(text)
    if len(text_safe) > 1500:
        text_safe = text_safe[:1500] + "..."

    # 3. CONSTRUCT THE PROMPT
    # We send the instructions as 'system' and the specific text as 'user'.
    # We append "### Label:\n" to the user message to force the completion style.

    user_content = f"### Text:\n{text_safe}\n### Label:\n"

    payload = {
        "model": STAGE_2_MODEL_NAME,  # Hits the Base Model
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.1,  # Low temp for deterministic results
        "max_tokens": 5  # We only need 1 token, but 5 is safe
    }

    try:
        response = requests.post(VLLM_URL, json=payload, timeout=20)

        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']

            # 4. PARSE LOGIC
            # Since API returns just the completion, we don't need to split "### Label:"
            # We just look for the number in the response.
            if "0" in content:
                return 0
            if "1" in content:
                return 1

            # Fallback if model output is unexpected
            return 0
        else:
            print(f"⚠️ Stage 2 API Error: {response.status_code}")
            return 1  # Fallback to 1 if API fails (preserves Stage 1's decision)

    except Exception as e:
        print(f"⚠️ Stage 2 Connection Error: {e}")
        return 1


def match_civic_keywords(text):
    """
    Checks if any word in the text fuzzily matches the civic keywords.
    Returns: (is_match (int), remark (str or None))
    """
    if not text:
        return 0, None

    # Ensure text is string and split into words
    words = str(text).split()

    for idx, word in enumerate(words):
        for kw in CIVIC_KEYWORDS:
            # Calculate fuzzy ratio
            score = fuzz.ratio(word, kw)
            
            if score >= FUZZY_THRESHOLD:
                remark = f"index={idx}, word={word}, matched={kw}, score={score:.2f}%"
                return 1, remark

    return 0, None


def classify_civic_issue(gujarati_text):
    """
    Replaces the old ML/LLM pipeline with simple keyword matching.
    Returns: (prediction (int), remark (str or None))
    """
    if not gujarati_text or not str(gujarati_text).strip():
        return 0, None

    # 1. Clean Text (Keep only Gujarati characters and spaces)
    text_clean = re.sub(r'[^\u0A80-\u0AFF\s\.]', '', str(gujarati_text))
    text_clean = re.sub(r'\s+', ' ', text_clean).strip()

    # 2. Match
    is_match, remark = match_civic_keywords(text_clean)
    
    if is_match:
        print(f"🏗️ Civic Match Found: {remark}")
        return 1, remark
    
    return 0, None


# Keep this empty function so existing imports in tasks.py don't break
def _load_civic_model():
    pass