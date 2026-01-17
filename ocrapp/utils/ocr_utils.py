import os
from pydoc import text
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import fitz
import cv2
import numpy as np
import pytesseract
import torch
import re
import tempfile
import time
import requests
import joblib
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image, ImageOps, ImageFilter
from django.conf import settings
from pgvector.django import L2Distance
from openai import OpenAI  # <--- Added Import

# --- Model Imports ---
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ultralytics import YOLO
from doclayout_yolo import YOLOv10
from sentence_transformers import SentenceTransformer
from surya.foundation import FoundationPredictor
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor

from ..models import ArticleInfo
from .sql_executor import insert_news_analysis_entry
from ocrapp.utils.govt_info import GovtInfo
from ocrapp.prabhag_utils.prabhag import PrabhagPredictor
from .llm_utils import analyze_english_text_with_llm  # <--- IMPORT THIS

# Check for GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on device: {DEVICE}")

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(os.path.join(BASE_DIR, ".env"))

# ==============================================================================
# 1. DEFINE GLOBAL VARIABLES (Initially None)
# ==============================================================================
ARTICLE_MODEL = None
LAYOUT_MODEL = None
CLASSIFIER_MODEL = None
DISTRICT_MODEL = None
SENT_MODEL = None
TOKENIZER = None
EMBEDDER = None
CLASSIFIER_SVM = None
FOUNDATION_MODEL = None
DETECTION_MODEL = None
RECOGNITION_MODEL = None
GOVT_NEWS_MODEL = None
PRABHAG_PREDICTOR = None

# --- DUPLICATE DETECTION CONFIG ---
# Threshold for L2 Distance (Lower = Stricter/More Similar)
# 0.50 L2 Distance is approximately 87% Cosine Similarity
DUPLICATE_THRESHOLD_L2 = 0.50 

# --- SIMILARITY LLM CONFIG ---
LLM_BASE_URL = "http://localhost:8100/v1"
LLM_API_KEY = "EMPTY"
SIMILARITY_MODEL_NAME = "./qwen-7b-awq"

try:
    LLM_CLIENT = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
except Exception as e:
    print(f"⚠️ Warning: OpenAI Client Init Failed: {e}")
    LLM_CLIENT = None

# ==============================================================================
# 2. LAZY LOADING FUNCTION (Called inside the worker process)
# ==============================================================================
def load_models_if_needed():
    global ARTICLE_MODEL, LAYOUT_MODEL, CLASSIFIER_MODEL, DISTRICT_MODEL
    global SENT_MODEL, TOKENIZER, EMBEDDER, CLASSIFIER_SVM
    global FOUNDATION_MODEL, DETECTION_MODEL, RECOGNITION_MODEL
    global GOVT_NEWS_MODEL, PRABHAG_PREDICTOR

    if ARTICLE_MODEL is not None:
        return  # Already loaded in this process

    print(f"⏳ [Worker {os.getpid()}] Loading Models into VRAM...")

    # ---- YOLO Models ----
    ARTICLE_MODEL = YOLO(settings.BASE_DIR / "ocrapp" / "utils" / "model" / "A-1.pt")
    ARTICLE_MODEL.to(DEVICE)

    LAYOUT_MODEL = YOLOv10(settings.BASE_DIR / "ocrapp" / "utils" / "model" / "h2.pt")
    LAYOUT_MODEL.to(DEVICE)

    CLASSIFIER_MODEL = YOLO(settings.BASE_DIR / "ocrapp" / "utils" / "model" / "classifier.pt")
    CLASSIFIER_MODEL.to(DEVICE)

    DISTRICT_MODEL = YOLO(settings.BASE_DIR / "ocrapp" / "utils" / "model" / "district_best.pt")
    DISTRICT_MODEL.to(DEVICE)

    # ---- Sentiment ----
    SENT_MODEL_PATH = settings.BASE_DIR / "ocrapp" / "utils" / "SentimentAnalysis" / "local_model"
    TOKENIZER = AutoTokenizer.from_pretrained(SENT_MODEL_PATH)
    SENT_MODEL = AutoModelForSequenceClassification.from_pretrained(SENT_MODEL_PATH).to(DEVICE)

    # ---- Transformer Models ----
    MODEL_DIR_TRANSFORMER = os.path.join(settings.BASE_DIR, "ocrapp", "utils", "models_transformer")
    EMBEDDER_PATH = os.path.join(MODEL_DIR_TRANSFORMER, "sentence_transformer_model")
    SVM_PATH = os.path.join(MODEL_DIR_TRANSFORMER, "svm_transformer.joblib")
    
    try:
        if os.path.exists(EMBEDDER_PATH) and os.path.exists(SVM_PATH):
            EMBEDDER = SentenceTransformer(EMBEDDER_PATH, device=DEVICE)
            CLASSIFIER_SVM = joblib.load(SVM_PATH)
    except Exception as e:
        print(f"⚠️ Transformer Load Error: {e}")

    # ---- Surya Models ----
    try:
        FOUNDATION_MODEL = FoundationPredictor()
        DETECTION_MODEL = DetectionPredictor()
        RECOGNITION_MODEL = RecognitionPredictor(FOUNDATION_MODEL)
    except Exception as e:
        print(f"⚠️ Surya Load Error: {e}")

    # ---- Misc Models ----
    GOVT_CLS_PATH = settings.BASE_DIR / "ocrapp" / "utils" / "model" / "govt_news_classifier_best.pkl"
    if os.path.exists(GOVT_CLS_PATH):
        GOVT_NEWS_MODEL = joblib.load(GOVT_CLS_PATH)
    
    PRABHAG_PREDICTOR = PrabhagPredictor()

    print(f"✅ [Worker {os.getpid()}] All Models Loaded Successfully.")


# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================
OCR_ENGINE = "surya"
ASR_API_URL = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"
ASR_API_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": os.getenv("bhasini_KEY")
}
STOPWORDS = {
    "a","an","the","and","or","but","if","while","of","at","by","for","with",
    "about","against","between","into","through","during","before","after","to",
    "from","in","out","on","off","over","under","again","further","then","once",
    "here","there","all","any","both","each","few","more","most","other","some",
    "such","no","nor","not","only","own","same","so","than","too","very",
    "can","will","just","don","should","now","is","am","are","was","were",
    "be","been","being","do","does","did","doing","have","has","had","having"
}
custom_config = r"--oem 3 --psm 6 -l guj"
title_config = r"--oem 3 --psm 7 -l guj"

def check_similarity_with_llm(text1, text2):
    """
    Sends two texts to the LLM and asks if they are strictly similar.
    Returns True if Similar, False otherwise.
    """
    if not LLM_CLIENT:
        return False

    # Truncate text to avoid token limits and speed up processing
    t1_safe = str(text1)[:1000]
    t2_safe = str(text2)[:1000]

    prompt = f"""
    Compare the following two news article texts. 
    Do they refer to the exact same specific event or story?

    TEXT 1: {t1_safe}

    TEXT 2: {t2_safe}

    Answer only with the word "YES" if they are the same event, or "NO" if they are different.
    """

    try:
        response = LLM_CLIENT.chat.completions.create(
            model=SIMILARITY_MODEL_NAME,
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant that compares news articles. You only answer YES or NO."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for consistent deterministic answers
            max_tokens=10
        )
        content = response.choices[0].message.content.strip().upper()

        # Check if YES is in the response (handles cases like "YES." or "Yes, similar")
        return "YES" in content
    except Exception as e:
        print(f"⚠️ Similarity LLM Error: {e}")
        return False

def page_to_bgr(page, scale=2.0):
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def run_surya_ocr(image_input):
    if RECOGNITION_MODEL is None or DETECTION_MODEL is None:
        return ""
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    else:
        image = image_input
    try:
        # Added langs=["gu"]
        rec_results = RECOGNITION_MODEL(images=[image], det_predictor=DETECTION_MODEL)
        if rec_results and len(rec_results) > 0:
            text_lines = [line.text for line in rec_results[0].text_lines]
            return " ".join(text_lines).strip()
    except Exception as e:
        print(f"❌ OCR Error: {e}")
    return ""

def ocr_crop(crop, config=custom_config, enhance=True):
    if crop is None or crop.size == 0: return ""
    if isinstance(crop, np.ndarray):
        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
    else:
        pil_img = crop

    if enhance:
        pil_img = ImageOps.autocontrast(pil_img, cutoff=2)
        pil_img = pil_img.filter(ImageFilter.SHARPEN)
        pil_img = pil_img.filter(ImageFilter.EDGE_ENHANCE_MORE)

    if OCR_ENGINE == "surya":
        txt = run_surya_ocr(pil_img)
    else:
        txt = pytesseract.image_to_string(pil_img, config=config)
    return txt.strip()

def translate_text(text):
    payload = {
        "pipelineTasks": [{
            "taskType": "translation",
            "config": {"language": {"sourceLanguage": "gu", "targetLanguage": "en"}}
        }],
        "inputData": {"input": [{"source": text}]},
    }
    try:
        resp = requests.post(ASR_API_URL, json=payload, headers=ASR_API_HEADERS, timeout=60)
        if resp.status_code == 200:
            return resp.json().get("pipelineResponse", [{}])[0].get("output", [{}])[0].get("target", "")
    except Exception:
        pass
    return ""

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return ' '.join([w for w in text.split() if w not in STOPWORDS])

def analyze_sentiment(text):
    text = preprocess_text(text)
    if not text: return "neutral : 0.00%"
    
    tokens = TOKENIZER(text, truncation=False, return_tensors="pt")["input_ids"][0]
    chunks = [tokens[i:i + 512] for i in range(0, len(tokens), 512)]
    agg = np.zeros(3)
    for c in chunks:
        c = torch.cat([
            torch.tensor([TOKENIZER.cls_token_id]),
            c[:510],
            torch.tensor([TOKENIZER.sep_token_id])
        ]).to(DEVICE)
        
        with torch.no_grad():
            logits = SENT_MODEL(c.unsqueeze(0)).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0] 
        agg += probs
    agg /= len(chunks)
    label = ["negative", "neutral", "positive"][np.argmax(agg)]
    return f"{label} : {agg[np.argmax(agg)] * 100:.2f}%"

# ... (Keep your _area, _intersection_area, _is_duplicate, _y_overlap, _merge_boxes, _normalize_blocks functions exactly as they are) ...
def _area(b):
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)

def _intersection_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x_left = max(ax1, bx1)
    y_top = max(ay1, by1)
    x_right = min(ax2, bx2)
    y_bottom = min(ay2, by2)
    if x_right < x_left or y_bottom < y_top:
        return 0
    return (x_right - x_left) * (y_bottom - y_top)

def _is_duplicate(a, b, threshold=0.50):
    inter = _intersection_area(a, b)
    if inter == 0:
        return False
    small = min(_area(a), _area(b))
    return (inter / small) > threshold

def _normalize_blocks(blocks, article_height):
    if not blocks: return []
    items = []
    for x, y, txt, w, h in blocks:
        items.append([(x, y, x + w, y + h), txt])

    unique_items = []
    for bbox, txt in items:
        is_dup = False
        for existing_bbox, _ in unique_items:
            if _is_duplicate(bbox, existing_bbox, threshold=0.50):
                is_dup = True
                break
        if not is_dup:
            unique_items.append((bbox, txt))
    
    if not unique_items: return []

    total_h = 0
    for bbox, _ in unique_items:
        h = bbox[3] - bbox[1]
        total_h += h
    avg_h = total_h / len(unique_items)
    y_tolerance = avg_h * 0.25 
    if y_tolerance < 1: y_tolerance = 10 

    unique_items = sorted(unique_items, key=lambda z: (int(z[0][1] / y_tolerance), z[0][0]))

    out = []
    for (x1, y1, x2, y2), t in unique_items:
        out.append([x1, y1, t, x2 - x1, y2 - y1])
    return out

# ==============================================================================
# 4. MAIN PROCESS FUNCTION
# ==============================================================================
def process_pdf(pdf_path, news_paper="", pdf_link="", lang="gu", is_article=False, article_district=None, is_connect=False, is_urgent=False, uuid=False):
    
    # 🔥 CRITICAL: Load models ONLY when the task starts
    load_models_if_needed()

    is_gs = "GS" in (news_paper or "").upper()
    render_scale = 2.0 
    use_enhancement = False if is_gs else True
    
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        img = page_to_bgr(page, scale=render_scale)
        
        if is_article:
            h, w = img.shape[:2]
            boxes_iter = [(0, 0, w, h)]
        else:
            article_preds = ARTICLE_MODEL.predict(img, verbose=False)
            boxes_iter = article_preds[0].boxes

        for i, b in enumerate(boxes_iter):
            try:
                if is_article:
                    x1, y1, x2, y2 = b
                else:
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())

                crop = img[y1:y2, x1:x2]
                
                cls_results = CLASSIFIER_MODEL.predict(crop, verbose=False)
                cls_result = cls_results[0]
                layout_preds = LAYOUT_MODEL.predict(crop, imgsz=1280, conf=0.2, verbose=False)
                top1_index = cls_result.probs.top1
                article_type_pred = cls_result.names[top1_index]
                
                # --- Only process Title and Plain Text with conf > 0.5 ---
                guj_title = ""
                title_parts = []
                plain_parts = []
                is_govt = False
                result = layout_preds[0]
                rects = result.boxes.xyxy.cpu().numpy()
                cls_ids = result.boxes.cls.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy() if hasattr(result.boxes, "conf") else np.ones_like(cls_ids)

                names = getattr(result, "names", None)
                if names is None:
                    names = getattr(LAYOUT_MODEL, "names", {}) or {}

                def class_name(cid):
                    try:
                        return str(names[int(cid)])
                    except Exception:
                        return str(int(cid))

                for rect, cls_id, conf in zip(rects, cls_ids, confs):
                    if conf <= 0.5:
                        continue
                    x1b, y1b, x2b, y2b = map(int, rect)
                    name_l = class_name(cls_id).lower()
                    w = x2b - x1b
                    h = y2b - y1b

                    # Title detection
                    if ("title" in name_l) or ("headline" in name_l):
                        sub_crop = crop[y1b:y2b, x1b:x2b]
                        txt = ocr_crop(sub_crop, config=title_config, enhance=use_enhancement)
                        if txt:
                            title_parts.append((x1b, y1b, txt, w, h))
                        continue

                    # Plain text detection
                    if any(k in name_l for k in ["text", "paragraph", "body", "plain", "figure_caption"]):
                        sub_crop = crop[y1b:y2b, x1b:x2b]
                        txt = ocr_crop(sub_crop, config=custom_config, enhance=use_enhancement)
                        if txt:
                            txt += ' --- '
                            plain_parts.append((x1b, y1b, txt, w, h))
                        continue
                    continue

                # -------- APPLY MERGING + DEDUP --------
                article_height = y2 - y1
                ordered_titles = _normalize_blocks(title_parts, article_height)
                ordered_plain = _normalize_blocks(plain_parts, article_height)

                guj_title = " ".join([b[2] for b in ordered_titles])
                guj_text = " ".join([b[2] for b in ordered_plain])

                if not guj_text.strip():
                    continue
                
                gujarati_only = re.sub(r'[^\u0A80-\u0AFF\s\.]', '', guj_text)
                gujarati_only = re.sub(r'\s+', ' ', gujarati_only).strip()
                eng_text = translate_text(gujarati_only)

                # --- LLM OBSERVATION START ---
                cate_llm, is_govt_llm, conf_llm, sentiment_llm, prabhag_llm, prabhag_id_llm = analyze_english_text_with_llm(eng_text)
                
                # Derive cat_id_llm using mapping
                cat_id_llm = GovtInfo.govt_cat_id_mapping.get(cate_llm, None)

                print(f"🔍 LLM Observation -> Category: {cate_llm} (ID: {cat_id_llm}), Is_Govt: {is_govt_llm}, Conf: {conf_llm}%, Prabhag: {prabhag_llm}, ID: {prabhag_id_llm}")
                # --- LLM OBSERVATION END ---

                # --- sentiment ---
                sentiment_gravity = 0.0
                sentiment_label = ""
                sentiment_raw = analyze_sentiment(eng_text)
                try:
                    parts = sentiment_raw.split(":")
                    sentiment_label = parts[0].strip()
                    pct_str = parts[1].replace("%", "").strip()
                    sentiment_gravity = float(pct_str)
                except Exception:
                    sentiment_gravity = 0.0
                sentiment = sentiment_label
                # --- end sentiment ---

                # --- government classifier (Transformer-based) ---
                model_pred = 0
                is_govt = False
                try:
                    if EMBEDDER and CLASSIFIER_SVM and eng_text:
                        lines = [line.strip() for line in eng_text.split("\n") if line.strip()]
                        if not lines: lines = [eng_text.strip()]
                        if lines:
                            embeddings = EMBEDDER.encode(lines)
                            preds = CLASSIFIER_SVM.predict(embeddings)
                            if 1 in preds:
                                model_pred = 1
                                is_govt = True
                            else:
                                model_pred = 0
                                is_govt = False
                except Exception as e:
                    print(f"Govt Classification Error: {e}")
                    is_govt = False
                govt_word = ""
                # --- end government classifier ---

                # Print diagnostics
                if guj_title: print(f"[Title] {guj_title}")
                else: print("[Title] (none)")
                print(f"[Sentiment] label={sentiment_label} gravity={sentiment_gravity}")
                print(f"[GovClassifier] is_govt={is_govt}")

                dist = "Unknown"
                dist_token = None
                is_govt_rule_based, govt_word = GovtInfo.detect_govt_word(guj_text) 
                category, cat_word, cat_id  = GovtInfo.detect_category(guj_text)

                ### district strings ####
                unwanted = ["%", "[", "|", "]", "દ્વારા", "જામનગર મોર્નિંગ", ',']
                
                # Initialize district variables
                district = None
                taluka = None
                dcode = None
                tcode = None
                string_type = "NA"
                match_index = -1
                matched_token = "NA"

                # 1. Try District Model (Visual Detection)
                district_ocr_text = ""
                try:
                    dist_preds = DISTRICT_MODEL.predict(crop, verbose=False)
                    if dist_preds and len(dist_preds[0].boxes) > 0:
                        d_box = dist_preds[0].boxes[0]
                        dx1, dy1, dx2, dy2 = map(int, d_box.xyxy[0].tolist())
                        h_c, w_c = crop.shape[:2]
                        dx1, dy1 = max(0, dx1), max(0, dy1)
                        dx2, dy2 = min(w_c, dx2), min(h_c, dy2)
                        if dx2 > dx1 and dy2 > dy1:
                            dist_crop = crop[dy1:dy2, dx1:dx2]
                            district_ocr_text = ocr_crop(dist_crop, config=title_config, enhance=use_enhancement)
                except Exception as e:
                    print(f"District Model Error: {e}")

                if district_ocr_text:
                    clean_dist = district_ocr_text
                    for u in unwanted: clean_dist = clean_dist.replace(u, "")
                    district, taluka, dcode, tcode, string_type, match_index, matched_token = GovtInfo.detect_district_rapidfuzz(clean_dist)
                    if district: string_type = f"YOLO_Model_{string_type}"

                # 2. Fallback to Full Text
                if district is None:
                    clean_text = guj_text
                    for u in unwanted: clean_text = clean_text.replace(u, "")
                    district, taluka, dcode, tcode, string_type, match_index, matched_token = GovtInfo.detect_district_rapidfuzz(clean_text)

                # 3. Fallback to Title
                if district is None and guj_title:
                    clean_title = guj_title
                    for u in unwanted: clean_title = clean_title.replace(u, "")
                    district, taluka, dcode, tcode, string_type, match_index, matched_token = GovtInfo.detect_district_rapidfuzz(clean_title)
                #### district strings #####

                # --- Disambiguation Logic (Jetpur, Mandvi, Mangrol, Mahuva) ---
                SOUTH_GUJARAT_PAPERS = {
                    "Bharuch-Narmada Bhaskar", "Gujarat Mitra (Bardoli-Vyara-Bharuch)",
                    "Gujarat Mitra (Navsari-Valsad-Vapi)", "Gujarat Mitra (Surat)",
                    "Janadesh (Surat)", "Loksatta Jansatta (Surat)", "Navsari Bhaskar",
                    "Sandesh (Surat)", "Sandesh (Surat-Tapi)", "Soneri Surat",
                    "Surat City Bhaskar", "Surat Mitra", "Valsad-Vapi Bhaskar",
                    "Sandesh (Navsari-Dang)"
                }
                
                SAURASHTRA_PAPERS = {
                    "Akila (Rajkot)", "Akila (Saurashtra)", "Amreli Bhaskar", "Amreli Express",
                    "Botad Samachar", "Divya Bhaskar (Rajkot)", "Gujarat Samachar (Bhavnagar)",
                    "Gujarat Samachar (Rajkot)", "Gujarat Samachar (Surendranagar)",
                    "Jamnagar Bhaskar", "Jamnagar Morning", "Jamnagar Uday", "Khas Khabar (Rajkot)",
                    "Phulchhab (Rajkot)", "Porbandar Khabar", "Porbandar-Sorath Bhaskar",
                    "Rajkot Halchal", "Rajkot Sandesh Evening Daily", "Sandesh (Bhavnagar)",
                    "Sandesh (Halar)", "Sandesh (Morbi)", "Sandesh (Rajkot)",
                    "Sandesh (Saurashtra)", "Sandesh (Zalawad-Ahmedadad)",
                    "Sanj Samachar (Rajkot)", "Sanj Samachar (Saurashtra)", "Soneri Surat",
                    "Surendranagar Bhaskar", "Divya Bhaskar (Bhavnagar)", "Rajkot Mirror",
                    "Saurashtra Aaj Tak", "Saurashtra Headline", "Hatavo Brashtachar", "Halar Update"
                }

                current_paper_name = news_paper.strip() if news_paper else ""
                is_south_paper = any(p.lower() in current_paper_name.lower() for p in SOUTH_GUJARAT_PAPERS)
                is_saurashtra_paper = any(p.lower() in current_paper_name.lower() for p in SAURASHTRA_PAPERS)

                # 1. Jetpur (Rajkot vs Chhota Udepur)
                if taluka == "જેતપુર":
                    if is_saurashtra_paper:
                        district = "રાજકોટ"
                        dcode = 9
                        tcode = "09020" # Jetpur City
                        string_type += "_JetpurRajkotFix"
                        print(f"📍 Jetpur Disambiguation: Forcing to Rajkot for paper '{current_paper_name}'")

                # 2. Mandvi (Kutch vs Surat)
                if taluka == "માંડવી":
                    if is_south_paper:
                        district = "સુરત"
                        dcode = 22
                        tcode = "22007" # Surat-Mandvi
                        string_type += "_MandviSuratFix"
                        print(f"📍 Mandvi Disambiguation: Forcing to Surat for paper '{current_paper_name}'")

                # 3. Mangrol (Junagadh vs Surat)
                if taluka in ["માંગરોળ", "માંગરોલ"]:
                    if is_south_paper:
                        district = "સુરત"
                        dcode = 22
                        tcode = "22002" # Surat-Mangrol (Code for Mangrol in Surat)
                        string_type += "_MangrolSuratFix"
                        print(f"📍 Mangrol Disambiguation: Forcing to Surat for paper '{current_paper_name}'")

                # 4. Mahuva (Bhavnagar vs Surat)
                if taluka == "મહુવા":
                    if is_south_paper:
                        district = "સુરત"
                        dcode = 22
                        tcode = "22015" # Surat-Mahuva
                        string_type += "_MahuvaSuratFix"
                        print(f"📍 Mahuva Disambiguation: Forcing to Surat for paper '{current_paper_name}'")

                
                prabhag_name, prabhag_ID, confidence = PRABHAG_PREDICTOR.predict(eng_text)
                print(f"{is_govt, govt_word, category, cat_word, district, taluka, cat_id, dcode, tcode, prabhag_name, prabhag_ID} --- model_pred: {model_pred} ")
                
                # save crop image
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                img_name = f"article_{page_num+1}_{i+1}_{ts}.png"
                save_path = os.path.join(settings.MEDIA_ROOT, "articles", img_name)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, crop)

                final_newspaper_name = news_paper if news_paper else os.path.basename(pdf_path)
                article_remarks = (
                        f"{district}, {taluka}, "
                        f"{dcode}-{tcode}, "
                        f"{string_type}, "
                        f"match_index={match_index}, "
                        f"matched_token={matched_token}, "
                        f"model_pred={model_pred}, "
                        f"article_type_pred={article_type_pred},"
                        f"{conf_llm}%"
                    )
            
                # --- CONSOLIDATED DUPLICATE CHECK (DB BASED) ---
                is_duplicate = False
                duplicate_original_id = None
                vec = None

                if EMBEDDER and eng_text:
                    try:
                        # 1. Generate Vector
                        vec = EMBEDDER.encode(eng_text) 
                        
                        # 2. Query DB (With District Filter)
                        if district and district != "NA":
                            similar_articles = ArticleInfo.objects.filter(
                                created_at__date=datetime.now().date(),
                                district=district  # <--- KEPT AS REQUESTED
                            ).annotate(
                                distance=L2Distance('embedding', vec)
                            ).filter(distance__lt=DUPLICATE_THRESHOLD_L2).order_by('distance')[:1]

                            if similar_articles.exists():
                                match = similar_articles.first()
                                is_duplicate = True
                                duplicate_original_id = match.id
                                print(f"⚡ Duplicate Found in DB! Matches ID: {match.id}")
                        
                        # --- 3. RACE CONDITION FIX (The 0.3s Delay) ---
                        # Only run this if we think it's unique so far
                        if not is_duplicate and district and district != "NA":
                            import time, random
                            # Wait 0.1 to 0.5 seconds to let other workers commit
                            time.sleep(random.uniform(0.1, 0.5))
                            
                            # Re-run the exact same check
                            # (This catches the record inserted by another worker during the sleep)
                            double_check = ArticleInfo.objects.filter(
                                created_at__date=datetime.now().date(),
                                district=district
                            ).annotate(
                                distance=L2Distance('embedding', vec)
                            ).filter(distance__lt=DUPLICATE_THRESHOLD_L2).order_by('distance')[:1]

                            if double_check.exists():
                                match = double_check.first()
                                is_duplicate = True
                                duplicate_original_id = match.id
                                print(f"⚡ Race-Condition Duplicate Caught! Matches ID: {match.id}")

                    except Exception as e:
                        print(f"⚠️ Embedding/Duplicate Check Error: {e}")

                is_govt_push_nic = False
                if article_type_pred == "article" and model_pred == 1  and sentiment_label in ["negative"] \
                    and is_govt_llm == True and district not in [None, "Unknown"] and sentiment_llm == "Negative" and is_duplicate == False :

                    # --- CHECK FOR SEMANTIC SIMILARITY ---
                    is_similar = False
                    similar_rec_id = None
                    try:
                        # Fetch candidates: Same District, Same Day
                        # We only care about checking against potential existing Govt News
                        candidates = ArticleInfo.objects.filter(
                            created_at__date=datetime.now().date(),
                            district=district
                        ).exclude(translated_text__isnull=True).exclude(translated_text__exact="")

                        # OPTIONAL: Limit candidates to recent ones or specific count if performance is issue
                        # candidates = candidates.order_by('-id')[:20]

                        for cand in candidates:
                            # Use LLM to check similarity
                            if check_similarity_with_llm(cand.translated_text, eng_text):
                                is_similar = True
                                similar_rec_id = cand.id
                                print(f"🔍 Semantic Similarity Found with ID: {cand.id} - Skipping NIC Push")
                                break
                    except Exception as e:
                        print(f"⚠️ Similarity Check Process Error: {e}")

                    if is_similar:
                        is_govt_push_nic = False
                        article_remarks += f", Similar_to_ID={similar_rec_id}"
                    else:
                        is_govt_push_nic = True
                    

                is_manual = False
                uploadType = 'newspaper'
                if is_article:
                    is_manual = True
                    # Manual article: Strictly get district from dcode only
                    try:
                        if article_district:
                            is_govt_push_nic = True
                            uploadType = 'article'
                            # Convert input to int (dcode)
                            d_id = int(article_district)
                            
                            # Map directly using GovtInfo
                            if d_id in GovtInfo.district_id_mapping:
                                district = GovtInfo.district_id_mapping.get(d_id)
                                dcode = d_id
                                
                                # Reset other fields to ensure data consistency
                                taluka = None
                                tcode = None
                                string_type = "MANUAL_DCODE"
                                match_index = -1
                                matched_token = str(article_district)
                    except Exception as e:
                        print(f"⚠️ Manual dcode mapping error: {e}")
                        district, taluka, dcode, tcode, string_type, match_index, matched_token = GovtInfo.detect_district_rapidfuzz(article_district)
                    # keep is_urgent as passed (you can force if desired)
                    # is_urgent = True  # optional: force urgent for manual articles

                if uuid:
                    uuid = int(uuid)

                article = ArticleInfo.objects.create(
                    pdf_name=final_newspaper_name if final_newspaper_name else "NA",
                    pdf_link=pdf_link if pdf_link else "NA",
                    page_number=page_num + 1,
                    article_id=f"Article_{i+1}",
                    gujarati_title=guj_title if guj_title else "NA",
                    gujarati_text=guj_text if guj_text else "NA",
                    translated_text=eng_text,
                    image=f"articles/{img_name}",
                    sentiment=sentiment if sentiment else "NA",
                    sentiment_gravity=sentiment_gravity,
                    article_type="Unknown",
                    article_category=  cate_llm if cate_llm else "NA" ,
                    category_word = cat_word if cat_word else "NA",
                    cat_Id = cat_id_llm,
                    is_govt = is_govt ,
                    govt_word = govt_word,
                    govt_word_rule_based = is_govt_rule_based,
                    district = district if district else "NA",
                    dist_token=dist_token,
                    distict_word = taluka,
                    Dcode = dcode,
                    Tcode = tcode ,
                    prabhag = prabhag_llm,
                    prabhag_ID = prabhag_id_llm,
                    is_govt_push_nic = is_govt_push_nic,
                    remarks = article_remarks,
                    is_manual = is_manual,
                    is_govt_llm = is_govt_llm,
                    is_duplicate = is_duplicate,          
                    duplicate_id = duplicate_original_id , 
                    is_govt_llm_confidence = conf_llm ,
                    embedding = vec, # Save vector to DB
                    is_urgent = is_urgent,
                    extra_flag_text = uuid if uuid else 0 
                    
                )
                print(article.image)
                if is_govt_push_nic:
                    #insert_news_analysis_entry(article)
                    article_info_insert = (
                            article.page_number or 1,                              # 1 -> @Page_id INT
                            i + 1 or 1,                                            # 2 -> @Article_id INT
                            article.pdf_name or "",                                # 3 -> @Newspaper_name NVARCHAR(200)
                            str(article.image) or "" ,                             # 4 -> @Article_link NVARCHAR(500)
                            article.gujarati_text or "",                           # 5 -> @Gujarati_Text NVARCHAR(MAX)
                            article.translated_text or "",                         # 6 -> @English_Text NVARCHAR(MAX)
                            article.sentiment or "",                               # 7 -> @Text_Sentiment NVARCHAR(100)
                            article.is_govt or 0,                                  # 8 -> @Is_govt BIT
                            article.article_category or "",                        # 9 -> @Category NVARCHAR(200)
                            article.prabhag or "",                                 # 10 -> @Prabhag NVARCHAR(200)
                            article.district or "",                                # 11 -> @District NVARCHAR(200)
                            article.Dcode or None,                                 # 12 -> @Dcode INT
                            article.Tcode or "",                                   # 13 -> @Tcode VARCHAR(50)
                            article.cat_Id or None,                                # 14 -> @Cat_code INT
                            article.article_type or "",                            # 15 -> @Title NVARCHAR(500)
                            article.prabhag_ID,                                    # 16 - prabhagID
                            article.id,                                            # 17 - AI_ID INT
                            1 if article.is_urgent else 0,                         # 18 - @Is_Urgent INT (1/0)
                            1 if article.is_duplicate else 0,                      # 19 - @Is_Duplicate INT (1/0)
                            article.duplicate_id if article.duplicate_id else 0 ,  # 20 - @Duplicate_AI_ID INT
                            uuid if uuid else 0 ,                                  # 21 - @UUID INT
                            uploadType                                            # 22 - @UploadType NVARCHAR(50)
                    )

                    # --------------------------------------------
                    # Push the data to SQL Server
                    # --------------------------------------------
                    insert_news_analysis_entry(article_info_insert)
            except Exception as e:
                print("Error:", e)
    doc.close()
    return True
