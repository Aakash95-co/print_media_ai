import os
import fitz
import cv2
import numpy as np
import pytesseract
import torch
import re
import tempfile
import time
import requests
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ultralytics import YOLO
from doclayout_yolo import YOLOv10
from django.conf import settings
from ..models import ArticleInfo
from .sql_executor import insert_news_analysis_entry
from ocrapp.utils.govt_info import GovtInfo
from ocrapp.prabhag_utils.prabhag import PrabhagPredictor
prabhag_predictor = PrabhagPredictor()
from dotenv import load_dotenv
from pathlib import Path
import joblib
from PIL import Image, ImageOps, ImageFilter  # Added for preprocessing
from .govrt_model import load_models, predict_texts
#tfidf, svd, mlp, scaler, svm = load_models("./models")
from django.conf import settings

# --- Surya OCR Imports ---
from surya.foundation import FoundationPredictor
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
# -------------------------

MODEL_DIR_G = os.path.join(settings.BASE_DIR, "ocrapp", "utils", "model")
tfidf, svd, mlp, scaler, svm = load_models(MODEL_DIR_G)

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(os.path.join(BASE_DIR, ".env"))

# ---- Load Models ----
ARTICLE_MODEL = YOLO(settings.BASE_DIR / "ocrapp" / "utils" / "model" / "A-1.pt")
LAYOUT_MODEL = YOLOv10(settings.BASE_DIR / "ocrapp" / "utils" / "model" / "h2.pt")
CLASSIFIER_MODEL = YOLO(settings.BASE_DIR / "ocrapp" / "utils" / "model" / "classifier.pt")

# ---- Surya Models Initialization ----
try:
    print("⏳ Loading Surya Models...")
    FOUNDATION_MODEL = FoundationPredictor()
    DETECTION_MODEL = DetectionPredictor()
    RECOGNITION_MODEL = RecognitionPredictor(FOUNDATION_MODEL)
    print("✅ Surya Models Loaded.")
except Exception as e:
    print(f"⚠️ Surya Models failed to load: {e}")
    FOUNDATION_MODEL = None
    DETECTION_MODEL = None
    RECOGNITION_MODEL = None

# Set this to "surya" or "tesseract"
OCR_ENGINE = "surya" 
# ------------------------------

# ---- Translation ----
ASR_API_URL = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"
ASR_API_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": os.getenv("bhasini_KEY")
}

# ---- Sentiment ----
SENT_MODEL_PATH = settings.BASE_DIR / "ocrapp"  / "utils" / "SentimentAnalysis" / "local_model"
TOKENIZER = AutoTokenizer.from_pretrained(SENT_MODEL_PATH)
SENT_MODEL = AutoModelForSequenceClassification.from_pretrained(SENT_MODEL_PATH)

STOPWORDS = {
    "a","an","the","and","or","but","if","while","of","at","by","for","with",
    "about","against","between","into","through","during","before","after","to",
    "from","in","out","on","off","over","under","again","further","then","once",
    "here","there","all","any","both","each","few","more","most","other","some",
    "such","no","nor","not","only","own","same","so","than","too","very",
    "can","will","just","don","should","now","is","am","are","was","were",
    "be","been","being","do","does","did","doing","have","has","had","having"
}

# ---- OCR helper ----
custom_config = r"--oem 3 --psm 6 -l guj"
title_config = r"--oem 3 --psm 7 -l guj"  # Added title config

def page_to_bgr(page, scale=2.0):  # match Streamlit scale
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def run_surya_ocr(image_input):
    """
    Takes a PIL Image or file path and returns the extracted text string.
    """
    if RECOGNITION_MODEL is None or DETECTION_MODEL is None:
        return ""

    # Ensure input is a PIL Image
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    else:
        image = image_input

    try:
        # Run Recognition
        rec_results = RECOGNITION_MODEL(images=[image], det_predictor=DETECTION_MODEL)
        
        # Extract and Join Text
        if rec_results and len(rec_results) > 0:
            text_lines = [line.text for line in rec_results[0].text_lines]
            full_text = " ".join(text_lines)
            return full_text.strip()
            
    except Exception as e:
        print(f"❌ OCR Error: {e}")
    return ""

# Update: Added 'enhance' parameter to toggle filters
def ocr_crop(crop, config=custom_config, enhance=True):
    if crop is None or crop.size == 0:
        return ""
    
    # Convert BGR to PIL Image if needed
    if isinstance(crop, np.ndarray):
        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
    else:
        pil_img = crop

    # -----------------------------
    # 🔥 Conditional Preprocessing
    if enhance:
        pil_img = ImageOps.autocontrast(pil_img, cutoff=2)
        pil_img = pil_img.filter(ImageFilter.SHARPEN)
        pil_img = pil_img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    # -----------------------------

    if OCR_ENGINE == "surya":
        txt = run_surya_ocr(pil_img)
    else:
        txt = pytesseract.image_to_string(pil_img, config=config)
        
    return txt.strip()

# ---- Translation ----
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

# ---- Sentiment ----
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return ' '.join([w for w in text.split() if w not in STOPWORDS])

def analyze_sentiment(text):
    text = preprocess_text(text)
    if not text:
        return "neutral : 0.00%"
    tokens = TOKENIZER(text, truncation=False, return_tensors="pt")["input_ids"][0]
    chunks = [tokens[i:i + 512] for i in range(0, len(tokens), 512)]
    agg = np.zeros(3)
    for c in chunks:
        c = torch.cat([
            torch.tensor([TOKENIZER.cls_token_id]),
            c[:510],
            torch.tensor([TOKENIZER.sep_token_id])
        ])
        with torch.no_grad():
            logits = SENT_MODEL(c.unsqueeze(0)).logits
        probs = torch.softmax(logits, dim=1).numpy()[0]
        agg += probs
    agg /= len(chunks)
    label = ["negative", "neutral", "positive"][np.argmax(agg)]
    return f"{label} : {agg[np.argmax(agg)] * 100:.2f}%"

GOVT_CLS_PATH = settings.BASE_DIR / "ocrapp" / "utils" / "model" / "govt_news_classifier_best.pkl"
GOVT_NEWS_MODEL = joblib.load(GOVT_CLS_PATH)

# ============================================================
#   GEOMETRY HELPERS (full-area duplicate + Y-merge) - FROM REF
# ============================================================

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

def _y_overlap(a, b, y_percent=0.25):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    a_h = ay2 - ay1
    b_h = by2 - by1
    avg_h = (a_h + b_h) / 2
    return abs(ay1 - by1) <= avg_h * y_percent

def _merge_boxes(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return (min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2))

def _normalize_blocks(blocks, article_height):
    """
    blocks = [(x, y, text, w, h)]
    Output merged blocks sorted properly.
    """
    if not blocks:
        return []

    # Convert to bbox + text format
    items = []
    for x, y, txt, w, h in blocks:
        items.append([(x, y, x + w, y + h), txt])

    # Sort by y then x
    items = sorted(items, key=lambda z: (z[0][1], z[0][0]))

    final_items = []

    for bbox, txt in items:
        skip = False

        # -------------------------------
        #   RULE 1: REMOVE DUPLICATES
        # -------------------------------
        for fb, _ in final_items:
            if _is_duplicate(bbox, fb, threshold=0.50):
                skip = True
                break

        if skip:
            continue

        # -------------------------------
        #   RULE 2: Y-LINE MERGE
        # -------------------------------
        merged = False
        for i, (fb, ftxt) in enumerate(final_items):
            if _y_overlap(bbox, fb, y_percent=0.25):
                newbox = _merge_boxes(bbox, fb)

                # merge text left→right
                if bbox[0] < fb[0]:
                    final_items[i] = (newbox, txt + " " + ftxt)
                else:
                    final_items[i] = (newbox, ftxt + " " + txt)

                merged = True
                break

        if not merged:
            final_items.append((bbox, txt))

    # Convert back to format (x, y, text, w, h)
    out = []
    for (x1, y1, x2, y2), t in final_items:
        out.append([x1, y1, t, x2 - x1, y2 - y1])

    return out

# ---- Process PDF ----
def process_pdf(pdf_path, news_paper="", pdf_link="", lang="gu", is_article=False, district=None, is_connect=False) :
    
    # ---------------------------------------------------------
    # 🔧 GS SPECIFIC TWEAKS
    # If newspaper is GS, use higher res (3.0) and disable edge enhancement
    # ---------------------------------------------------------
    is_gs = "GS" in (news_paper or "").upper()
    render_scale = 3.0 if is_gs else 2.0
    use_enhancement = False if is_gs else True
    
    if is_gs:
        print(f"ℹ️ Detected GS: Using High-Res Mode (Scale {render_scale}) & No Filters")

    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        img = page_to_bgr(page, scale=render_scale) # <--- Use dynamic scale
        
        article_preds = ARTICLE_MODEL.predict(img, verbose=False)
        for i, b in enumerate(article_preds[0].boxes):
            try:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                crop = img[y1:y2, x1:x2]
                
                cls_results = CLASSIFIER_MODEL.predict(crop, verbose=False)
                cls_result = cls_results[0]
                layout_preds = LAYOUT_MODEL.predict(crop, imgsz=1280, conf=0.2, verbose=False)
                top1_index = cls_result.probs.top1
                article_type_pred = cls_result.names[top1_index]
                conf_score = cls_result.probs.top1conf.item()
                print(f"conf score ----------- {conf_score} ----  {article_type_pred}")
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
                        # Pass enhance flag
                        txt = ocr_crop(sub_crop, config=title_config, enhance=use_enhancement)
                        if txt:
                            title_parts.append((x1b, y1b, txt, w, h))
                        continue

                    # Plain text detection
                    if any(k in name_l for k in ["text", "paragraph", "body", "plain", "figure_caption"]):
                        sub_crop = crop[y1b:y2b, x1b:x2b]
                        # Pass enhance flag
                        txt = ocr_crop(sub_crop, config=custom_config, enhance=use_enhancement)
                        if txt:
                            txt += ' --- '
                            plain_parts.append((x1b, y1b, txt, w, h))
                        continue

                    # Ignore all other classes
                    continue

                # -------- APPLY MERGING + DEDUP (FROM REF) --------
                article_height = y2 - y1
                ordered_titles = _normalize_blocks(title_parts, article_height)
                ordered_plain = _normalize_blocks(plain_parts, article_height)

                guj_title = " ".join([b[2] for b in ordered_titles])
                guj_text = " ".join([b[2] for b in ordered_plain])
                # --- End filtering ---

                if not guj_text.strip():
                    continue
                eng_text = translate_text(guj_text)

                # --- sentiment (unchanged) ---
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

                # --- government classifier (model-based) ---
                model_pred = 3
                try:
                    #gov_pred = GOVT_NEWS_MODEL.predict([eng_text])[0]
                    #is_govt = bool(gov_pred == 1)
                    model_pred, _ = predict_texts([eng_text], tfidf, svd, mlp, scaler, svm)
                    model_pred = int(model_pred[0])
                    if model_pred == 1:
                        is_govt = True
                except Exception:
                    is_govt = False
                # keep compatibility; no keyword detected here
                govt_word = ""
                # --- end government classifier ---

                # Print diagnostics (title & sentiment gravity)
                if guj_title:
                    print(f"[Title] {guj_title}")
                else:
                    print("[Title] (none)")
                print(f"[Sentiment] label={sentiment_label} gravity={sentiment_gravity}")
                print(f"[GovClassifier] is_govt={is_govt}")

                dist = "Unknown"
                dist_token = None
                is_govt_rule_based, govt_word = GovtInfo.detect_govt_word(guj_text) 
                category, cat_word, cat_id  = GovtInfo.detect_category(guj_text)
                # district, taluka, dcode, tcode = GovtInfo.detect_district(guj_text)

                #### district strings #####
                unwanted = ["%", "[", "|", "]", "દ્વારા", "જામનગર મોર્નિંગ"]
                for u in unwanted:
                    guj_text = guj_text.replace(u, "")

                str_district = guj_text
                district, taluka, dcode, tcode, string_type, match_index, matched_token = GovtInfo.detect_district_rapidfuzz(str_district)

                if district is None and guj_title:
                    unwanted = ["%", "[", "|", "]", "દ્વારા", "જામનગર મોર્નિંગ"]
                    for u in unwanted:
                        guj_title = guj_title.replace(u, "")
                    str_district = guj_title
                    district, taluka, dcode, tcode, string_type, match_index, matched_token = GovtInfo.detect_district_rapidfuzz(str_district)
                ### district strings ###

                prabhag_name, prabhag_ID, confidence = prabhag_predictor.predict(eng_text)
                print(f"{is_govt, govt_word, category, cat_word, district, taluka, cat_id, dcode, tcode, prabhag_name, prabhag_ID} --- model_pred: {model_pred} ")
                # save crop image
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                img_name = f"article_{page_num+1}_{i+1}_{ts}.png"
                save_path = os.path.join(settings.MEDIA_ROOT, "articles", img_name)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, crop)
                is_govt_push_nic = False
                if article_type_pred == "article" and model_pred == 1 and district is not None and sentiment_label in ["negative", "neutral"]:
                    is_govt_push_nic = True

                # Use provided newspaper name, fallback to filename if empty
                final_newspaper_name = news_paper if news_paper else os.path.basename(pdf_path)
                # article_remarks = district + ', '+ taluka +', '+ str(dcode) + str(tcode) + string_type + str(match_index) + matched_token + f" model pred {model_pred}" +  f"article_type_pred : {article_type_pred} " 
                article_remarks = (
                        f"{district}, {taluka}, "
                        f"{dcode}-{tcode}, "
                        f"{string_type}, "
                        f"match_index={match_index}, "
                        f"matched_token={matched_token}, "
                        f"model_pred={model_pred}, "
                        f"article_type_pred={article_type_pred}"
                    )

                article = ArticleInfo.objects.create(
                    pdf_name=final_newspaper_name if final_newspaper_name else "NA",
                    pdf_link=pdf_link if pdf_link else "NA",  # <--- Save the PDF link here
                    page_number=page_num + 1,
                    article_id=f"Article_{i+1}",
                    gujarati_title=guj_title if guj_title else "NA",
                    gujarati_text=guj_text if guj_text else "NA",
                    translated_text=eng_text,
                    image=f"articles/{img_name}",
                    sentiment=sentiment if sentiment else "NA",
                    sentiment_gravity=sentiment_gravity,
                    article_type="Unknown",
                    article_category=  category if category else "NA" ,
                    category_word = cat_word if cat_word else "NA",
                    cat_Id = cat_id,
                    is_govt = is_govt ,
                    govt_word = govt_word,
                    govt_word_rule_based = is_govt_rule_based,
                    district = district if district else "NA",
                    dist_token=dist_token,
                    distict_word = taluka,
                    Dcode = dcode,
                    Tcode = tcode ,
                    prabhag = prabhag_name,
                    prabhag_ID = prabhag_ID,
                    is_govt_push_nic = is_govt_push_nic,
                    remarks = article_remarks,

                )
                print(article.image)
                if is_govt_push_nic:
                    #insert_news_analysis_entry(article)
                    article_info_insert = (
                            article.page_number or 1,                              # 1 -> @Page_id INT
                            i + 1 or 1,                                          # 2 -> @Article_id INT (✅ changed from string to int)
                            article.pdf_name or "",                           # 3 -> @Newspaper_name NVARCHAR(200)
                            str(article.image) or "" ,                        # 4 -> article.image or "" ,                          # 4 -> @Article_link NVARCHAR(500)
                            article.gujarati_text or "",                # 5 -> @Gujarati_Text NVARCHAR(MAX)
                            article.translated_text or "",         # 6 -> @English_Text NVARCHAR(MAX)
                            article.sentiment or "",               # 7 -> @Text_Sentiment NVARCHAR(100)
                            article.is_govt or 0,                             # 8 -> @Is_govt BIT
                            article.article_category or "",        #9 -> @Category NVARCHAR(200)
                            article.prabhag or "",                       # 10 -> @Prabhag NVARCHAR(200)
                            article.district or "",                # 11 -> @District NVARCHAR(200)
                            article.Dcode or None,                                  # 12 -> @Dcode INT
                            article.Tcode or "",            # 13 -> @Tcode VARCHAR(50)
                            article.cat_Id or None,                                  # 14 -> @Cat_code INT
                            article.article_type or "",             # 15 -> @Title NVARCHAR(500)
                            0 ,                                     # 16 - prabhagID
                    )

                    # --------------------------------------------
                    # Push the data to SQL Server
                    # --------------------------------------------
                    insert_news_analysis_entry(article_info_insert)
            except Exception as e:
                print("Error:", e)
    doc.close()
    return True
