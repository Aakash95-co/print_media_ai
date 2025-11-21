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
from PIL import Image, ImageOps, ImageFilter  # <-- added

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(os.path.join(BASE_DIR, ".env"))
#load_dotenv(os.path.join(BASE_DIR, ".env"))


#from pathlib import Path
#BASE_DIR = Path(__file__).resolve().parent

# ---- Load Models ----
#ARTICLE_MODEL = YOLO("/home/cmoadmin/am/print_media/ocrapp/utils/model/A-1.pt")
#LAYOUT_MODEL = YOLOv10("/home/cmoadmin/am/print_media/ocrapp/utils/model/h2.pt")

ARTICLE_MODEL = YOLO(settings.BASE_DIR / "ocrapp" / "utils" / "model" / "A-1.pt")
LAYOUT_MODEL = YOLOv10(settings.BASE_DIR / "ocrapp" / "utils" / "model" / "h2.pt")

# ---- Translation ----
ASR_API_URL = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"
ASR_API_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": os.getenv("bhasini_KEY")
}

# ---- Sentiment ----
SENT_MODEL_PATH = "/home/cmoadmin/am/print_media/ocrapp/utils/SentimentAnalysis/local_model"
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
title_config = r"--oem 3 --psm 7 -l guj"  # <-- added (title OCR)
def page_to_bgr(page, scale=2.0):  # match Streamlit scale
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def ocr_crop(crop, config=custom_config):  # <-- updated to match ref
    if crop is None or crop.size == 0:
        return ""
    # Ensure PIL image
    if isinstance(crop, np.ndarray):
        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
    else:
        pil_img = crop
    # Auto-contrast + sharpen (as in ref)
    pil_img = ImageOps.autocontrast(pil_img, cutoff=2)
    pil_img = pil_img.filter(ImageFilter.SHARPEN)
    pil_img = pil_img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    txt = pytesseract.image_to_string(pil_img, config=config)
    return txt.strip()

# ===== Helpers for merging OCR blocks (from ref) =====
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
    blocks: list of tuples (x, y, text, w, h)
    returns: merged list sorted properly, same shape
    """
    if not blocks:
        return []

    items = []
    for x, y, txt, w, h in blocks:
        items.append([(x, y, x + w, y + h), txt])

    # Sort by y then x
    items = sorted(items, key=lambda z: (z[0][1], z[0][0]))
    final_items = []

    for bbox, txt in items:
        # Rule 1: remove duplicates
        dup = any(_is_duplicate(bbox, fb, threshold=0.50) for fb, _ in final_items)
        if dup:
            continue

        # Rule 2: merge same line (y-overlap)
        merged = False
        for i, (fb, ftxt) in enumerate(final_items):
            if _y_overlap(bbox, fb, y_percent=0.25):
                newbox = _merge_boxes(bbox, fb)
                if bbox[0] < fb[0]:
                    final_items[i] = (newbox, txt + " " + ftxt)
                else:
                    final_items[i] = (newbox, ftxt + " " + txt)
                merged = True
                break

        if not merged:
            final_items.append((bbox, txt))

    out = []
    for (x1, y1, x2, y2), t in final_items:
        out.append([x1, y1, t, x2 - x1, y2 - y1])
    return out
# ===== end helpers =====

# ---- Process PDF ----
def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        img = page_to_bgr(page)
        article_preds = ARTICLE_MODEL.predict(img, verbose=False)
        for i, b in enumerate(article_preds[0].boxes):
            try:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                crop = img[y1:y2, x1:x2]
                layout_preds = LAYOUT_MODEL.predict(crop, imgsz=1280, conf=0.2, verbose=False)

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

                    # Title block: use title_config (psm 7)
                    if ("title" in name_l) or ("headline" in name_l):
                        sub_crop = crop[y1b:y2b, x1b:x2b]
                        txt = ocr_crop(sub_crop, config=title_config)
                        if txt:
                            title_parts.append((x1b, y1b, txt, w, h))
                        continue

                    # Plain/body text: use custom_config (psm 6)
                    if any(k in name_l for k in ["plain text", "text", "body", "paragraph", "plain"]):
                        sub_crop = crop[y1b:y2b, x1b:x2b]
                        txt = ocr_crop(sub_crop, config=custom_config)
                        if txt:
                            txt += '---'  # keep same marker as ref
                            plain_parts.append((x1b, y1b, txt, w, h))
                        continue

                    # Ignore others
                    continue

                # ---- Merge and order like ref ----
                article_height = y2 - y1
                ordered_titles = _normalize_blocks(title_parts, article_height)
                ordered_plain = _normalize_blocks(plain_parts, article_height)

                guj_title = " ".join([b[2] for b in ordered_titles])
                guj_text = " ".join([b[2] for b in ordered_plain])
                # --- End filtering/merging ---

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
                try:
                    gov_pred = GOVT_NEWS_MODEL.predict([eng_text])[0]
                    is_govt = bool(gov_pred == 1)
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
                district, taluka, dcode, tcode = GovtInfo.detect_district_whole_word(guj_text) 
                prabhag_name, prabhag_ID, confidence = prabhag_predictor.predict(eng_text)
                print(f"{is_govt, govt_word, category, cat_word, district, taluka, cat_id, dcode, tcode, prabhag_name, prabhag_ID}")
                # save crop image
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                img_name = f"article_{page_num+1}_{i+1}_{ts}.png"
                save_path = os.path.join(settings.MEDIA_ROOT, "articles", img_name)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, crop)

                article = ArticleInfo.objects.create(

                    pdf_name=os.path.basename(pdf_path),
                    page_number=page_num + 1,
                    article_id=f"Article_{i+1}",
                    gujarati_title=guj_title,
                    gujarati_text=guj_text,
                    translated_text=eng_text,
                    image=f"articles/{img_name}",

                    sentiment=sentiment,
                    sentiment_gravity=sentiment_gravity,

                    article_type="Unknown",
                    article_category= category,
                    category_word = cat_word ,
                    cat_Id = cat_id,

                    is_govt = is_govt ,
                    govt_word = govt_word,
                    govt_word_rule_based = is_govt_rule_based,

                    district = district,
                    dist_token=dist_token,
                    distict_word = taluka,
                    Dcode = dcode,
                    Tcode = tcode ,
                    
                    prabhag = prabhag_name,
                    prabhag_ID = prabhag_ID,

                )
                print(article.image)
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
