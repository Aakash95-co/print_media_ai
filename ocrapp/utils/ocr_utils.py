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
def page_to_bgr(page, scale=4.0):
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def ocr_crop(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    txt = pytesseract.image_to_string(gray, lang="guj").strip()
    return txt.replace("\n", " ") if txt else ""

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

                    # Title detection
                    if ("title" in name_l) or ("headline" in name_l):
                        sub_crop = crop[y1b:y2b, x1b:x2b]
                        txt = ocr_crop(sub_crop)
                        if txt:
                            title_parts.append((x1b, txt))
                        continue

                    # Plain text detection
                    if any(k in name_l for k in ["text", "paragraph", "body", "plain"]):
                        sub_crop = crop[y1b:y2b, x1b:x2b]
                        txt = ocr_crop(sub_crop)
                        if txt:
                            plain_parts.append((x1b, txt))
                        continue

                    # Ignore all other classes
                    continue

                # Concatenate by ascending x-axis
                if title_parts:
                    guj_title = " ".join([t for _, t in sorted(title_parts, key=lambda x: x[0])])

                guj_text = " ".join([t for _, t in sorted(plain_parts, key=lambda x: x[0])])
                # --- End filtering ---

                if not guj_text.strip():
                    continue
                eng_text = translate_text(guj_text)
                sentiment = analyze_sentiment(eng_text)
                dist = "Unknown"
                dist_token = None
                is_govt, govt_word = GovtInfo.detect_govt_word(guj_text) 
                category, cat_word, cat_id  = GovtInfo.detect_category(guj_text)
                district, taluka, dcode, tcode = GovtInfo.detect_district(guj_text)
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
                    gujarati_text=guj_text,
                    translated_text=eng_text,
                    sentiment=sentiment,
                    image=f"articles/{img_name}",

                    article_type="Unknown",
                    article_category= category,
                    category_word = cat_word ,
                    cat_Id = cat_id,

                    is_govt = is_govt ,
                    govt_word = govt_word,

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
                         str(article.image) or "", # article.image or "" ,                          # 4 -> @Article_link NVARCHAR(500)
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
