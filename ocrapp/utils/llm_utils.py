import requests
import json
import re

# --- Configuration ---
VLLM_URL = "http://localhost:8100/v1/chat/completions"
MODEL_NAME = "./qwen-7b-awq"

# --- Gujarati -> English mapping ---
GUJ_TO_ENG = {
    "યોજના બાબત": "Scheme related", "કુદરતી આફતની અસર": "Natural disaster impact",
    "સલામતી/ સુરક્ષા બાબત": "Safety/Security related", "કૃષિ બજાર/વેચાણ લગત": "Agricultural market/sales related",
    "કૃષિ યુનિવર્સિટી": "Agriculture university", "જાહેર વિરોધ": "Public protest",
    "સરકારી ઓફિસ બાબતે અસંતોષ": "Dissatisfaction with government offices", "વહીવટી તંત્ર": "Administrative system",
    "વર્ગખંડ બાંધકામ બાબત": "Classroom construction related", "ખાતર/યુરીયા લગત": "Fertilizer/urea related",
    "કામની ગુણવત્તા બાબતે": "Work quality issues", "આર. ટી. ઓ. લગત": "RTO related",
    "અનિયમિતતા લગત": "Irregularities related", "પશુ/પશુ આરોગ્ય લગત": "Animal/animal health related",
    "ચૂંટણી": "Elections", "સ્વચ્છતા લગત": "Sanitation related",
    "કુપોષણ લગત": "Malnutrition related", "બાળમજૂરી": "Child labor",
    "ડૉક્ટર/સ્ટાફની અછત અથવા દવા/સાધનોની અછત, મર્યાદિત ઓપીડી": "Shortage of doctors/staff or medicines/equipment, limited OPD",
    "પેન્ડેમિક / રોગચાળા લગત": "Pandemic/epidemic related", "એમરજન્સી સેવાઓ": "Emergency services",
    "પર્યાવરણ લગત": "Environment related", "જંગલ કટિંગ લગત": "Forest cutting related",
    "વન્ય પ્રાણી/પક્ષીઓ લગત": "Wild animals/birds related", "ફોરેસ્ટ પરમીશન": "Forest permission",
    "અનાજના વિતરણ/ રેશન દુકાન લગત": "Food grain distribution/ration shop related", "તકનિકી સમસ્યાઓ": "Technical issues",
    "બેંક લગત": "Bank related", "ગુના અને કાયદા અમલીકરણ લગત": "Crime and law enforcement related",
    "ટ્રાફિક અને માર્ગ સુરક્ષા બાબતે": "Traffic and road safety related", "દબાણ લગત પ્રશ્નો": "Pressure-related issues",
    "વિજ ચોરી લગત": "Electricity theft related", "અકસ્માત / અકસ્માતનુ જોખમ": "Accident/accident risk",
    "ભ્રષ્ટાચાર": "Corruption", "ગેરકાયદેસર ખનન": "Illegal mining",
    "કર્મચારીઓ અંગે": "Employees related", "નીતિ અમલીકરણ લગત": "Policy implementation related",
    "બેરોજગારી/આર્થિક સમસ્યા": "Unemployment/economic issues", "પ્રદૂષણ": "Pollution",
    "સ્ટ્રીટ લાઈટને લગત": "Street light related", "ફૂડ/વોટર સેફ્ટી બાબતે": "Food/water safety related",
    "રોડ-રસ્તા લગત": "Roads/streets related", "પુલ/બ્રિજ બાબત": "Bridge related",
    "ઇન્ફ્રાસ્ટ્રક્ચર બાબત": "Infrastructure related", "ગેરકાયદેસર કામગીરી": "Illegal activities",
    "કૃષિ સંસાધન લગત": "Agricultural resources related", "બાગાયતી": "Horticulture",
    "અન્ય-વૃક્ષ": "Other - tree", "આરોગ્ય લગત": "Health related",
    "ઇન્ફ્રાસ્ટ્રક્ચર સુવિધા બાબત": "Infrastructure facilities related", "ઓફિસ વ્યવસ્થા લગત": "Office administration related",
    "ગટર/સ્ટ્રોમ વોટર ડ્રેઇન સુવિધા લગતી ફરિયાદ": "Sewer/storm water drainage facility complaint",
    "ગેસની પાઈપ લાઈનના મરામત અંગેના પ્રશ્નો": "Gas pipeline repair issues", "ટ્રાન્સપોર્ટ સુવિધાને લગત પ્રશ્નો": "Transport facility issues",
    "પાણી ભરાવવાના પ્રશ્ન": "Waterlogging issues", "પાણીને લગતા પ્રશ્ન (પાણીની અછત, ગંદુ પાણી, વગેરે)": "Water-related issues (shortage, dirty water, etc.)",
    "મજૂરોના શોષણ બાબત ફરિયાદ": "Complaints about labor exploitation", "મહેકમ બાબત": "Department related",
    "વેરો વસૂલવા અંગે": "Tax collection related", "શહેરી ઇન્ફ્રાસ્ટ્રક્ચરને લગતા પ્રશ્ન": "Urban infrastructure issues",
}

ENGLISH_CATEGORIES = list(GUJ_TO_ENG.values())
ENG_TO_GUJ = {eng: guj for guj, eng in GUJ_TO_ENG.items()}

def _call_vllm(prompt, json_mode=False, timeout=30):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1, 
        "max_tokens": 512
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    try:
        resp = requests.post(VLLM_URL, json=payload, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()['choices'][0]['message']['content']
        else:
            print(f"⚠️ vLLM Status {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"⚠️ vLLM Connection Error: {e}")
    return None

def _clean_json_response(text):
    """
    Removes markdown code blocks (```json ... ```) from LLM response
    to ensure json.loads() works.
    """
    if not text: return ""
    # Regex to capture content inside ```json ... ``` or just ``` ... ```
    pattern = r"```(?:json)?\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def analyze_english_text_with_llm(text):
    """
    Input: English text (string)
    Output: (gujarati_category_string, is_govt_boolean, confidence_score, sentiment_string)
    """
    if not text or not text.strip():
        return "અન્ય", False, 0, "Neutral"

    # --- 1. Check Government Relevance (JSON Mode) ---
    prompt_govt = (
        f"You are a professional news editor for a government monitoring cell. "
        f"TEXT: '{text}'\n\n"
        f"TASKS:\n"
        f"GOVERNMENT RELEVANCE: Decide if this is Government or Public Interest related (Yes or No).\n"
        f"   - MARK 'Yes' IF: Involves Govt Departments (Health, Police, Municipal, Education), "
        f"Public Grievances (Water, Roads, Pollution), Systemic issues, or Politicians.\n"
        f"   - MARK 'No' IF: Classified ads, Name change notices, Job postings, Matrimonial ads, "
        f"Private family disputes, or Individual accidents with no govt negligence.\n\n"
        f"OUTPUT: Return strictly a JSON object with keys: 'is_govt' (Yes/No) and 'confidence' (0-100 integer)."
    )
    
    is_govt_bool = False
    confidence = 0

    resp_govt = _call_vllm(prompt_govt, json_mode=True)
    
    if resp_govt:
        try:
            # Clean markdown before parsing
            clean_resp = _clean_json_response(resp_govt)
            data = json.loads(clean_resp)
            
            is_govt_str = str(data.get("is_govt", "No")).lower()
            is_govt_bool = True if "yes" in is_govt_str else False
            confidence = int(data.get("confidence", 0))
        except Exception as e:
            # Print the RAW response to debug why parsing failed
            print(f"⚠️ JSON Parse Error: {e}")
            print(f"⚠️ RAW LLM OUTPUT: {resp_govt}")
            pass

    # --- 2. Check Category (Strict Text Mode) ---
    cat_list = "\n".join(ENGLISH_CATEGORIES)
    prompt_cat = (
        f"You are a strict text classifier. Classify the given English text into ONE category from the predefined English list.\n\n"
        f"TEXT: '{text}'\n\n"
        f"CATEGORY LIST:\n{cat_list}\n\n"
        f"INSTRUCTIONS:\n"
        f"- Read the text carefully\n"
        f"- Choose EXACTLY ONE category that best matches the content\n"
        f"- Respond with ONLY the category name (in English)\n"
        f"- Do not add explanations or extra text\n"
        f"- If none of the categories fit, respond OTHER as category name."
    )

    eng_category = "OTHER"
    resp_cat = _call_vllm(prompt_cat, json_mode=False)
    if resp_cat:
        # Clean potential markdown from category response too
        eng_category = _clean_json_response(resp_cat).strip()

    # Map English category back to Gujarati (fallback to "અન્ય")
    clean_cat = eng_category.replace('"', '').replace("'", "").strip()
    # Handle cases where LLM might return "Category: Scheme related"
    for cat in ENGLISH_CATEGORIES:
        if cat.lower() in clean_cat.lower():
            clean_cat = cat
            break
            
    guj_category = ENG_TO_GUJ.get(clean_cat, "અન્ય")

    # --- 3. Check Sentiment (New) ---
    sentiment_str = "Neutral"
    try:
        # Manually constructing payload to support System Prompt
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a strict data classifier. Analyze the sentiment based ONLY on the factual content (events/actions). Return exactly one word: 'Positive', 'Negative', or 'Neutral'."
                },
                {"role": "user", "content": f"Text: {text}"}
            ],
            "temperature": 0.1, 
            "max_tokens": 64
        }
        
        resp_sent = requests.post(VLLM_URL, json=payload, timeout=30)
        if resp_sent.status_code == 200:
            raw_sent = resp_sent.json()['choices'][0]['message']['content']
            cleaned_sent = _clean_json_response(raw_sent).strip()
            
            # Normalize response
            if "positive" in cleaned_sent.lower(): sentiment_str = "Positive"
            elif "negative" in cleaned_sent.lower(): sentiment_str = "Negative"
            else: sentiment_str = "Neutral"
        else:
            print(f"⚠️ vLLM Sentiment Status {resp_sent.status_code}")

    except Exception as e:
        print(f"⚠️ vLLM Sentiment Error: {e}")
    
    return guj_category, is_govt_bool, confidence, sentiment_str