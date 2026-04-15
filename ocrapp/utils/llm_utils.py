import requests
import json
import re
from difflib import get_close_matches

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
    "કૌભાંડ / ભ્રષ્ટાચાર": "Corruption / Scam", "ગેરકાયદેસર ખનન": "Illegal mining",
    "કર્મચારીઓ અંગે": "Employees related", "નીતિ અમલીકરણ લગત": "Policy implementation related",
    "બેરોજગારી/આર્થિક સમસ્યા": "Unemployment/economic issues", "પ્રદૂષણ": "Pollution",
    "સ્ટ્રીટ લાઈટને લગત": "Street light related", "ફૂડ/પીવાના પાણી સંબંધિત": "Food/drinking water related",
    "રોડ-રસ્તા લગત": "Roads/streets related", "પુલ/બ્રિજ બાબત": "Bridge related",
    "ઇન્ફ્રાસ્ટ્રક્ચર બાબત": "Infrastructure related", "ગેરકાયદેસર કામગીરી": "Illegal activities",
    "કૃષિ સંસાધન લગત": "Agricultural resources related", "બાગાયતી": "Horticulture",
    "અન્ય-વૃક્ષ": "Other - tree", "આરોગ્ય લગત": "Health related",
    "ઇન્ફ્રાસ્ટ્રક્ચર સુવિધા બાબત": "Infrastructure facilities related", "ઓફિસ વ્યવસ્થા લગત": "Office administration related",
    "ગટર/સ્ટ્રોમ વોટર ડ્રેઇન સુવિધા લગતી ફરિયાદ": "Sewer/storm water drainage facility complaint",
    "ગેસની પાઈપ લાઈનના મરામત અંગેના પ્રશ્નો": "Gas pipeline repair issues", "ટ્રાન્સપોર્ટ સુવિધાને લગત પ્રશ્નો": "Transport facility issues",
    "પાણી ભરાવવાના પ્રશ્ન": "Waterlogging issues", "પાણીને લગતા પ્રશ્ન (પાણીની અછત, ગંદુ પાણી, વગેરે)": "Water-related issues (shortage, dirty water, etc.)",
    "મજૂરોના શોષણ બાબત ફરિયાદ": "Complaints about labor exploitation", "મહેકમ બાબત": "Department related",
    "વેરો વસૂલવા અંગે": "Tax collection related", "શહેરી ઇન્ફ્રાસ્ટ્રક્ચરને લગતા પ્રશ્ન": "Urban infrastructure issues","જાહેર આરોગ્ય": "Public health related",
    "ચોરી બાબત": "Theft related", "શિક્ષણ વિષયક": "Education related"  }

ENGLISH_CATEGORIES = list(GUJ_TO_ENG.values())
ENG_TO_GUJ = {eng: guj for guj, eng in GUJ_TO_ENG.items()}



PRABHAG_DATA = [
    {"id": 1, "guj": "અન્ન નાગરિક પુરવઠો અને ગ્રાહક બાબતોનો વિભાગ", "eng": "Food, Civil Supplies & Consumer Affairs Department"},
    {"id": 2, "guj": "આદિજાતિ વિકાસ વિભાગ", "eng": "Tribal Development Department"},
    {"id": 3, "guj": "આરોગ્ય અને પરિવાર કલ્યાણ વિભાગ (આરોગ્ય શિક્ષણ)", "eng": "Health & Family Welfare Department (Health)"},
    {"id": 5, "guj": "ઉદ્યોગ અને ખાણ વિભાગ", "eng": "Industries & Mines Department"},
    {"id": 6, "guj": "ઉદ્યોગ અને ખાણ વિભાગ (કુટીર ઉદ્યોગ અને ગ્રામોદ્યોગ)", "eng": "Industries & Mines Department (Cottage & Rural Industries)"},
    {"id": 7, "guj": "ઉદ્યોગ અને ખાણ વિભાગ (પ્રવાસન, યાત્રા પ્રવાસ, દેવસ્થાન સં.)", "eng": "Industries & Mines Department (Tourism, Pilgrimage, Devsthanam Management)"},
    {"id": 8, "guj": "ઉર્જા અને પેટ્રો કેમિકલ્સ વિભાગ", "eng": "Energy & Petrochemicals Department"},
    {"id": 9, "guj": "ક્લાઇમેટ ચેન્જ વિભાગ", "eng": "Climate change Department"},
    {"id": 10, "guj": "કાયદા વિભાગ", "eng": "Legal Department"},
    {"id": 11, "guj": "કૃષિ, ખેડૂત કલ્યાણ અને સહકાર વિભાગ", "eng": "Agriculture, Farmer Welfare & Co-operation Department"},
    {"id": 12, "guj": "કૃષિ, ખેડૂત કલ્યાણ અને સહકાર વિભાગ (પશુપાલન, ગૌસંવર્ધન, મત્સ્યોદ્યોગ, સહકાર)", "eng": "Agriculture, Farmer Welfare & Co-operation Department (Animal Husbandry, Cow Breeding, Fisheries, Co-operation)"},
    {"id": 13, "guj": "ગૃહ વિભાગ", "eng": "Home Department"},
    {"id": 15, "guj": "નર્મદા, જળ સંપત્તિ, પાણી પુરવઠો અને કલ્પસર વિભાગ (સિંચાઈ)", "eng": "Narmada, Water Resourses, Water Supply & Kalpasar Department (Irrigation)"},
    {"id": 16, "guj": "નર્મદા, જળ સંપત્તિ, પાણી પુરવઠો અને કલ્પસર વિભાગ (નર્મદા)", "eng": "Narmada, Water Resourses, Water Supply & Kalpasar Department (Narmada)"},
    {"id": 17, "guj": "નર્મદા, જળ સંપત્તિ, પાણી પુરવઠો અને કલ્પસર વિભાગ (પાણી પુરવઠા)", "eng": "Narmada, Water Resourses, Water Supply & Kalpasar Department (Water Supply)"},
    {"id": 19, "guj": "નાણાં વિભાગ", "eng": "Finance Department"},
    {"id": 22, "guj": "પંચાયત, ગ્રામ ગૃહનિર્માણ અને ગ્રામ વિકાસ વિભાગ", "eng": "Panchayat, Rural Housing & Rural Development Department"},
    {"id": 23, "guj": "પંચાયત, ગ્રામ ગૃહનિર્માણ અને ગ્રામ વિકાસ વિભાગ (ગ્રામ વિકાસ)", "eng": "Panchayat, Rural Housing & Rural Development Department (Rural Development)"},
    {"id": 24, "guj": "બંદરો અને વાહન વ્યવહાર વિભાગ", "eng": "Ports & Transport Department"},
    {"id": 27, "guj": "મહેસૂલ વિભાગ", "eng": "Revenue Department"},
    {"id": 28, "guj": "મહિલા અને બાળ કલ્યાણ વિકાસ વિભાગ", "eng": "Women & Child Development Department"},
    {"id": 29, "guj": "માર્ગ અને મકાન વિભાગ", "eng": "Road & Building Department"},
    {"id": 30, "guj": "માહિતી અને પ્રસારણ વિભાગ", "eng": "Information & Broadcasting Department"},
    {"id": 31, "guj": "રમતગમત, યુવા અને સાંસ્કૃતિક પ્રવૃત્તિ વિભાગ", "eng": "Sports, Youth & Cultural Activities Department"},
    {"id": 32, "guj": "વન અને પર્યાવરણ વિભાગ", "eng": "Forest & Environment Department"},
    {"id": 33, "guj": "વૈધાનિક અને સંસદીય બાબતોનો વિભાગ (વૈધાનિક અને સંસદીય બાબતો)", "eng": "Legislative & Parliamentary Affairs Department"},
    {"id": 36, "guj": "ગુજરાત વિધાનસભા", "eng": "Legislative & Parliamentary Affairs Department (Gujarat Legislative Assembly)"},
    {"id": 37, "guj": "વિજ્ઞાન અને પ્રૌદ્યોગિકી વિભાગ", "eng": "Science & Technology Department"},
    {"id": 38, "guj": "સામાજિક ન્યાય અને અધિકારીતા વિભાગ", "eng": "Social Justice & Empowerment Department"},
    {"id": 39, "guj": "શહેરી વિકાસ અને શહેરી ગૃહ નિર્માણ વિભાગ (શહેરી વિકાસ અને શહેરી હાઉસિંગ)", "eng": "Urban Development & Urban Housing Department"},
    {"id": 41, "guj": "શિક્ષણ વિભાગ (પ્રાથમિક અને માધ્યમિક શિક્ષણ)", "eng": "Education Department (Primary & Secondary Education)"},
    {"id": 42, "guj": "શિક્ષણ વિભાગ (ઉચ્ચ અને ટેકનિકલ શિક્ષણ)", "eng": "Education Department (Higher & Technical Education)"},
    {"id": 43, "guj": "શ્રમ અને રોજગાર વિભાગ", "eng": "Labour & Employment Department"},
    {"id": 44, "guj": "સામાન્ય વહીવટ વિભાગ (ક.ગ.)", "eng": "General Administration Department"},
    {"id": 45, "guj": "સામાન્ય વહીવટ વિભાગ (આયોજન)", "eng": "General Administration Department (Planning)"},
    {"id": 48, "guj": "મુખ્યમંત્રીશ્રીનું કાર્યાલય", "eng": "CMO"},
    {"id": 49, "guj": "મુખ્ય સચિવશ્રીનું કાર્યાલય, ગુજરાત", "eng": "CS Office"},
    {"id": 50, "guj": "ભારતીય રાષ્ટ્રીય રાજમાર્ગ પ્રાધિકરણ", "eng": "national highway authority of india"},
    {"id": 51, "guj": "રેલવે", "eng": "Railways"},
]



DEPT_MAPPING = {
    1: "Food, Civil Supplies & Consumer Affairs Department",
    2: "Tribal Development Department",
    3: "Health & Family Welfare Department (Health Education)",
    5: "Industries & Mines Department",
    6: "Industries & Mines Department (Cottage & Rural Industries)",
    7: "Industries & Mines Department (Tourism, Pilgrimage)",
    8: "Energy & Petrochemicals Department",
    9: "Climate Change Department",
    10: "Legal Department",
    11: "Agriculture, Farmer Welfare & Co-operation Department",
    12: "Agriculture (Animal Husbandry, Cow Breeding, Fisheries, Co-operation)",
    13: "Home Department (Police, Law & Order)",
    15: "Narmada, Water Resources, Water Supply & Kalpasar (Irrigation)",
    16: "Narmada, Water Resources, Water Supply & Kalpasar (Narmada)",
    17: "Narmada, Water Resources, Water Supply & Kalpasar (Water Supply)",
    19: "Finance Department",
    22: "Panchayat, Rural Housing & Rural Development Department",
    23: "Panchayat (Rural Development)",
    24: "Ports & Transport Department",
    27: "Revenue Department (Land, Collector, Mamlatdar)",
    28: "Women & Child Development Department",
    29: "Road & Building Department (State Highways, Govt Buildings)",
    30: "Information & Broadcasting Department",
    31: "Sports, Youth & Cultural Activities Department",
    32: "Forest & Environment Department",
    33: "Legislative & Parliamentary Affairs Department",
    36: "Gujarat Legislative Assembly",
    37: "Science & Technology Department",
    38: "Social Justice & Empowerment Department",
    39: "Urban Development & Urban Housing Department",
    41: "Education Department (Primary & Secondary Education)",
    42: "Education Department (Higher & Technical Education)",
    43: "Labour & Employment Department",
    44: "General Administration Department",
    45: "General Administration Department (Planning)",
    48: "CMO (Chief Minister's Office)",
    49: "CS Office (Chief Secretary)",
    50: "National Highway Authority of India (NHAI)",
    51: "Railways"
}

PRABHAG_ENG_LIST = [item["eng"] for item in PRABHAG_DATA]

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

def _pick_prabhag(model_output: str):
    cleaned = (model_output or "").strip()

    # Since the new prompt outputs an ID, we use regex to extract the first number found
    match = re.search(r'\d+', cleaned)
    if match:
        extracted_id = int(match.group())
        # Find the matching department by ID in PRABHAG_DATA
        for item in PRABHAG_DATA:
            if item["id"] == extracted_id:
                return item

    # Default fallback if LLM gave an invalid response or no ID was found
    return PRABHAG_DATA[0]

def analyze_english_text_with_llm(text):
    """
    Input: English text (string)
    Output: (gujarati_category_string, is_govt_boolean, confidence_score, sentiment_string, prabhag_name, prabhag_id)
    """
    if not text or not text.strip():
        return "અન્ય", False, 0, "Neutral", None, None

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
    
    # --- 4. Check Prabhag (New Prompt with Hints) ---
    prabhag_name = None
    prabhag_id = None
    
    numbered_depts = "\n".join([f"{i+1}. {dept}" for i, dept in enumerate(PRABHAG_ENG_LIST)])

    DEPT_DESCRIPTIONS = {
        #1: "ration shops fair price shop ration card food grains public distribution consumer complaints",
        1: "ration shops fair price shop ration card food grains public distribution consumer complaints LPG PNG LNG gas cylinder distribution ",
        2: "tribal welfare tribal education tribal hostel tribal development schemes",
        3: "hospital doctor nurse phc chc medicine pharmacy food adulteration food license health services",
        5: "mining illegal sand mining quarry mineral extraction",
        6: "cottage industry handicraft rural industry village industry",
        7: "tourism pilgrimage temple tourism tourist facilities",
        8: "electricity gas cylinders power supply transformer electric pole electrocution power grid",
        9: "climate change environmental sustainability global warming",
        10: "legal department court cases government legal advice",
        11: "farmer agriculture crop fertilizer seeds crop insurance apmc",
        12: "animal husbandry dairy cattle breeding milk cooperative fisheries fishermen",
        13: "police crime theft robbery murder fir law and order traffic police",
        15: "irrigation canal irrigation water agricultural irrigation dam",
        16: "narmada project sardar sarovar dam narmada river project kalpsar",
        17: "drinking water pipeline water supply scheme water shortage",
        19: "gst tax commercial tax government revenue finance",
        22: "village / gram panchayat rural governance rural development mgnrega grameen",
        23: "rural housing village development",
        24: "bus gsrtc bus depot bus stand rto public transport",
        27: "land records 7-12 mamlatdar talati land dispute land survey",
        28: "anganwadi women welfare child nutrition",
        29: "road pothole bridge state highway road construction",
        30: "government publicity information broadcasting",
        31: "sports playground sports complex indoor games outdoor games khelo india",
        32: "forest environment pollution wildlife conservation",
        33: "parliamentary affairs legislative coordination",
        36: "legislative assembly proceedings",
        37: "science research technology innovation",
        38: "scholarship social welfare sc st obc welfare scheme",
        39: "municipal corporation garbage drainage sewer sanitation stray cattle fire safety",
        41: "primary school secondary school education teachers",
        42: "college university technical education higher education",
        43: "labour workers employment worker rights",
        44: "government recruitment gpsc administrative reforms",
        45: "government planning outcome budget district planning",
        50: "national highway nhai highway construction",
        51: "railway train railway station rail transport",
    }

    prompt_prabhag = (
            "You are an expert Gujarat government grievance classification system. "
            "Read the news text carefully and return the SINGLE department ID most responsible for resolving the issue.\n\n"

            "DEPARTMENT LIST (ID : Name : Keywords):\n" +
            "\n".join([f"{k} : {v} : {DEPT_DESCRIPTIONS.get(k, '')}" for k, v in DEPT_MAPPING.items()]) + "\n\n"

                                                                                                          "CLASSIFICATION RULES (apply in priority order):\n"
                                                                                                          "R1.  National highway / NHAI work or accident                          → 50\n"
                                                                                                          "R2.  State road / pothole / bridge / road contractor / PWD             → 29\n"
                                                                                                          "R3.  Railway / train services / station                                → 51\n"
                                                                                                          "R4.  GSRTC bus / ST bus / bus depot / RTO                              → 24\n"
                                                                                                          #"R5.  Ration shop / Fair Price Shop / PDS / ration card                 → 1\n"
                                                                                                          
"R5.  Ration shop / Fair Price Shop / PDS / ration card / LPG / Gas Cylinder / PNG / LNG                  → 1\n"
"R6.  7-12 / mamlatdar / talati / land record / land scam               → 27\n"
                                                                                                          "R7.  Farmer / crop / fertilizer / APMC / crop insurance                → 11\n"
                                                                                                          "R8.  Fishermen / animal husbandry / dairy / milk cooperative           → 12\n"
                                                                                                          "R9.  Hospital / doctor / PHC / CHC / medicine / food adulteration / food license     → 3\n"
                                                                                                          #"R10. Electricity / electric pole / electrocution / power supply / GEB / PNG gas → 8\n"
"R10. Electricity / electric pole / electrocution / power supply / GEB  → 8\n"
                                                                                                          
"R11. Scholarship / SC-ST-OBC welfare / social justice scheme           → 38\n"
                                                                                                          "R12. Primary / secondary school / teachers / textbooks                 → 41\n"
                                                                                                          "R13. College / university / MBA / technical education                  → 42\n"
                                                                                                          "R14. Anganwadi / women & child nutrition / bal vatika                  → 28\n"
                                                                                                          "R15. Municipal corporation / city garbage / city sewer / stray cattle / urban fire → 39\n"
                                                                                                          "R16. Village panchayat / rural development / MGNREGA / grameen         → 22\n"
                                                                                                          "R17. Sand mining / illegal mining / stone quarry                       → 5\n"
                                                                                                          "R18. Pollution / forest / wildlife conservation                        → 32\n"
                                                                                                          "R19. GST / commercial tax / revenue / finance                          → 19\n"
                                                                                                          "R20. Crime / murder / theft / robbery / law & order / traffic police   → 13\n"
                                                                                                          "R21. GPSC recruitment / government service / election / admin reforms  → 44\n"
                                                                                                          "R22. Drinking water pipeline / water supply scheme                     → 17\n"
                                                                                                          "R23. Irrigation canal / agricultural water                             → 15\n"
                                                                                                          "R24. Tribal welfare / tribal education                                 → 2\n\n"

                                                                                                          "CONFLICT RULES:\n"
                                                                                                          "- Urban road dug by municipality but not repaired → 39 (municipal issue, not 29)\n"
                                                                                                          "- Scholarship for tribal students → 38 (not 2)\n"
                                                                                                          "- Traffic congestion / traffic management in city → 13\n"
                                                                                                          "- Accident caused by bad state road → 29\n"
                                                                                                          "- Accident caused by bad national highway → 50\n"
                                                                                                          "- Rural sanitation / open defecation → 22; urban sanitation / sewer → 39\n"
                                                                                                          "- Crime at hospital/school/road: classify by the CORE issue (corruption in school=41, theft=13)\n\n"
                                                                                                          "- Shortage of LPG gas cylinders → 1 (not 8)\n\n"
                                                                                                           
                                                                                                          "EXAMPLES:\n\n"

                                                                                                          "Input: There has been increasing resentment among traders over the harassment by GST inspectors. "
                                                                                                          "Traders starting business with a new GST number are put to hardship by the GST department.\n"
                                                                                                          "Output: 19\n\n"

                                                                                                          "Input: Stray cattle are terrorising residential areas of Gandhinagar city. "
                                                                                                          "The problem is increasing day by day in the state capital.\n"
                                                                                                          "Output: 39\n\n"

                                                                                                          "Input: Poor quality ration is being distributed in Amraiwadi. Some ration shops are supplying "
                                                                                                          "adulterated paddy and plastic shredded rice to people from poorer sections.\n"
                                                                                                          "Output: 1\n\n"

                                                                                                          "Input: In Anand district the mamlatdar office e-Dhara has a no-work policy. "
                                                                                                          "Employees do not hesitate to take bribes. Deputy mamlatdars have been caught red-handed.\n"
                                                                                                          "Output: 27\n\n"

                                                                                                          "Input: Farmers in Saurashtra are protesting because they have not received the insurance money "
                                                                                                          "for their cotton crops destroyed by unseasonal rain.\n"
                                                                                                          "Output: 11\n\n"

                                                                                                          "Input: Serious irregularities are taking place in government schemes and subsidies given to "
                                                                                                          "fishermen and small boat owners in major ports including Okha and Porbandar.\n"
                                                                                                          "Output: 12\n\n"

                                                                                                          "Input: The condition of the State Highway connecting Rajkot and Jamnagar is pathetic. "
                                                                                                          "Big potholes are causing accidents daily.\n"
                                                                                                          "Output: 29\n\n"

                                                                                                          "Input: A memorandum was submitted regarding the disorganized bus system at Dharampur bus station. "
                                                                                                          "Complaints have been raised about non-arrival of GSRTC buses on several routes.\n"
                                                                                                          "Output: 24\n\n"

                                                                                                          "Input: The government hospital in Seemalia village has no surgeon or gynaecologist. "
                                                                                                          "Poor tribal people are forced to go to private hospitals or deliver at risk at home.\n"
                                                                                                          "Output: 3\n\n"

                                                                                                          "Input: Electrocution death reported after a high-tension electric pole collapsed in Bharuch. "
                                                                                                          "The GEB has not replaced the damaged transformer for three days.\n"
                                                                                                          "Output: 8\n\n"

                                                                                                          "Input: 1379 bicycles purchased under the Saraswati Sadhana Yojana have not been distributed "
                                                                                                          "to girl students due to fault of the authorities.\n"
                                                                                                          "Output: 38\n\n"

                                                                                                          "Input: Primary school in Waghai taluka has salary sanctioned for 6 teachers but only 2 are present. "
                                                                                                          "The headmaster is also frequently absent.\n"
                                                                                                          "Output: 41\n\n"

                                                                                                          "Input: Students and parents of an MBA college in Porbandar protested vehemently after the "
                                                                                                          "principal forgot to fill their examination forms.\n"
                                                                                                          "Output: 42\n\n"

                                                                                                          "Input: Anganwadi workers in Patan district have not received their honorarium for 4 months. "
                                                                                                          "Children's nutrition supplements are also not being distributed.\n"
                                                                                                          "Output: 28\n\n"

                                                                                                          "Input: Piles of garbage have accumulated in the eastern zone of the city. "
                                                                                                          "The Municipal Corporation has failed to collect waste for a week.\n"
                                                                                                          "Output: 39\n\n"

                                                                                                          "Input: Unknown mafias are running illegal sand-laden vehicles from the Banas river bed in Deesa. "
                                                                                                          "Villagers demand a ban on sand mining vehicles.\n"
                                                                                                          "Output: 5\n\n"

                                                                                                          "Input: The GPSC Committee Board submitted a memorandum demanding action against casteist elements "
                                                                                                          "in the government recruitment system violating SC/ST constitutional rights.\n"
                                                                                                          "Output: 44\n\n"

                                                                                                          "Input: A jeweller was looted at gunpoint in the market area. The police arrived late.\n"
                                                                                                          "Output: 13\n\n"

                                                                                                          "Input: Traffic jams at Rajkot Road near Subhash Bridge are a daily affair due to heavy vehicles "
                                                                                                          "crossing. Residents demand smooth traffic regulation.\n"
                                                                                                          "Output: 13\n\n"

                                                                                                          "Input: Food adulteration found in Mahadev Trading Company — non-permitted synthetic colorants "
                                                                                                          "detected in red chili powder along with wheat and rice starch.\n"
                                                                                                          "Output: 3\n\n"

                                                                                                          "Input: In Surat's Varachha zone the Pay and Park contract has ended but the new tendering process "
                                                                                                          "is delayed, causing daily revenue loss to the Municipal Corporation.\n"
                                                                                                          "Output: 39\n\n"

                                                                                                          "Input: On the main road in Patan town a municipality sewer line was laid but potholes were not "
                                                                                                          "repaired after work completion, causing accidents to bikers at night.\n"
                                                                                                          "Output: 39\n\n"

                                                                                                          "Output Requirement: Return ONLY the department ID number — a single integer (e.g. 19)."

                                       f"INPUT TEXT:\n'{text}'\n\n"
                                       "PREDICTED ID: "

)
    
    try:
        resp_prabhag = _call_vllm(prompt_prabhag, json_mode=False)
        picked_prabhag = _pick_prabhag(resp_prabhag)

        if picked_prabhag:
            prabhag_name = picked_prabhag["eng"]
            prabhag_id = picked_prabhag["id"]
    except Exception as e:
        print(f"⚠️ Prabhag Classification Error: {e}")

    return guj_category, is_govt_bool, confidence, sentiment_str, prabhag_name, prabhag_id


def classify_personal_tragedy_crime(text):
    """
    Classifies text into '1' (Personal Tragedies & Resolved Crimes) or '0' (Systemic/Other).
    Returns: int (1 or 0)
    """
    if not text or len(str(text)) < 5:
        return 0

    # TRUNCATE TEXT to 2048 characters to prevent Token Limit Error
    truncated_text = str(text)[:2048]

    system_prompt = (

        "You are a strict news classifier. Analyze the provided news text and output ONLY '1' or '0'.\n\n"

        "CLASSIFICATION RULES:\n"
        "Return '1' (BLOCK THIS CONTENT) if ANY of the following are true:\n"
        "1. ROAD ACCIDENTS: Any vehicle collisions or traffic-related mishaps.\n"
        "2. RESOLVED/ACTIONED INCIDENTS: Any crime or theft or incident where authorities have ALREADY taken action "
        "(e.g., 'Police arrested', 'FIR registered', 'Case filed', 'Accused caught', 'Probe ordered', 'Court judgement').\n"
        "3. SUICIDES: Suicides due to personal reasons (family disputes, love affairs, exams, debt, depression) by consuming poison, hanging, jumping from building/bridge "
        "that DO NOT involve allegations against the State Government or officials.\n"
        "4. DEATHS: Death of a person due to natural causes like heart attack/ailments that do not require State Government involvement.\n"
        "5. PRIVATE/CIVIL DISPUTES & NOTICES: Public notices, missing persons, or cyber fraud where a case is already registered.\n\n"

        "Return '0' (KEEP THIS CONTENT) if:\n"
        "1. GOVERNMENT/SYSTEMIC FAULT: Suicides or tragedies blamed on government negligence, official harassment, court orders for government agencies or policy failure.\n"
        "2. UNRESOLVED CRIMES: Heinous crimes or law & order situations where NO police action/arrest is mentioned yet.\n"
        "3. OTHER: General negative news that requires prompt action from government.\n\n"

        "EXAMPLES:\n\n"

        "Input: A tragic road accident occurred on the highway where a truck collided with a motorcycle, resulting in the death of the rider.\n"
        "Output: 1\n\n"

        "Input: The city police have arrested the main accused involved in the bank robbery. An FIR has been registered and the investigation is ongoing.\n"
        "Output: 1\n\n"

        "Input: Distressed over family disputes and mounting personal debt, a 35-year-old man committed suicide by consuming poison at his residence.\n"
        "Output: 1\n\n"

        "Input: Two groups fought in public over a parking issue. One person attacked another with a sharp weapon. Police reached the spot and nabbed the attackers.\n"
        "Output: 1\n\n"

        "Input: A software engineer lost Rs 50,000 due to online cyber fraud. A case has been filed with the Cyber Crime cell.\n"
        "Output: 1\n\n"

        "Input: A senior citizen passed away this morning due to a sudden heart attack while returning from the market.\n"
        "Output: 1\n\n"

        "Input: A complaint has been filed regarding a teenager who went missing yesterday evening.\n"
        "Output: 1\n\n"

        "Input: This Public Notice informs the public and all concerned, if any person, institution, or any person has a dispute with respect to the sale, mortgage, or any other kind of encumbrance of the said property, please contact the undersigned.\n"
        "Output: 1\n\n"

        "Input: Villagers are protesting against the electricity department, alleging that a loose wire caused the death of three farmers. No officials have visited the spot yet.\n"
        "Output: 0\n\n"

        "Input: A jeweller was shot dead in broad daylight by unidentified assailants. The attackers escaped and police is yet to take any action yet.\n"
        "Output: 0\n\n"

        "Input: A farmer committed suicide, leaving a note blaming the Tehsildar for demanding bribes and harassment regarding land records.\n"
        "Output: 0\n\n"

        "Output Requirement: Return strictly ONLY the number '1' or '0'. No explanation."

    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"News Text: {truncated_text}"}
        ],
        "temperature": 0.0,
        "max_tokens": 10
    }

    try:
        response = requests.post(VLLM_URL, json=payload, timeout=30)

        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content'].strip()
            # Simple check for 1 or 0
            if "1" in content:
                return 1
            if "0" in content:
                return 0
            return 0
        else:
            print(f"⚠️ Tragedies/Crime Classifier Status {response.status_code}")
            return 0

    except Exception as e:
        print(f"⚠️ Tragedies/Crime Classifier Error: {e}")
        return 0
