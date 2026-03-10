import requests
import json
import re
from difflib import get_close_matches

# --- Configuration ---
VLLM_URL = "http://localhost:8100/v1/chat/completions"
MODEL_NAME = "./qwen-14b-awq"

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

# --- UPDATED CONFIGURATION: PRABHAG DATA FROM IMAGE (IDs 1-49) ---
# PRABHAG_DATA = [
#     {"id": 1, "guj": "અન્ન નાગરિક પુરવઠો અને ગ્રાહક બાબતોનો વિભાગ", "eng": "Food, Civil Supplies & Consumer Affairs Department"},
#     {"id": 2, "guj": "આદિજાતિ વિકાસ વિભાગ", "eng": "Tribal Development Department"},
#     {"id": 3, "guj": "આરોગ્ય અને પરિવાર કલ્યાણ વિભાગ (આરોગ્ય શિક્ષણ)", "eng": "Health & Family Welfare Department (Health)"},
#     {"id": 4, "guj": "આરોગ્ય અને પરિવાર કલ્યાણ વિભાગ (જાહેર આરોગ્ય અને પરિવાર કલ્યાણ)", "eng": "Health & Family Welfare Department (Public Health & Family Welfare)"},
#     {"id": 5, "guj": "ઉદ્યોગ અને ખાણ વિભાગ", "eng": "Industries & Mines Department"},
#     {"id": 6, "guj": "ઉદ્યોગ અને ખાણ વિભાગ (કુટીર ઉદ્યોગ અને ગ્રામોદ્યોગ)", "eng": "Industries & Mines Department (Cottage & Rural Industries)"},
#     {"id": 7, "guj": "ઉદ્યોગ અને ખાણ વિભાગ (પ્રવાસન, યાત્રા પ્રવાસ, દેવસ્થાન સં.)", "eng": "Industries & Mines Department (Tourism, Pilgrimage, Devsthanam Management)"},
#     {"id": 8, "guj": "ઉર્જા અને પેટ્રો કેમિકલ્સ વિભાગ", "eng": "Energy & Petrochemicals Department"},
#     {"id": 9, "guj": "ક્લાઇમેટ ચેન્જ વિભાગ", "eng": "Climate change Department"},
#     {"id": 10, "guj": "કાયદા વિભાગ", "eng": "Legal Department"},
#     {"id": 11, "guj": "કૃષિ, ખેડૂત કલ્યાણ અને સહકાર વિભાગ", "eng": "Agriculture, Farmer Welfare & Co-operation Department"},
#     {"id": 12, "guj": "કૃષિ, ખેડૂત કલ્યાણ અને સહકાર વિભાગ (પશુપાલન, ગૌસંવર્ધન, મત્સ્યોદ્યોગ, સહકાર)", "eng": "Agriculture, Farmer Welfare & Co-operation Department (Animal Husbandry, Cow Breeding, Fisheries, Co-operation)"},
#     {"id": 13, "guj": "ગૃહ વિભાગ", "eng": "Home Department"},
#     {"id": 14, "guj": "નર્મદા, જળ સંપત્તિ, પાણી પુરવઠો અને કલ્પસર વિભાગ (પુન: વસવાટ)", "eng": "Narmada, Water Resourses, Water Supply & Kalpasar Department (Rehabilitation)"},
#     {"id": 15, "guj": "નર્મદા, જળ સંપત્તિ, પાણી પુરવઠો અને કલ્પસર વિભાગ (સિંચાઈ)", "eng": "Narmada, Water Resourses, Water Supply & Kalpasar Department (Irrigation)"},
#     {"id": 16, "guj": "નર્મદા, જળ સંપત્તિ, પાણી પુરવઠો અને કલ્પસર વિભાગ (નર્મદા)", "eng": "Narmada, Water Resourses, Water Supply & Kalpasar Department (Narmada)"},
#     {"id": 17, "guj": "નર્મદા, જળ સંપત્તિ, પાણી પુરવઠો અને કલ્પસર વિભાગ (પાણી પુરવઠા)", "eng": "Narmada, Water Resourses, Water Supply & Kalpasar Department (Water Supply)"},
#     {"id": 18, "guj": "નર્મદા, જળ સંપત્તિ, પાણી પુરવઠો અને કલ્પસર વિભાગ (કલ્પસર)", "eng": "Narmada, Water Resourses, Water Supply & Kalpasar Department (Kalpasar)"},
#     {"id": 19, "guj": "નાણાં વિભાગ", "eng": "Finance Department"},
#     {"id": 20, "guj": "નાણાં વિભાગ (આર્થિક બાબતો)", "eng": "Finance Department (Economical Affairs)"},
#     {"id": 21, "guj": "નાણાં વિભાગ (ખર્ચ)", "eng": "Finance Department (Expenses)"},
#     {"id": 22, "guj": "પંચાયત, ગ્રામ ગૃહનિર્માણ અને ગ્રામ વિકાસ વિભાગ", "eng": "Panchayat, Rural Housing & Rural Development Department"},
#     {"id": 23, "guj": "પંચાયત, ગ્રામ ગૃહનિર્માણ અને ગ્રામ વિકાસ વિભાગ (ગ્રામ વિકાસ)", "eng": "Panchayat, Rural Housing & Rural Development Department (Rural Development)"},
#     {"id": 24, "guj": "બંદરો અને વાહન વ્યવહાર વિભાગ", "eng": "Ports & Transport Department"},
#     {"id": 25, "guj": "બંદરો અને વાહન વ્યવહાર વિભાગ (વાહન વ્યવહાર)", "eng": "Ports & Transport Department (Transport)"},
#     {"id": 26, "guj": "બંદરો અને વાહન વ્યવહાર વિભાગ (બંદરો)", "eng": "Ports & Transport Department (Ports)"},
#     {"id": 27, "guj": "મહેસૂલ વિભાગ", "eng": "Revenue Department"},
#     {"id": 28, "guj": "મહિલા અને બાળ કલ્યાણ વિકાસ વિભાગ", "eng": "Women & Child Development Department"},
#     {"id": 29, "guj": "માર્ગ અને મકાન વિભાગ", "eng": "Road & Building Department"},
#     {"id": 30, "guj": "માહિતી અને પ્રસારણ વિભાગ", "eng": "Information & Broadcasting Department"},
#     {"id": 31, "guj": "રમતગમત, યુવા અને સાંસ્કૃતિક પ્રવૃત્તિ વિભાગ", "eng": "Sports, Youth & Cultural Activities Department"},
#     {"id": 32, "guj": "વન અને પર્યાવરણ વિભાગ", "eng": "Forest & Environment Department"},
#     {"id": 33, "guj": "વૈધાનિક અને સંસદીય બાબતોનો વિભાગ (વૈધાનિક અને સંસદીય બાબતો)", "eng": "Legislative & Parliamentary Affairs Department"},
#     {"id": 34, "guj": "વૈધાનિક અને સંસદીય બાબતોનો વિભાગ (વૈધાનિક બાબતો)", "eng": "Legislative & Parliamentary Affairs Department (Legislative Affairs)"},
#     {"id": 35, "guj": "વૈધાનિક અને સંસદીય બાબતોનો વિભાગ (સંસદીય બાબતો)", "eng": "Legislative & Parliamentary Affairs Department (Parliamentary Affairs)"},
#     {"id": 36, "guj": "ગુજરાત વિધાનસભા", "eng": "Legislative & Parliamentary Affairs Department (Gujarat Legislative Assembly)"},
#     {"id": 37, "guj": "વિજ્ઞાન અને પ્રૌદ્યોગિકી વિભાગ", "eng": "Science & Technology Department"},
#     {"id": 38, "guj": "સામાજિક ન્યાય અને અધિકારીતા વિભાગ", "eng": "Social Justice & Empowerment Department"},
#     {"id": 39, "guj": "શહેરી વિકાસ અને શહેરી ગૃહ નિર્માણ વિભાગ (શહેરી વિકાસ અને શહેરી હાઉસિંગ)", "eng": "Urban Development & Urban Housing Department"},
#     {"id": 40, "guj": "શહેરી વિકાસ અને શહેરી ગૃહ નિર્માણ વિભાગ (શહેરી ગૃહ નિર્માણ અને નિર્મળ ગુજરાત)", "eng": "Urban Development & Urban Housing Department (Housing)"},
#     {"id": 41, "guj": "શિક્ષણ વિભાગ (પ્રાથમિક અને માધ્યમિક શિક્ષણ)", "eng": "Education Department (Primary & Secondary Education)"},
#     {"id": 42, "guj": "શિક્ષણ વિભાગ (ઉચ્ચ અને ટેકનિકલ શિક્ષણ)", "eng": "Education Department (Higher & Technical Education)"},
#     {"id": 43, "guj": "શ્રમ અને રોજગાર વિભાગ", "eng": "Labour & Employment Department"},
#     {"id": 44, "guj": "સામાન્ય વહીવટ વિભાગ (ક.ગ.)", "eng": "General Administration Department (Personnel)"},
#     {"id": 45, "guj": "સામાન્ય વહીવટ વિભાગ (આયોજન)", "eng": "General Administration Department (Planning)"},
#     {"id": 46, "guj": "સામાન્ય વહીવટ વિભાગ (વ.સુ.તા.પ્ર. અને એન.આર.આઈ.)", "eng": "General Administration Department (A.R.T.D. and NRI)"},
#     {"id": 47, "guj": "સામાન્ય વહીવટ વિભાગ (ચૂંટણી)", "eng": "General Administration Department (Election)"},
#     {"id": 48, "guj": "મુખ્યમંત્રીશ્રીનું કાર્યાલય", "eng": "CMO"},
#     {"id": 49, "guj": "મુખ્ય સચિવશ્રીનું કાર્યાલય, ગુજરાત", "eng": "CS Office"}
# ]

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

# def _pick_prabhag(model_output: str):
#     cleaned = (model_output or "").strip()
#     norm_lower = cleaned.lower()
#
#     # Direct Matching
#     for item in PRABHAG_DATA:
#         if norm_lower == item["eng"].lower():
#             return item
#
#     # Fuzzy Matching
#     matches = get_close_matches(norm_lower, [d.lower() for d in PRABHAG_ENG_LIST], n=1, cutoff=0.6)
#     if matches:
#         for item in PRABHAG_DATA:
#             if item["eng"].lower() == matches[0]:
#                 return item
#
#     return PRABHAG_DATA[0] # Default fallback

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

    prompt_prabhag = (
        # f"TASK: You are a Gujarat Government department classifier. Your job is to classify the given text to exactly ONE department from the numbered list below.\n\n"
        # f"STRICT OUTPUT RULES:\n"
        # f"1. Output ONLY the exact department name from the list - nothing else\n"
        # f"2. Copy the department name character-by-character exactly as written\n"
        # f"3. NO explanations, NO reasoning, NO additional text\n"
        # f"4. NO department numbers or IDs in output\n"
        # f"5. Output must be in English only\n"
        # f"6. Do NOT translate or modify the department name\n"
        # f"7. Output on a single line only\n\n"
        # f"DEPARTMENT CLASSIFICATION HINTS:\n"
        # f"- Police, crime, security, law & order → Home Department\n"
        # f"- Schools, primary education → Education Department (Primary & Secondary Education)\n"
        # f"- Colleges, universities, technical → Education Department (Higher & Technical Education)\n"
        # f"- Farmers, crops, agriculture → Agriculture, Farmer Welfare & Co-operation Department\n"
        # f"- Cattle, dairy, fishing → Agriculture, Farmer Welfare & Co-operation Department (Animal Husbandry, Cow Breeding, Fisheries, Co-operation)\n"
        # f"- Roads, bridges, causeway → Road & Building Department\n"
        # f"- Electricity, power, petroleum → Energy & Petrochemicals Department\n"
        # f"- Water supply → Narmada, Water Resourses, Water Supply & Kalpasar Department (Water Supply)\n"
        # f"- Irrigation, canals → Narmada, Water Resourses, Water Supply & Kalpasar Department (Irrigation)\n"
        # f"- Dams, Narmada river → Narmada, Water Resourses, Water Supply & Kalpasar Department (Narmada)\n"
        # f"- Women, children, ICDS, anganwadi → Women & Child Development Department\n"
        # f"- Hospitals, doctors, medical → Health & Family Welfare Department (Public Health & Family Welfare)\n"
        # f"- Medical education, nursing → Health & Family Welfare Department (Health)\n"
        # f"- Budget, finance, taxation → Finance Department\n"
        # f"- Village, panchayat, gram sabha → Panchayat, Rural Housing & Rural Development Department\n"
        # f"- City, municipality, urban → Urban Development & Urban Housing Department\n"
        # f"- Forests, wildlife → Forest & Environment Department\n"
        # f"- Industries, factories, manufacturing, GIDC → Industries & Mines Department\n"
        # f"- Tourism, pilgrimage, temples → Industries & Mines Department (Tourism, Pilgrimage, Devsthanam Management)\n"
        # f"- Ports, ships, maritime → Ports & Transport Department (Ports)\n"
        # f"- Vehicles, RTO, transport → Ports & Transport Department (Transport)\n"
        # f"- Sports, youth, culture → Sports, Youth & Cultural Activities Department\n"
        # f"- Land revenue, collector → Revenue Department\n"
        # f"- Ration, PDS, consumer → Food, Civil Supplies & Consumer Affairs Department\n"
        # f"- Labour, workers, employment → Labour & Employment Department\n"
        # f"- SC/ST welfare, disabled → Social Justice & Empowerment Department\n"
        # f"- Chief Minister related → CMO\n"
        # f"- Chief Secretary related → CS Office\n"
        # f"- Media, press, broadcasting → Information & Broadcasting Department\n"
        # f"- Climate, carbon, green initiatives → Climate change Department\n"
        # f"- Assembly, legislature → Legislative & Parliamentary Affairs Department\n"
        # f"- Election, voting, transfer → General Administration Department (Election)\n"
        # f"- Science, research, technology → Science & Technology Department\n"
        # f"- Legal, courts, judiciary → Legal Department\n\n"
        # f"NUMBERED DEPARTMENT LIST:\n"
        # f"{numbered_depts}\n\n"
        # f"TEXT TO CLASSIFY:\n"
        # f"\"\"\"{text}\"\"\"\n\n"
        # f"DEPARTMENT NAME:"

            "You are an intelligent government grievance router. Analyze the provided news text and map it to the most relevant Government Department ID from the list below.\n\n"

            "DEPARTMENT LIST (ID : Name):\n" +
            "\n".join([f"{k} : {v}" for k, v in DEPT_MAPPING.items()]) + "\n\n"

             "CLASSIFICATION RULES:\n"
             "- Identify the core issue needing resolution.\n"
             "- *Road / state highway / bridge / road contractor*  -> Map to ID 29"
             "- *Farmers / Fertilizers / Crops / APMC / Cooperative* -> Map to ID 11 (Agriculture).\n"
             "- *Pollution*-> Map to ID 32 (Forest & Environment Department)\n"
             "- *GST / Taxes / Commercial Tax* -> Map to ID 19 (Finance).\n"
             "- *ST Buses / Bus Depots / GSRTC / RTO* -> Map to ID 24 (Ports & Transport).\n"
             "- *Welfare Schemes / Scholarships / Bicycles for poor or backward classes* -> Map to ID 38 (Social Justice).\n"
             "- *Ration Shops / Fair price shop* -> Map to ID 1.\n"
             "- *Land Titles / 7-12 / Mamlatdar / Land Scams / Talati / Land survey* -> Map to ID 27.\n"
             "- *Elections / GPSC Recruitment / Government personnel related / Administrative Reforms and Training / NRI* -> Map to ID 44 (General Admin).\n"
             "- *Preparation of Outcome Budget / Decentralized District Planning* -> Map to ID 45 (General Admin Planning).\n"
             "- *National Highways* (ID 50) vs *State Roads* (ID 29).\n"
             "- *Urban/City, municipal corporation, Stray cattle, municipality, fire related accidents* -> Map to ID 39"
             "- *Panchayat, Village/Rural, MGNREGA, Rural employment scheme* -> Map to ID 22.\n"
             "- *Oil, PNG, gas, GEB, power grid, electrocution, electric pole, power supply, High-tension power lines* -> Map to ID 8 (Energy).\n"
             "- *Medical, Doctor, Pharmacy, Civil Hospitals, CHC/PHC, Drugs, Food adulteration* -> Map to ID 3 (Health).\n"
             "- *Mining, land mafia, Illegal sand mining* -> Map to ID 5 (Industries & Mines).\n"
             "- *Anganwadi, Women Child Development, nutrition* -> Map to ID 28 (Women & Child Development Department).\n"
             "- **Heavy vehicles, traffic, vehicles, crime, murder, theft -> Home Department (Police, Law & Order). \n"
             "- Output strictly ONLY the ID number.\n\n"

             "EXAMPLES:\n\n"

             "Input: There has been increasing resentment among traders over the harassment of inspectors, officers of the GST department for some time. There have been allegations that traders, especially those keen on starting business with a new GST number, are being put to hardship by the GST department's lackadaisical approach.\n"
             "Output: 19\n\n"

             "Input: The ST bus stand in Umra, built in 2014, has not been renovated even after 10 years of its inauguration. Necessary facilities for the passengers in the bus stand such as reservations, arrangements for passes, gift houses, print shops, etc., are in a dilapidated condition. The roof of the bus stand is also in a dilapidated condition, leading to the risk of accidents.\n"
             "Output: 24\n\n"

             "Input: Despite having stoppages on the Petlad town to Nadiad road, some ST bus operators do not allow ST buses to be parked at both the entry and exit points, causing inconvenience to commuters. There have been murmurs that the buses do not stop and run even if there are a couple of passengers at the station.\n"
             "Output: 24\n\n"

             "Input: 1379 bicycles purchased 10 years ago under the Saraswati Sadhana Yojana have not been distributed on time due to the fault of the authorities, the girl students have been deprived of the scheme and the expenditure made on the purchase of bicycles has proved to be a waste today.\n"
             "Output: 38\n\n"

             "Input: Scholarships from the government can be considered a blessing for economically weaker students. However, there have been widespread complaints from students about the amount not being credited to their accounts despite the 2022-23 scholarship being a holiday.\n"
             "Output: 38\n\n"

             "Input: Poor quality ration being distributed to the poor in Amraiwadi. There have been allegations that some ration shops in Amraiwadi were supplying items, including paddy, to people from poorer sections by adulterating them. Plastic shredded rice and low-quality salt are distributed.\n"
             "Output: 1\n\n"

             "Input: Consumers who do not get e-KYC done are angry at the closure of food grains. The shopkeepers are now in a worrying situation as the government has forced the shopkeepers of cheap grains (Fair Price Shop) to distribute the cheap grains of the coming May and June in the month of May. On the one hand, there are clashes between shopkeepers and customers as the names of poor beneficiaries of eKYC are missing from ration cards.\n"
             "Output: 1\n\n"

             "Input: With Mundra-Baroi in Kutch district getting the status of a joint municipality, an estimated Rs 100 crore land scam surfaced in both areas. And after the investigation into the matter, the Revenue Department is still showing sluggishness in handing over the scam lands to the municipality.\n"
             "Output: 27\n\n"

             "Input: In Anand district, the e-Dhara dam owned by the mamlatdar office has a no-work policy. In which the employees do not hesitate to take bribes. In the past, deputy mamlatdars and employees have been caught red-handed taking bribes in many offices.\n"
             "Output: 27\n\n"

             "Input: The GPSC Committee Board of Gujarat has submitted a memorandum to the Governor, Chief Minister, demanding justice by taking action against the casteist and SC / ST indigenous community in the government's recruitment system in violation of the constitutional rights of 5 SC, ST, OBC communities. Registering FIRs and taking action on incidents of atrocities against Dalit and Adivasi communities. It also demanded that the investigation be carried out in a fair and impartial manner and not hand over the investigation to a racist motivated officer during the investigation and all the recruitment process of the GPSC be investigated by an independent high-level committee and strict action be taken by holding the racist elements accountable.\n"
             "Output: 44\n\n"

             "Input: After the submission of 996 nomination papers in the general election for sarpanch and 2817 in the ward in the village panchayats of Sabarkantha district, the verification of the candidate papers was held on Tuesday in the presence of the candidates at the office of the mamlatdar of eight taluks as well as at the taluka panchayat office. But the system of spending crores of rupees on elections has failed to provide time for scrutiny of nomination papers. There was no arrangement for water and fans in the scorching heat and the candidates were forced to sit on the floor while the nomination papers were being scrutinized at the Khedbrahma mamlatdar's office.\n"
             "Output: 44\n\n"

             "Input: The cloudy weather and scattered rains throughout the day have caused much distress to farmers. Unseasonal rains have caused heavy loss of crops to the farmers leaving farmers worried.\n"
             "Output: 11\n\n"

             "Input: A case of permanent and temporary embezzlement has come to light in Laloda of Idar taluka of Sabarkantha district, in which an audit of millions of rupees in a milk producer cooperative society by the registrar's office has revealed the embezzlement of this entire scam. A complaint has been registered against the two men at the Eder police station and both the accused have been exposed. The Laloda Milk Producers Cooperative located in Eder taluka.\n"
             "Output: 11\n\n"

             "Input: Farmers in Saurashtra are protesting because they have not received the insurance money for their cotton crops destroyed by unseasonal rain.\n"
             "Output: 11\n\n"

             "Input: Anapur Chhota village in Dhanera taluka of Banaskantha district, where farmers have threatened to agitate if action is not taken against malpractices, has now become the center of discussion due to serious allegations of corruption. Shocking visuals have emerged of the government sending crop failure assistance to a farmer battling natural calamities instead of reaching the needy farmers. The farmers allege that the aid here has been distributed on the basis of money settings and not on the basis of need. In many cases, the online forms of farmers have not been filled, due to which they have not been entitled to government assistance.\n"
             "Output: 11\n\n"

             "Input: The agents, instead of improving the condition of the beat owners for the welfare and economic empowerment of the plantations and fishermen, being exploited by the authorities, have been rightfully entitled to the benefits of the schemes instituted. Serious irregularities are taking place in various government schemes and subsidies given to fishermen and small boat owners in major ports of the state, including Okha Porbandar port.\n"
             "Output: 12\n\n"

             "Input: The condition of the State Highway connecting Rajkot and Jamnagar is pathetic. Big potholes are causing accidents daily.\n"
             "Output: 29\n\n"

             "Input: The age-old building of MD Science College, the only grant-in-aid science college in Porbandar district, has become dilapidated. As the building became dilapidated, so many students studying here were at risk, so this building was required to be tested and the entire building was sealed by the Municipal Corporation. At the same time, the education of many students studying in this college has been interrupted.\n"
             "Output: 42\n\n"

             "Input: The first batch of students of Swaminarayan J N Parel MBA College in Porbandar started their MBA exams today. Today, the students and parents reached the college and protested vehemently. The principal admitted his mistake and said that he forgot to fill the examination forms of the students.\n"
             "Output: 42\n\n"

             "Input: The Government of Gujarat is spending crores of rupees on massive campaigns like Girl Child Education. But the reality paints a different picture. Seeing the deteriorating condition of education in Dagdianba Primary School in Waghai taluka of Dang district. Because there is a salary sanctioned for 6 teachers and there are only 02 teachers present. Serious allegations are being made by the locals that the headmaster, Acharya, is also frequently absent.\n"
             "Output: 41\n\n"

             "Input: In 28 Vansda taluka, children from tribal areas are facing severe constraints in educational facilities. Bal Vatika Class 1 and Class 2 students have not yet been provided with second-grade textbooks while Bal Vatika children have been denied uniforms and scholarships. Today, sarpanches of Vansda taluka have submitted a memorandum to the administration regarding these issues.\n"
             "Output: 41\n\n"

             "Input: The transfer of two teachers to Babsar Primary School in Wadali taluka has caused great resentment among the villagers. On learning of the transfer orders of the teachers, on Friday, the villagers of Babsar village, parents and people associated with the school management submitted a written representation to the District Education Officer.\n"
             "Output: 41\n\n"

             "Input: A jeweller was looted at gunpoint in the market area. The police arrived late and the thieves escaped.\n"
             "Output: 13\n\n"

             "Input: Piles of garbage have accumulated in the eastern zone of the city. The Municipal Corporation has failed to collect waste for a week.\n"
             "Output: 39\n\n"

             "Input: The government hospital is located in Seemalia village of Ghoghamba and it is the government's intention to benefit the poor and tribal people of the area. This government hospital has been built at the expense of the government to benefit the tribal people but there is no surgeon or gynaecologist here. Therefore, these tribal poor people have to go to other private hospitals or take delivery at risk at home.\n"
             "Output: 3\n\n"

             "Input: In a serious case of food adulteration in a wholesale shop named Mahadev Trading Company located in Asura Jhapa area of Dharampur taluka of the district, a strict and warning verdict has been given by the court. Laboratory investigations revealed the presence of nonpermitted synthetic oil soluble colorants in red chili powder as well as extraneous food ingredients such as wheat and rice starch.\n"
             "Output: 3\n\n"

             "Input: An alarming incident has been reported in Dahod city regarding the hygiene and food safety of frost food. There was a ruckus at a shop opposite the city bus stand when a customer ordered a pizza. The customer sitting to eat pizza was shocked when he saw the worm in the pizza slice. As soon as the information of the incident was received, the team of Food Department of Dahod Municipality reached the spot.\n"
             "Output: 3\n\n"

             "Input: The delay in action by the Deesa administration has raised questions. 25 Unknown mafias seem to have become unruly in the Deesa area. There is a lot of resentment among the villagers around the sand-laden vehicles on the old Deesa Vasna road. There are many minor accidents due to sand-laden vehicles on the old Deesa Vasna road, which is demanding a ban on sand-laden vehicles from the Banas river bed.\n"
             "Output: 5\n\n"

             "Input: A memorandum was submitted to the depot manager by the Akhil Bharatiya Vidyarthi Parishad Dharampur unit regarding the issues of the long-standing disorganized bus system at the bus station in Valsad Dharampur town. In particular, complaints have been raised about the non-arrival of buses on Dharampur Kaprada, Dharampur Aranai and other routes.\n"
             "Output: 24\n\n"

             "Input: Stray cattle have been terrorising residential areas, including main and interior roads, in rural areas of Gandhinagar city and the corporation area for quite some time now, causing panic among the local residents. The movement of cattle in the settlement area throughout the day has created an atmosphere of fear among the residents. There have been calls for the system to be overhauled. The problem of stray cattle is increasing day by day in the state capital and the surrounding rural areas. As a result, the condition of the residents living in the settlement area, including the main and inner Magi, has become miserable.\n"
             "Output: 39\n\n"

             "Input: Traffic jams at various places during the crossing between Rajkot Road near Subhash Bridge in the city with a large number of vehicles plying every day have become a daily affair. There is a problem of long jams especially at Mahaprabhu's Seat and Lalwadi Road Crossing during the crossing of heavy vehicles. If a system of smooth traffic regulation is put in place on the respective routes, partial relief can be achieved.\n"
             "Output: 13\n\n"

             "Input: On the main road leading to Dukhwada in Patan town, a sewer line was constructed by the municipality. But after the completion of the work, the potholes are not repaired properly, which has proved fatal for the motorists. According to the locals, the contractor has been satisfied only by pouring soil. As a result, many bikers are taking a nap as the potholes are not visible in the darkness of the night.\n"
             "Output: 39"

             "Input:  the municipality is losing a huge amount of revenue every day. The issue of the delay in the new tendering process, despite the completion of the Pay and Park contract in the Varachha zone of the Surat Municipal Corporation, has now heated up. In an aggressive tone, AAP corporator Vipul Suhagia said the delay was causing a loss of lakhs of rupees to the Surat Municipal Corporation's exchequer. Suhagia has written a letter to the municipal commissioner in this regard, raising questions on the administration. He has alleged that there was a deliberate delay in tendering. The municipal corporation's loss of lakhs to the exchequer is public money that is being squandered by the administration. "
             "Output: 39"

             "Output Requirement: Return strictly ONLY the ID number (e.g., '19', '38', '24', '44')."
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

        #     "You are a strict news classifier. Analyze the provided news text and output ONLY '1' or '0'.\n\n"
    #     "CLASSIFICATION RULES:\n"
    #     "Return '1' (BLOCK THIS CONTENT) if ANY of the following are true:\n"                                                                                                                   "1. ROAD ACCIDENTS: Any vehicle collisions or traffic-related mishaps.\n"                                                                                                               "2. RESOLVED/ACTIONED INCIDENTS: Any crime or theft or incident where authorities have ALREADY taken action "
    #     "(e.g., 'Police arrested', 'FIR registered', 'Case filed', 'Accused caught', 'Probe ordered', 'Court judgement').\n"                                                                    "3. SUICIDES: Suicides due to personal reasons (family disputes, love affairs, exams, debt, depression) by consuming poison, hanging, jumping from building/bridge "
    #     "that DO NOT involve allegations against the State Government or officials.\n"                                                                                                          "4. DEATHS: Death of a person due to natural causes like heart attack/ailments that do not require State Government involvement.\n"                                                     "5. PRIVATE/CIVIL DISPUTES & NOTICES: Public notices, missing persons, or cyber fraud where a case is already registered.\n\n"
    #     "Return '0' (KEEP THIS CONTENT) if:\n"
    #     "1. GOVERNMENT/SYSTEMIC FAULT: Suicides or tragedies blamed on government negligence, official harassment, court orders for government agencies or policy failure.\n"                   "2. UNRESOLVED CRIMES: Heinous crimes or law & order situations where NO police action/arrest is mentioned yet.\n"                                                                      "3. OTHER: General negative news that requires prompt action from government.\n\n"
    #     "EXAMPLES:\n\n"
    #     "Input: A tragic road accident occurred on the highway where a truck collided with a motorcycle, resulting in the death of the rider.\n"                                                "Output: 1\n\n"
    #     "Input: The city police have arrested the main accused involved in the bank robbery. An FIR has been registered and the investigation is ongoing.\n"
    #     "Output: 1\n\n"                                                                                                                                                                                                                                                                                                                                                                 "Input: Distressed over family disputes and mounting personal debt, a 35-year-old man committed suicide by consuming poison at his residence.\n"
    #     "Output: 1\n\n"
    #     "Input: Two groups fought in public over a parking issue. One person attacked another with a sharp weapon. Police reached the spot and nabbed the attackers.\n"                         "Output: 1\n\n"
    #     "Input: A software engineer lost Rs 50,000 due to online cyber fraud. A case has been filed with the Cyber Crime cell.\n"                                                               "Output: 1\n\n"
    #
    #     "Input: A senior citizen passed away this morning due to a sudden heart attack while returning from the market.\n"
    #     "Output: 1\n\n"
    #
    #     "Input: A complaint has been filed regarding a teenager who went missing yesterday evening.\n"
    #     "Output: 1\n\n"
    #
    #     "Input: This Public Notice informs the public and all concerned, if any person, institution, or any person has a dispute with respect to the sale, mortgage, or any other kind of e>    "
    # Output: 1\n\n"
    #
    # "Input: Villagers are protesting against the electricity department, alleging that a loose wire caused the death of three farmers. No officials have visited the spot yet.\n"
    # "Output: 0\n\n"
    #
    # "Input: A jeweller was shot dead in broad daylight by unidentified assailants. The attackers escaped and police is yet to take any action yet.\n"
    # "Output: 0\n\n"
    #
    # "Input: A farmer committed suicide, leaving a note blaming the Tehsildar for demanding bribes and harassment regarding land records.\n"
    # "Output: 0\n\n"
    #
    # "Output Requirement: Return strictly ONLY the number '1' or '0'. No explanation."

    #        "You are a strict news classifier. Analyze the provided news text and output ONLY '1' or '0'.\n\n"
    #
    #     "CLASSIFICATION RULES:\n"
    #     "Return '1' (BLOCK THIS CONTENT) if ANY of the following are true:\n"
    #     "1. ROAD ACCIDENTS: Any vehicle collisions or traffic-related mishaps.\n"
    #     "2. RESOLVED/ACTIONED INCIDENTS: Any crime or theft or incident where authorities have ALREADY taken action "
    #     "(e.g., 'Police arrested', 'FIR registered', 'Case filed', 'Accused caught', 'Probe ordered', 'Court judgement').\n"
    #     "3. SUICIDES: Suicides due to personal reasons (family disputes, love affairs, exams, debt, depression) by consuming poison, hanging, jumping from building/bridge"
    #     "that DO NOT involve allegations against the State Government or officials.\n\n"
	# "4. DEATHS: Death of a person due to natural causes that do not involve any allegations against the State Government or officials"
    #
    #     "Return '0' (KEEP THIS CONTENT) if:\n"
    #     "1. GOVERNMENT/SYSTEMIC FAULT: Suicides or tragedies blamed on government negligence, official harassment, court orders for government agencies or policy failure.\n"
    #     "2. UNRESOLVED CRIMES: Heinous crimes or law & order situations where NO police action/arrest is mentioned yet.\n"
    #     "3. OTHER: General negative news that requires prompt action from government"
    #
    #     "Output Requirement: Return strictly ONLY the number '1' or '0'. No explanation."

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