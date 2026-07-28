import requests
import json
import pandas as pd
from json_repair import repair_json
from sklearn.model_selection import train_test_split
import pyarabic.araby as araby

def clean_arabic_text(text):
    text = str(text)
    text = araby.strip_tashkeel(text)
    text = araby.normalize_alef(text)
    return text

def load_and_prepare_data(url="https://raw.githubusercontent.com/NoorBayan/Burhan/main/corpus/metaphors_data.json"):
    response = requests.get(url)
    fixed_json_string = repair_json(response.text)
    data = json.loads(fixed_json_string)

    records = []
    # خريطة الترتيب المعرفي (Ordinal Mapping)
    effort_map = {"Low": 0, "Medium": 1, "High": 2}

    for item in data:
        ayah = item.get('metadata', {}).get('ayah_text_uthmani', '')
        similes = item.get('rhetorical_analysis', {}).get('similes', []) # في الداتا ست الاستعارات توجد تحت مفتاح similes
        
        if not similes: continue
        
        for metaphor in similes:
            classification = metaphor.get('classification', {})
            effort = classification.get('processing_effort')
            segment = metaphor.get('simile_identity', {}).get('segment_text', '')
            
            if effort in effort_map and ayah:
                # دمج الآية مع المقطع الذي يحتوي الاستعارة
                combined_text = f"{ayah} [SEP] {segment}" if segment else ayah
                records.append({
                    'text': combined_text, 
                    'label_text': effort,
                    'label': effort_map[effort]
                })
                break # نأخذ الاستعارة الأولى لكل آية لتجنب التكرار

    df = pd.DataFrame(records)
    df['clean_text'] = df['text'].apply(clean_arabic_text)

    # التقسيم الطبقي للحفاظ على نسب (Low, Medium, High)
    train_df, test_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df['label'])
    
    # القاموس العكسي للاستخدام لاحقاً
    label_encoder = {0: "Low", 1: "Medium", 2: "High"}
    
    return train_df, test_df, label_encoder
