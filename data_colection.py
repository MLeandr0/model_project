import os
import zipfile
import pandas as pd
import requests
import re
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder

FILE_URL = "https://static.nhtsa.gov/odi/ffdd/cmpl/COMPLAINTS_RECEIVED_2015-2019.zip"
DOWNLOAD_DIR = "downloads"
EXTRACT_DIR = "extracted"
DATA_DIR = "data"
DOWNLOADED_DATA = "raw_complaints.csv"
OUTPUT_FILE = "data/complaints_with_category.csv"

HEADER = [
    "CMPLID", "ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT", "CRASH",
    "FAILDATE", "FIRE", "INJURED", "DEATHS", "COMPDESC", "CITY", "STATE", "VIN",
    "DATEA", "LDATE", "MILES", "OCCURENCES", "CDESCR", "CMPL_TYPE",
    "POLICE_RPT_YN", "PURCH_DT", "ORIG_OWNER_YN", "ANTI_BRAKES_YN",
    "CRUISE_CONT_YN", "NUM_CYLS", "DRIVE_TRAIN", "FUEL_SYS", "FUEL_TYPE",
    "TRANS_TYPE", "VEH_SPEED", "DOT", "TIRE_SIZE", "LOC_OF_TIRE", "TIRE_FAIL_TYPE",
    "ORIG_EQUIP_YN", "MANUF_DT", "SEAT_TYPE", "RESTRAINT_TYPE", "DEALER_NAME",
    "DEALER_TEL", "DEALER_CITY", "DEALER_STATE", "DEALER_ZIP", "PROD_TYPE",
    "REPAIRED_YN", "MEDICAL_ATTN", "VEHICLES_TOWED_YN"
]

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def download_file(file_url):
    try:
        file_name = file_url.split("/")[-1]
        file_path = os.path.join(DOWNLOAD_DIR, file_name)
        print(f"Downloading {file_name}...")
        
        response = requests.get(file_url)
        response.raise_for_status()
        
        with open(file_path, "wb") as file:
            file.write(response.content)
        
        print(f"Downloaded: {file_name}")
        return file_path
    
    except requests.RequestException as e:
        print(f"Error downloading file: {e}")
        return None

def extract_file(file_path):
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
        print(f"Extraction completed.")
    
    except zipfile.BadZipFile as e:
        print(f"Failed to extract {file_path}: {e}")

def process_csv_file(file_path):
    try:
        df = pd.read_csv(file_path, delimiter="\t", low_memory=False)
        output_path = os.path.join(DATA_DIR, DOWNLOADED_DATA)
        df.to_csv(output_path, index=False)
        print(f"File processed and saved as {output_path}")
    except Exception as e:
        print(f"Error processing file: {e}")

def preprocess_text(text, column_name, columns_to_keep_numbers, stop_words):
    if not isinstance(text, str):
        return text
    
    if column_name == "COMPDESC":
        text = text.replace('/', ' ').replace(':', ' ')
    if column_name in columns_to_keep_numbers:
        text = re.sub(r'[^a-zA-Z0-9\s.]', '', text)
    else:
        text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    return ' '.join(words)

def encode_categorical_columns(df, columns_to_encode):
    label_encoder = LabelEncoder()
    for col in columns_to_encode:
        df[col] = label_encoder.fit_transform(df[col].astype(str))
    return df

compdesc_to_category = {
    "air bags": "air bags",
    "electrical system": "electrical system",
    "engine": "engine",
    "power train": "power train",
    "unknown": "unknown",
    "steering": "steering",
    "brakes": "brakes",
    "suspension": "suspension",
    "structure": "structure",
    "seat": "seat & belts",
    "tires": "tires & wheels",
    "wheels": "tires & wheels",
    "fuel": "fuel system"
}

def map_category(description):
    if pd.isna(description):  
        return "others"
    
    description = description.lower()
    for keyword, category in compdesc_to_category.items():
        if keyword in description:
            return category
    return "others"

def download_data():
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    downloaded_file = download_file(FILE_URL)
    if downloaded_file:
        extract_file(downloaded_file)
        process_csv_file(downloaded_file)
    
    raw_data_path = os.path.join(DATA_DIR, DOWNLOADED_DATA)
    raw_data = pd.read_csv(raw_data_path, header=None, low_memory=False)
    raw_data.columns = HEADER
    raw_data.to_csv(os.path.join(DATA_DIR, "complaints_with_header.csv"), index=False)
    
    input_file = os.path.join(DATA_DIR, "complaints_with_header.csv")
    df = pd.read_csv(input_file, low_memory=False)
    
    threshold = 0.8
    manual_columns_to_remove = ["VIN", "PROD_TYPE", "POLICE_RPT_YN", "VEHICLES_TOWED_YN", "ANTI_BRAKES_YN", "CRUISE_CONT_YN"]
    non_null_percentage = df.notnull().mean()
    columns_to_keep = non_null_percentage[non_null_percentage >= threshold].index
    columns_to_keep = [col for col in columns_to_keep if col not in manual_columns_to_remove]
    df = df[columns_to_keep]

    columns_to_keep_numbers = ['MODELTXT', 'MAKETXT', 'MFR_NAME', 'DATEA', 'LDATE']
    text_columns = df.select_dtypes(include=['object']).columns
    for col in text_columns:
        df[col] = df[col].apply(lambda x: preprocess_text(x, col, columns_to_keep_numbers, stop_words))
    
    float_columns = df.select_dtypes(include=['float']).columns
    for col in float_columns:
        df[col] = df[col].apply(lambda x: int(x) if pd.notnull(x) else 0)
    
    columns_to_encode = ['MAKETXT', 'MODELTXT', 'CRASH', 'FIRE', 'DEATHS', 'FAILDATE', 'FIRE', 'MEDICAL_ATTN', 'ORIG_OWNER_YN']
    df = encode_categorical_columns(df, columns_to_encode)
    
    # Apply category mapping to 'COMPDESC'
    df['CATEGORY'] = df['COMPDESC'].apply(map_category)
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Preprocessed data saved as {OUTPUT_FILE}")

if __name__ == "__main__":
    download_data()