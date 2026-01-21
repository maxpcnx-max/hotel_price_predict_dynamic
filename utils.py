import pandas as pd
import joblib
import sqlite3
import os
import json
import gdown
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Constants
DB_FILE = "users.db"
DATA_FILE = "check_in_report.csv"
ROOM_FILE = "room_type.csv"
METRICS_FILE = "model_metrics.json"
MODEL_FILES = {
    'xgb': 'xgb_hotel_model.joblib',
    'lr': 'lr_hotel_model.joblib',
    'le_room': 'le_room.joblib',
    'le_res': 'le_res.joblib'
}

BASE_PRICES = {
    'Grand Suite Room': 2700,
    'Villa Suite (Garden)': 2700,
    'Executive Room': 2500,
    'Executive Room with Balcony': 2400,
    'Villa Suite (Bathtub)': 2000,
    'Deluxe Room': 1500,
    'Standard Room': 1000
}

DEFAULT_METRICS = {
    'xgb': {'mae': 1112.79, 'r2': 0.7256},
    'lr':  {'mae': 1162.27, 'r2': 0.7608},
    'importance': {
        'Night': 0.4364, 'Reservation': 0.1742, 'Month': 0.1315, 
        'Is Weekend': 0.0643, 'Room Type': 0.0640, 'Weekday': 0.0512, 
        'Guests': 0.0508, 'Is Holiday': 0.0275
    }
}

# --- Database Management ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)')
    c.execute('SELECT * FROM users WHERE username = "admin"')
    if not c.fetchone():
        c.execute('INSERT INTO users VALUES (?,?)', ("admin", "1234"))
        conn.commit()
    conn.close()

def login_user(username, password):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
    data = c.fetchone()
    conn.close()
    return data

def register_user(username, password):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('INSERT INTO users VALUES (?,?)', (username, password))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError: return False

# --- Data Management ---
def load_data():
    if not os.path.exists(DATA_FILE):
        try: gdown.download("https://drive.google.com/uc?id=1dxgKIvSTelLaJvAtBSCMCU5K4FuJvfri", DATA_FILE, quiet=True)
        except: return pd.DataFrame()

    try:
        df = pd.read_csv(DATA_FILE)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            df = df.dropna(subset=['Date'])
            df['is_weekend'] = df['Date'].dt.weekday.isin([5, 6]).astype(int)
            df['Year'] = df['Date'].dt.year.astype(int)
            df['month'] = df['Date'].dt.month
            
        if 'Room' in df.columns: df['Room'] = df['Room'].astype(str)

        if os.path.exists(ROOM_FILE):
            room_type = pd.read_csv(ROOM_FILE)
            room_type['Room'] = room_type['Room'].astype(str)
            df = df.merge(room_type, on='Room', how='left')
            if 'Room_Type' in df.columns: df = df.rename(columns={'Room_Type': 'Target_Room_Type'})
            elif 'Room_Type_y' in df.columns: df = df.rename(columns={'Room_Type_y': 'Target_Room_Type'})
        
        df = df.dropna(subset=['Target_Room_Type'])
        df['Reservation'] = df['Reservation'].fillna('Unknown')
        return df
    except: return pd.DataFrame()

def load_system_models():
    for name, file in MODEL_FILES.items():
        if not os.path.exists(file): return None, None, None, None, DEFAULT_METRICS
    xgb = joblib.load(MODEL_FILES['xgb'])
    lr = joblib.load(MODEL_FILES['lr'])
    le_room = joblib.load(MODEL_FILES['le_room'])
    le_res = joblib.load(MODEL_FILES['le_res'])
    metrics = DEFAULT_METRICS
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, 'r') as f: metrics = json.load(f)
    return xgb, lr, le_room, le_res, metrics