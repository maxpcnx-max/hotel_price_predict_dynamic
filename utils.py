import streamlit as st
import pandas as pd
import joblib
import sqlite3
import os
import json
import holidays
import gdown
import time
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# CONSTANTS
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
    'Grand Suite Room': 2700, 'Villa Suite (Garden)': 2700,
    'Executive Room': 2500, 'Executive Room with Balcony': 2400,
    'Villa Suite (Bathtub)': 2000, 'Deluxe Room': 1500, 'Standard Room': 1000
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

# DATABASE FUNCTIONS
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

# DATA PROCESSING
@st.cache_data
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
            if 'Room' in room_type.columns: room_type['Room'] = room_type['Room'].astype(str)
            if 'Room_Type' in room_type.columns:
                df = df.merge(room_type, on='Room', how='left')
                if 'Room_Type' in df.columns: df = df.rename(columns={'Room_Type': 'Target_Room_Type'})
                elif 'Room_Type_y' in df.columns: df = df.rename(columns={'Room_Type_y': 'Target_Room_Type'})
        df = df.dropna(subset=['Target_Room_Type'])
        df['Reservation'] = df['Reservation'].fillna('Unknown')
        return df
    except: return pd.DataFrame()

def calculate_historical_avg(df):
    if df.empty: return {}
    if 'Night' not in df.columns: df['Night'] = 1
    df_clean = df[df['Night'] > 0].copy()
    df_clean['ADR_Actual'] = df_clean['Price'] / df_clean['Night']
    return df_clean.groupby('Target_Room_Type')['ADR_Actual'].mean().to_dict()

# MODEL FUNCTIONS
@st.cache_resource
def load_system_models():
    for name, file in MODEL_FILES.items():
        if not os.path.exists(file): return None, None, None, None, None
    xgb = joblib.load(MODEL_FILES['xgb'])
    lr = joblib.load(MODEL_FILES['lr'])
    le_room = joblib.load(MODEL_FILES['le_room'])
    le_res = joblib.load(MODEL_FILES['le_res'])
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, 'r') as f: metrics = json.load(f)
    else: metrics = DEFAULT_METRICS
    return xgb, lr, le_room, le_res, metrics

def save_uploaded_data_with_cleaning(uploaded_file):
    try:
        uploaded_file.seek(0)
        new_data = pd.read_csv(uploaded_file)
        if 'Room' in new_data.columns: new_data['Room'] = new_data['Room'].astype(str)
        valid_rooms = set()
        if os.path.exists(ROOM_FILE):
            room_master = pd.read_csv(ROOM_FILE)
            if 'Room' in room_master.columns: valid_rooms = set(room_master['Room'].astype(str))
        if len(valid_rooms) > 0:
            good_rows = new_data[new_data['Room'].isin(valid_rooms)]
            bad_rows = new_data[~new_data['Room'].isin(valid_rooms)]
            if len(bad_rows) > 0:
                st.warning(f"⚠️ ตรวจพบข้อมูลห้องที่ไม่รู้จัก จำนวน {len(bad_rows)} รายการ")
                st.error(f"รายการที่ถูกตัดทิ้ง: {bad_rows['Room'].unique()}")
            data_to_save = good_rows
        else: data_to_save = new_data
        if not data_to_save.empty:
            if os.path.exists(DATA_FILE):
                current_df = pd.read_csv(DATA_FILE)
                if 'Room' in current_df.columns: current_df['Room'] = current_df['Room'].astype(str)
                updated_df = pd.concat([current_df, data_to_save], ignore_index=True)
            else: updated_df = data_to_save
            updated_df.to_csv(DATA_FILE, index=False)
            st.cache_data.clear()
            return True
        return False
    except Exception as e:
        st.error(f"Save failed: {e}")
        return False

def retrain_system():
    status_text = st.empty()
    progress_bar = st.progress(0)
    try:
        status_text.text("⏳ Reading & Cleaning data...")
        df = load_data() 
        if df.empty: return False, 0
        df = df.dropna(subset=['Price', 'Night'])
        for col, val in [('Night',1),('Adults',2),('Children',0),('Infants',0),('Extra Person',0)]: df[col] = df[col].fillna(val)
        if not os.path.exists("thai_holidays.csv"):
             try: gdown.download("https://drive.google.com/uc?id=1L-pciKEeRce1gzuhdtpIGcLs0fYHnbZw", "thai_holidays.csv", quiet=True)
             except: pass
        if os.path.exists("thai_holidays.csv"):
            h_csv = pd.read_csv("thai_holidays.csv")
            h_csv['Holiday_Date'] = pd.to_datetime(h_csv['Holiday_Date'], dayfirst=True, errors='coerce')
            df['is_holiday'] = df['Date'].isin(h_csv['Holiday_Date']).astype(int)
        else: df['is_holiday'] = 0
        df['is_weekend'] = df['Date'].dt.weekday.isin([5, 6]).astype(int)
        df['total_guests'] = df[['Adults', 'Children', 'Infants', 'Extra Person']].sum(axis=1)
        df['month'] = df['Date'].dt.month
        df['weekday'] = df['Date'].dt.weekday
        le_room_new, le_res_new = LabelEncoder(), LabelEncoder()
        df['RoomType_encoded'] = le_room_new.fit_transform(df['Target_Room_Type'].astype(str))
        df['Reservation_encoded'] = le_res_new.fit_transform(df['Reservation'].astype(str))
        feature_cols = ['Night', 'total_guests', 'is_holiday', 'is_weekend', 'month', 'weekday', 'RoomType_encoded', 'Reservation_encoded']
        X, y = df[feature_cols].fillna(0), df['Price']
        progress_bar.progress(40)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        xgb_new = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        xgb_new.fit(X_train, y_train)
        new_importance = { {'Night':'Night','total_guests':'Guests','is_holiday':'Is Holiday','is_weekend':'Is Weekend','month':'Month','weekday':'Weekday','RoomType_encoded':'Room Type','Reservation_encoded':'Reservation'}.get(c,c): float(v) for c,v in zip(feature_cols, xgb_new.feature_importances_) }
        lr_new = LinearRegression().fit(X_train, y_train)
        progress_bar.progress(80)
        joblib.dump(xgb_new, MODEL_FILES['xgb']); joblib.dump(lr_new, MODEL_FILES['lr'])
        joblib.dump(le_room_new, MODEL_FILES['le_room']); joblib.dump(le_res_new, MODEL_FILES['le_res'])
        new_metrics = { 'xgb': {'mae': mean_absolute_error(y_test, xgb_new.predict(X_test)), 'r2': r2_score(y_test, xgb_new.predict(X_test))},
                        'lr':  {'mae': mean_absolute_error(y_test, lr_new.predict(X_test)), 'r2': r2_score(y_test, lr_new.predict(X_test))},
                        'importance': new_importance }
        with open(METRICS_FILE, 'w') as f: json.dump(new_metrics, f)
        st.session_state['historical_avg'] = calculate_historical_avg(df)
        st.cache_resource.clear(); progress_bar.progress(100)
        return True, len(df)
    except Exception as e:
        st.error(f"Retrain Error: {e}"); return False, 0

# PRICING LOGIC HELPERS
def get_base_price(room_text):
    if not isinstance(room_text, str): return 0
    for key, val in BASE_PRICES.items():
        if key in room_text: return val
    return 0

def predict_segmented_price(model, start_date, n_nights, guests, r_code, res_code):
    MAX_CHUNK, total_predicted, remaining_nights, current_date = 7, 0, n_nights, start_date
    th_holidays = holidays.Thailand()
    while remaining_nights > 0:
        chunk_nights = min(remaining_nights, MAX_CHUNK)
        chunk_end_date = current_date + timedelta(days=chunk_nights)
        chunk_is_holiday = 0
        temp_date = current_date
        while temp_date < chunk_end_date:
            if temp_date in th_holidays: chunk_is_holiday = 1; break
            temp_date += timedelta(days=1)
        inp_chunk = pd.DataFrame([{'Night': chunk_nights, 'total_guests': guests, 'is_holiday': chunk_is_holiday, 'is_weekend': 1 if current_date.weekday() in [5, 6] else 0, 'month': current_date.month, 'weekday': current_date.weekday(), 'RoomType_encoded': r_code, 'Reservation_encoded': res_code}])
        total_predicted += model.predict(inp_chunk)[0]
        remaining_nights -= chunk_nights
        current_date = chunk_end_date
    return total_predicted

def calculate_rule_based_price(base_per_night, start_date, n_nights, use_holiday, use_weekend):
    th_holidays, total_price, current_date = holidays.Thailand(), 0, start_date
    for _ in range(n_nights):
        m, is_weekend, is_holiday, is_near = 1.0, current_date.weekday() in [5,6], current_date in th_holidays, any((current_date + timedelta(days=i)) in th_holidays for i in range(1,4))
        if is_holiday and use_holiday: m = 1.7 if (is_weekend and use_weekend) else 1.5
        elif is_weekend and use_weekend: m = 1.56 if (is_near and use_holiday) else 1.2
        elif is_near and use_holiday: m = 1.3
        total_price += (base_per_night * m)
        current_date += timedelta(days=1)
    return total_price

def calculate_clamped_price(model, start_date, n_nights, guests, r_code, res_code, room_name, use_h, use_w):
    raw_predicted = predict_segmented_price(model, start_date, n_nights, guests, r_code, res_code)
    base_per_night = get_base_price(room_name)
    rule_price = calculate_rule_based_price(base_per_night, start_date, n_nights, use_h, use_w)
    hist_avg = st.session_state.get('historical_avg', {}).get(room_name, 0)
    final_price = rule_price + (raw_predicted - (hist_avg * n_nights)) if hist_avg > 0 else rule_price
    return max(final_price, base_per_night * n_nights), raw_predicted, rule_price