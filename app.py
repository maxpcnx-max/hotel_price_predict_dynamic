import streamlit as st
import pandas as pd
import joblib
import sqlite3
import os
import json
import holidays
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gdown
import time
from datetime import datetime, timedelta

# Import Library สำหรับการ Retrain
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================================
# 1. SETUP & CONSTANTS
# ==========================================================
st.set_page_config(
    page_title="Hotel Price Forecasting System (Diagnostic Mode)",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_FILE = "users.db"
DATA_FILE = "check_in_report.csv"
ROOM_FILE = "room_type.csv" 
METRICS_FILE = "model_metrics.json"
BASE_PRICE_FILE = "base_prices.json" 
CHANNELS_FILE = "channels.json" 

MODEL_FILES = {
    'xgb': 'xgb_hotel_model.joblib',
    'lr': 'lr_hotel_model.joblib',
    'le_room': 'le_room.joblib',
    'le_res': 'le_res.joblib'
}

DEFAULT_BASE_PRICES = {
    'Grand Suite Room': 2700,
    'Villa Suite (Garden)': 2700,
    'Executive Room': 2500,
    'Executive Room with Balcony': 2400,
    'Villa Suite (Bathtub)': 2000,
    'Deluxe Room': 1500,
    'Standard Room': 1000
}

DEFAULT_CHANNELS = ["Agoda", "Booking.com", "Traveloka", "Walk-in", "Direct", "Expedia", "Trip.com"]

DEFAULT_METRICS = {'xgb': {'mae': 0, 'r2': 0}, 'lr': {'mae': 0, 'r2': 0}, 'importance': {}}

if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'username' not in st.session_state: st.session_state['username'] = ""

# ==========================================================
# 2. HELPER FUNCTIONS
# ==========================================================

def load_base_prices():
    if not os.path.exists(BASE_PRICE_FILE):
        with open(BASE_PRICE_FILE, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_BASE_PRICES, f, ensure_ascii=False, indent=4)
        return DEFAULT_BASE_PRICES
    try:
        with open(BASE_PRICE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return DEFAULT_BASE_PRICES

def save_base_prices(price_dict):
    with open(BASE_PRICE_FILE, 'w', encoding='utf-8') as f:
        json.dump(price_dict, f, ensure_ascii=False, indent=4)

def load_channels():
    if not os.path.exists(CHANNELS_FILE):
        with open(CHANNELS_FILE, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_CHANNELS, f, ensure_ascii=False, indent=4)
        return DEFAULT_CHANNELS
    try:
        with open(CHANNELS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return DEFAULT_CHANNELS

def save_channels(channel_list):
    with open(CHANNELS_FILE, 'w', encoding='utf-8') as f:
        json.dump(channel_list, f, ensure_ascii=False, indent=4)

def get_base_price(room_text):
    if not isinstance(room_text, str): return 0
    prices = load_base_prices()
    for key in prices:
        if key in room_text: return prices[key]
    return 0

# ==========================================================
# 3. DATABASE
# ==========================================================
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

init_db()

# ==========================================================
# 4. BACKEND SYSTEM (Diagnostic Mode)
# ==========================================================

@st.cache_data
def load_raw_data():
    """โหลดข้อมูลดิบ (พยายามอ่าน Date เป็น String ก่อนเพื่อไม่ให้ข้อมูลเพี้ยน)"""
    if not os.path.exists(DATA_FILE):
        try: gdown.download("https://drive.google.com/uc?id=1dxgKIvSTelLaJvAtBSCMCU5K4FuJvfri", DATA_FILE, quiet=True)
        except: return pd.DataFrame()

    try:
        # อ่าน Date เป็น object (String) ไปเลย จะได้เห็นว่า User ใส่อะไรมา
        df = pd.read_csv(DATA_FILE)
        
        if 'Room' in df.columns:
            df['Room'] = df['Room'].astype(str)
            
        return df
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()

def process_data_for_dashboard(df_raw):
    """เตรียมข้อมูลสำหรับ Dashboard"""
    if df_raw.empty: return df_raw
    df = df_raw.copy()
    
    # พยายามแปลง Date เฉพาะตอนจะใช้ Dashboard
    if 'Date' in df.columns:
        df['Date_Obj'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    else:
        df['Date_Obj'] = pd.NaT

    # 1. Feature Engineering
    mask = df['Date_Obj'].notna()
    df.loc[mask, 'is_weekend'] = df.loc[mask, 'Date_Obj'].dt.weekday.isin([5, 6]).astype(int)
    df.loc[mask, 'Year'] = df.loc[mask, 'Date_Obj'].dt.year.astype(int)
    df.loc[mask, 'month'] = df.loc[mask, 'Date_Obj'].dt.month
    
    # คืนค่า Date เป็น Object เหมือนเดิม แต่เอา Date_Obj ไว้ใช้คำนวณ
    # (หรือจะแทนที่ Date เลยก็ได้ แต่เพื่อความชัวร์ เก็บตัวแปรใหม่ไว้ filter)
    
    # 2. Merge Room Type
    if os.path.exists(ROOM_FILE):
        try:
            room_type = pd.read_csv(ROOM_FILE)
            if 'Room' in room_type.columns: room_type['Room'] = room_type['Room'].astype(str)
            
            if 'Target_Room_Type' in room_type.columns:
                df = df.merge(room_type[['Room', 'Target_Room_Type']], on='Room', how='left')
            elif 'Room_Type' in room_type.columns:
                room_type = room_type.rename(columns={'Room_Type': 'Target_Room_Type'})
                df = df.merge(room_type[['Room', 'Target_Room_Type']], on='Room', how='left')
        except: pass
    
    if 'Target_Room_Type' in df.columns:
        df['Target_Room_Type'] = df['Target_Room_Type'].fillna(df['Room'])
    else:
        df['Target_Room_Type'] = df['Room']
    
    df['Reservation'] = df['Reservation'].fillna('Unknown')
    return df

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

# 🔥 NEW: Diagnostic Validator (นักสืบ)
def validate_and_save_data(df_to_check, save_to_file=True):
    valid_rooms = set(load_base_prices().keys())
    valid_channels = set(load_channels())
    
    if os.path.exists(ROOM_FILE):
        try:
            rt = pd.read_csv(ROOM_FILE)
            if 'Room' in rt.columns: valid_rooms.update(rt['Room'].astype(str))
        except: pass

    df_clean = df_to_check.copy()
    
    # แปลง Room/Res เป็น String เพื่อเทียบ
    if 'Room' in df_clean.columns: df_clean['Room'] = df_clean['Room'].astype(str)
    if 'Reservation' in df_clean.columns: df_clean['Reservation'] = df_clean['Reservation'].astype(str)

    # 🕵️‍♂️ 1. ตรวจสอบวันที่ (ตัวปัญหา)
    # เราจะลองแปลงดู ถ้าพัง -> invalid
    if 'Date' in df_clean.columns:
        # ใช้ errors='coerce' เพื่อหาตัวที่พัง
        df_clean['Date_Parsed'] = pd.to_datetime(df_clean['Date'], dayfirst=True, errors='coerce')
    else:
        df_clean['Date_Parsed'] = pd.NaT

    # สร้าง Mask
    mask_date = df_clean['Date_Parsed'].notna()
    mask_room = df_clean['Room'].isin(valid_rooms)
    mask_channel = df_clean['Reservation'].isin(valid_channels)

    # 🔥 สร้างรายงานสาเหตุการตัดทิ้ง
    df_clean['Error_Reason'] = ""
    df_clean.loc[~mask_date, 'Error_Reason'] += "Date Invalid/Empty; "
    df_clean.loc[~mask_room, 'Error_Reason'] += "Room Unknown; "
    df_clean.loc[~mask_channel, 'Error_Reason'] += "Channel Unknown; "

    mask_valid = mask_date & mask_room & mask_channel
    
    df_good = df_clean[mask_valid].copy()
    df_bad = df_clean[~mask_valid].copy()
    
    # ลบคอลัมน์ช่วยคำนวณออกก่อน Save
    if save_to_file and not df_good.empty:
        save_cols = ['Date', 'Room', 'Price', 'Reservation', 'Name', 'Night', 'Adults', 'Children', 'Infants', 'Extra Person']
        
        # เอา Date ที่แปลงแล้ว (Date_Parsed) ไปเก็บ หรือเอา Date เดิม?
        # เอา Date เดิมดีกว่า เพราะถ้าแปลงแล้วอาจเปลี่ยน Format ที่ User ไม่ชอบ
        # แต่... ถ้า Date เดิมมันเป็น format มั่วๆ การแปลงแล้ว (Date_Parsed) จะ safe กว่า
        # ตัดสินใจ: ใช้ Date เดิม (Raw) แต่ User ต้องรับผิดชอบ Format เอง
        # หรือถ้าจะให้ดี ใช้ Date_Parsed ที่แปลงเป็น String ISO
        df_good['Date'] = df_good['Date_Parsed'].dt.strftime('%Y-%m-%d') # บังคับ ISO Format ลงไฟล์เลย กันเหนียว
        
        final_cols = [c for c in save_cols if c in df_good.columns]
        df_good[final_cols].to_csv(DATA_FILE, index=False)
        st.cache_data.clear()
        
    return df_good, df_bad

def retrain_system():
    status_text = st.empty()
    progress_bar = st.progress(0)
    try:
        status_text.text("⏳ Reading & Cleaning data...")
        df_raw = load_raw_data()
        df = process_data_for_dashboard(df_raw) # ใช้ฟังก์ชัน Process ที่มี Date_Obj
        
        # ใช้ Date_Obj แทน Date ปกติในการกรอง
        if df.empty: return False, 0
        df = df.dropna(subset=['Price', 'Night', 'Date_Obj'])
        df['Date'] = df['Date_Obj'] # คืนค่าให้คอลัมน์ Date หลัก

        df['Night'] = df['Night'].fillna(1)
        df['Adults'] = df['Adults'].fillna(2)
        df['Children'] = df['Children'].fillna(0)
        
        if not os.path.exists("thai_holidays.csv"):
             try: gdown.download("https://drive.google.com/uc?id=1L-pciKEeRce1gzuhdtpIGcLs0fYHnbZw", "thai_holidays.csv", quiet=True)
             except: pass
        if os.path.exists("thai_holidays.csv"):
            holidays_csv = pd.read_csv("thai_holidays.csv")
            holidays_csv['Holiday_Date'] = pd.to_datetime(holidays_csv['Holiday_Date'], dayfirst=True, errors='coerce')
            df['is_holiday'] = df['Date'].isin(holidays_csv['Holiday_Date']).astype(int)
        else: df['is_holiday'] = 0

        df['is_weekend'] = df['Date'].dt.weekday.isin([5, 6]).astype(int)
        df['total_guests'] = df[['Adults', 'Children']].sum(axis=1)
        df['month'] = df['Date'].dt.month
        df['weekday'] = df['Date'].dt.weekday
        
        le_room_new = LabelEncoder()
        df['RoomType_encoded'] = le_room_new.fit_transform(df['Target_Room_Type'].astype(str))
        le_res_new = LabelEncoder()
        df['Reservation_encoded'] = le_res_new.fit_transform(df['Reservation'].astype(str))
        
        feature_cols = ['Night', 'total_guests', 'is_holiday', 'is_weekend', 'month', 'weekday', 'RoomType_encoded', 'Reservation_encoded']
        X = df[feature_cols].fillna(0)
        y = df['Price']
        
        progress_bar.progress(40)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        xgb_new = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        xgb_new.fit(X_train, y_train)
        pred_xgb = xgb_new.predict(X_test)
        new_xgb_mae = mean_absolute_error(y_test, pred_xgb)
        new_xgb_r2 = r2_score(y_test, pred_xgb)
        
        lr_new = LinearRegression()
        lr_new.fit(X_train, y_train)
        
        progress_bar.progress(80)
        joblib.dump(xgb_new, MODEL_FILES['xgb'])
        joblib.dump(lr_new, MODEL_FILES['lr'])
        joblib.dump(le_room_new, MODEL_FILES['le_room'])
        joblib.dump(le_res_new, MODEL_FILES['le_res'])
        
        new_metrics = {'xgb': {'mae': new_xgb_mae, 'r2': new_xgb_r2}, 'lr':  {'mae': 0, 'r2': 0}, 'importance': {}}
        with open(METRICS_FILE, 'w') as f: json.dump(new_metrics, f)
            
        st.cache_resource.clear()
        progress_bar.progress(100)
        status_text.success(f"✅ Retraining Complete! R²: {new_xgb_r2:.4f}")
        return True, len(df)
    except Exception as e:
        st.error(f"Retrain Error: {e}")
        return False, 0

# ==========================================================
# 5. MAIN UI PAGES
# ==========================================================

def login_page():
    st.markdown("""<style>.stTextInput > div > div > input {text-align: center;}</style>""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/2933/2933116.png", width=120)
        st.title("🔒 Login System")
        st.divider()
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login", type="primary", use_container_width=True):
            if login_user(u, p): 
                st.session_state['logged_in'] = True
                st.session_state['username'] = u
                st.rerun()
            else: 
                st.error("Invalid Username or Password")

if not st.session_state['logged_in']:
    login_page()
else:
    df_raw = load_raw_data() 
    df_dash = process_data_for_dashboard(df_raw)
    xgb_model, lr_model, le_room, le_res, metrics = load_system_models()
    
    def show_dashboard_page():
        st.title("📊 Financial Executive Dashboard")
        
        # 🔥 SECTION: DATA HEALTH CHECK (โชว์ความจริง)
        total_rows = len(df_raw)
        # เช็คจาก Date_Obj ที่ process แล้ว
        if 'Date_Obj' in df_dash.columns:
            valid_rows = len(df_dash.dropna(subset=['Date_Obj']))
        else:
            valid_rows = 0
        bad_rows = total_rows - valid_rows
        
        c1, c2, c3 = st.columns(3)
        c1.metric("📂 ข้อมูลทั้งหมดในไฟล์", f"{total_rows:,}")
        c2.metric("✅ ข้อมูลที่สมบูรณ์ (มีวันที่)", f"{valid_rows:,}")
        c3.metric("⚠️ ข้อมูลที่ไม่มีวันที่ (ถูกซ่อน)", f"{bad_rows:,}", delta_color="inverse")
        
        if bad_rows > 0:
            with st.expander(f"🔎 ดูรายการข้อมูลที่ไม่มีวันที่ ({bad_rows} รายการ)"):
                st.warning("รายการเหล่านี้ Date เป็น NaN หรือ Parse ไม่ได้")
                if 'Date_Obj' in df_dash.columns:
                    st.dataframe(df_raw[df_dash['Date_Obj'].isna()])
                else:
                    st.dataframe(df_raw)

        if valid_rows == 0: st.error("ไม่พบข้อมูลที่สมบูรณ์สำหรับแสดงกราฟ"); return

        df_filtered = df_dash.dropna(subset=['Date_Obj']).copy()
        # คืนค่า Date สำหรับโชว์
        df_filtered['Date'] = df_filtered['Date_Obj']

        with st.expander("🔎 Filter Data (ตัวกรองกราฟ)", expanded=True):
            f_col1, f_col2, f_col3 = st.columns(3)
            if not df_filtered.empty:
                years = sorted(df_filtered['Year'].dropna().unique())
                year_opts = ['All'] + [str(int(y)) for y in years]
                with f_col1: sel_year = st.selectbox("📅 Select Year (เลือกปี)", year_opts)
                
                months = sorted(df_filtered['month'].dropna().unique())
                month_opts = ['All'] + [datetime(2024, int(m), 1).strftime('%B') for m in months]
                with f_col2: sel_month_str = st.selectbox("🗓️ Select Month (เลือกเดือน)", month_opts)

                if sel_year != 'All': df_filtered = df_filtered[df_filtered['Year'] == int(sel_year)]
                if sel_month_str != 'All':
                    sel_month_num = datetime.strptime(sel_month_str, "%B").month
                    df_filtered = df_filtered[df_filtered['month'] == sel_month_num]

        st.divider()
        k1, k2, k3 = st.columns(3)
        with k1: st.metric("💰 Total Revenue", f"{df_filtered['Price'].sum()/1e6:.2f} M THB")
        with k2: st.metric("📦 Total Bookings", f"{len(df_filtered):,} รายการ")
        with k3: st.metric("🏷️ Avg. Booking Value", f"{df_filtered['Price'].mean():,.0f} THB")
        
        st.divider()
        tab1, tab2, tab3 = st.tabs(["💰 Financial Overview", "📢 Channel Strategy", "🛌 Product & Behavior"])
        group_col = 'Target_Room_Type' if 'Target_Room_Type' in df_filtered.columns else 'Room'

        with tab1:
            st.markdown("### 1. Financial Overview")
            c1, c2 = st.columns(2)
            with c1:
                room_perf = df_filtered.groupby(group_col).agg({'Price': 'sum', 'Night': 'sum'}).reset_index().sort_values('Price', ascending=False)
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Bar(x=room_perf[group_col], y=room_perf['Price'], name="Revenue", marker_color='#1f77b4'), secondary_y=False)
                fig.add_trace(go.Scatter(x=room_perf[group_col], y=room_perf['Night'], name="Nights", mode='lines+markers', marker_color='#ff7f0e'), secondary_y=True)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                monthly = df_filtered.groupby('month').agg({'Price': 'sum', 'Room': 'count'}).reset_index().sort_values('month')
                monthly['M_Name'] = monthly['month'].apply(lambda x: datetime(2024, int(x), 1).strftime('%b'))
                fig2 = make_subplots(specs=[[{"secondary_y": True}]])
                fig2.add_trace(go.Scatter(x=monthly['M_Name'], y=monthly['Price'], name="Revenue", line=dict(color='green', width=3)), secondary_y=False)
                fig2.add_trace(go.Scatter(x=monthly['M_Name'], y=monthly['Room'], name="Bookings", line=dict(color='blue', dash='dot')), secondary_y=True)
                st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            st.markdown("### 2. Channel Strategy")
            if 'Reservation' in df_filtered.columns:
                res_rev = df_filtered.groupby('Reservation')['Price'].sum().reset_index()
                st.plotly_chart(px.pie(res_rev, values='Price', names='Reservation', hole=0.4), use_container_width=True)

        with tab3:
            st.markdown("### 3. Product & Behavior")
            mt_room = df_filtered.groupby(['month', group_col])['Price'].sum().reset_index()
            mt_room['M_Name'] = mt_room['month'].astype(int).apply(lambda x: datetime(2024, x, 1).strftime('%b'))
            st.plotly_chart(px.bar(mt_room, x='M_Name', y='Price', color=group_col), use_container_width=True)

        st.divider()
        st.subheader("📋 Raw Data Explorer (Filtered)")
        with st.expander("คลิกเพื่อดูตารางข้อมูลที่ผ่านการกรองแล้ว"): st.dataframe(df_filtered)

    def show_manage_data_page():
        st.title("📥 ระบบจัดการฐานข้อมูล (Data Management)")
        
        tab_trans, tab_master, tab_channel, tab_train = st.tabs([
            "📝 ข้อมูลการจอง (Transactions)", 
            "⚙️ ราคาฐาน (Base Price)", 
            "📢 ช่องทางการจอง (Channels)", 
            "🚀 อัปเดตโมเดล (Retrain)"
        ])

        with tab_trans:
            st.subheader("1. นำเข้าข้อมูลใหม่ (Import)")
            up_file = st.file_uploader("เลือกไฟล์ CSV", type=['csv'])
            if up_file is not None:
                if st.button("➕ Merge File (ต่อท้ายข้อมูลเดิม)", type="secondary"):
                    try:
                        new_df = pd.read_csv(up_file)
                        if os.path.exists(DATA_FILE):
                            old_df = pd.read_csv(DATA_FILE)
                            merged_df = pd.concat([old_df, new_df], ignore_index=True)
                        else: merged_df = new_df
                        
                        good_df, bad_df = validate_and_save_data(merged_df, save_to_file=True)
                        
                        if not bad_df.empty:
                            st.warning(f"⚠️ พบข้อมูลที่ไม่ถูกต้อง {len(bad_df)} รายการ")
                            with st.expander("🔴 ดูรายการที่ถูกตัดออก (พร้อมสาเหตุ)"): 
                                st.dataframe(bad_df[['Date', 'Room', 'Reservation', 'Error_Reason']])
                        
                        st.success(f"✅ บันทึกข้อมูลที่ถูกต้อง {len(good_df)} รายการ เรียบร้อย!")
                        time.sleep(1); st.rerun()
                    except Exception as e: st.error(f"Error: {e}")

            st.divider()
            st.subheader("2. แก้ไขข้อมูล (Gatekeeper Mode)")
            st.info("💡 ระบบจะตรวจสอบวันที่, ห้อง, และช่องทางก่อนบันทึก")
            
            # โหลดแบบ Raw String
            df_current = load_raw_data() 
            
            # ไม่แปลง Date ที่นี่ ปล่อยให้ Editor จัดการแบบ String หรือถ้าจะให้ดีแปลงเป็น datetime ให้ editor
            # แต่ถ้าแปลงแล้ว Error จะเป็น NaT -> User จะเห็นช่องว่าง
            # ลองแปลงดู ถ้า NaT User ต้องกรอกใหม่
            if 'Date' in df_current.columns:
                df_current['Date'] = pd.to_datetime(df_current['Date'], dayfirst=True, errors='coerce')

            edited_df = st.data_editor(
                df_current,
                num_rows="dynamic",
                use_container_width=True,
                key="booking_editor_gatekeeper",
                column_config={
                    "Date": st.column_config.DateColumn("Check-in Date", format="DD/MM/YYYY"),
                    "Price": st.column_config.NumberColumn("Price (THB)", format="%d"),
                }
            )

            col_save, col_reset = st.columns([1, 4])
            
            with col_save:
                if st.button("💾 บันทึกการเปลี่ยนแปลง (Save & Filter)", type="primary"):
                    try:
                        good_df, bad_df = validate_and_save_data(edited_df, save_to_file=True)
                        
                        if not bad_df.empty:
                            st.error(f"🚫 ระบบลบข้อมูลที่ไม่ถูกต้องออก {len(bad_df)} รายการ")
                            with st.expander("🔴 ดูรายการที่ถูกลบ (พร้อมสาเหตุ)"): 
                                # โชว์เฉพาะคอลัมน์สำคัญ + Error Reason
                                show_cols = ['Date', 'Room', 'Reservation', 'Error_Reason']
                                st.dataframe(bad_df[[c for c in show_cols if c in bad_df.columns]])
                        
                        st.success(f"✅ บันทึกข้อมูลที่ถูกต้อง {len(good_df)} รายการ สำเร็จ!")
                        time.sleep(1); st.rerun()
                        
                    except Exception as e:
                        st.error(f"Save Error: {e}")
            
            with col_reset:
                if st.button("🧨 ล้างข้อมูลทั้งหมด (Hard Reset)"):
                     if os.path.exists(DATA_FILE):
                        os.remove(DATA_FILE)
                        st.cache_data.clear()
                        st.success("ล้างข้อมูลเรียบร้อย!")
                        time.sleep(1)
                        st.rerun()

        with tab_master:
            st.subheader("⚙️ กำหนดราคาฐาน (รายชื่อห้องที่ระบบยอมรับ)")
            current_prices = load_base_prices()
            df_prices = pd.DataFrame(list(current_prices.items()), columns=['Room Type', 'Base Price'])
            edited_prices_df = st.data_editor(df_prices, num_rows="dynamic", use_container_width=True, key="price_editor_tab", column_config={"Base Price": st.column_config.NumberColumn("Price", format="%d THB")})
            if st.button("💾 บันทึกราคาฐาน"):
                new_prices_dict = {}
                for index, row in edited_prices_df.iterrows():
                    if row['Room Type'] and str(row['Room Type']).strip() != "":
                        new_prices_dict[row['Room Type']] = row['Base Price']
                save_base_prices(new_prices_dict)
                st.success("✅ อัปเดตราคาฐาน และ รายชื่อห้อง เรียบร้อย!")

        with tab_channel:
            st.subheader("📢 จัดการช่องทางการจอง (Channels)")
            current_channels = load_channels()
            df_channels = pd.DataFrame(current_channels, columns=['Channel Name'])
            edited_channels_df = st.data_editor(df_channels, num_rows="dynamic", use_container_width=True, key="channel_editor_tab")
            if st.button("💾 บันทึกช่องทาง"):
                new_channel_list = []
                for index, row in edited_channels_df.iterrows():
                    if row['Channel Name'] and str(row['Channel Name']).strip() != "":
                        new_channel_list.append(row['Channel Name'])
                save_channels(new_channel_list)
                st.success("✅ อัปเดตรายชื่อช่องทางเรียบร้อย!")

        with tab_train:
            st.subheader("🧠 สั่งให้โมเดลเรียนรู้ใหม่")
            st.write(f"Current Accuracy (R²): {metrics['xgb']['r2']:.4f}")
            if st.button("🚀 Start Retraining", type="primary"):
                success, count = retrain_system()
                if success: st.success(f"Done! Trained on {count} rows."); time.sleep(1); st.rerun()

    def show_pricing_page():
        st.title("🔮 ระบบพยากรณ์ราคา (Price Forecasting)")
        if xgb_model is None: st.error("❌ Model not found (Please Retrain first)"); return

        def get_base_price_safe(room_text):
            return get_base_price(room_text)

        def calculate_clamped_price(model, input_df, room_name_selected, n_nights):
            predicted_price = model.predict(input_df)[0]
            base_per_night = get_base_price_safe(room_name_selected)
            floor_price = base_per_night * n_nights
            final_price = max(predicted_price, floor_price)
            return final_price, predicted_price, floor_price

        with st.container(border=True):
            st.subheader("🛠️ กำหนดเงื่อนไขการจอง")
            c1, c2 = st.columns(2)
            with c1:
                date_range = st.date_input("Select Dates", value=[], min_value=None)
            
            nights = 1
            is_h = False
            checkin_date = datetime.now()
            
            if len(date_range) == 2:
                checkin_date = date_range[0]
                checkout_date = date_range[1]
                nights = (checkout_date - checkin_date).days
                if nights < 1: nights = 1
                current_date = checkin_date
                while current_date < checkout_date:
                    if current_date in holidays.Thailand(): is_h = True; break
                    current_date += timedelta(days=1)
            elif len(date_range) == 1:
                checkin_date = date_range[0]
            
            with c2:
                col_night, col_hol = st.columns(2)
                with col_night: st.number_input("Nights", value=nights, disabled=True)
                with col_hol:
                    manual_holiday = st.checkbox("Holiday (วันหยุด)", value=is_h)
                    final_is_holiday = 1 if manual_holiday else 0

            c3, c4, c5 = st.columns(3)
            with c3:
                prices = load_base_prices()
                room_display_map = {"All (เลือกทั้งหมด)": "All"}
                for r in prices:
                    room_display_map[f"{r} (Base: {prices[r]:,.0f})"] = r
                selected_room_display = st.selectbox("Room Type", list(room_display_map.keys()))
                selected_room_val = room_display_map[selected_room_display]

            with c4:
                max_g = 4
                guests = st.number_input(f"Guests", min_value=1, max_value=10, value=2)

            with c5:
                channels = load_channels()
                res_options = ["All (เลือกทั้งหมด)"] + channels
                selected_res = st.selectbox("Channel", res_options)
                selected_res_val = "All" if "All" in selected_res else selected_res

            if st.button("🚀 คำนวณราคา (Predict)", type="primary", use_container_width=True):
                if selected_room_val == "All" or selected_res_val == "All":
                    st.info(f"📊 รายงานผลการพยากรณ์รวม")
                    st.warning("Batch Predict Mode available in full version")
                else:
                    try:
                        r_code = le_room.transform([selected_room_val])[0]
                        res_code = le_res.transform([selected_res_val])[0]
                        
                        inp_norm = pd.DataFrame([{
                            'Night': nights, 'total_guests': guests, 
                            'is_holiday': final_is_holiday, 'is_weekend': 1 if checkin_date.weekday() in [5,6] else 0,
                            'month': checkin_date.month, 'weekday': checkin_date.weekday(),
                            'RoomType_encoded': r_code, 'Reservation_encoded': res_code
                        }])
                        
                        p_xgb, raw_xgb, floor_p = calculate_clamped_price(xgb_model, inp_norm, selected_room_val, nights)
                        p_lr, raw_lr, _ = calculate_clamped_price(lr_model, inp_norm, selected_room_val, nights)
                        
                        st.divider()
                        st.markdown(f"### 🏨 Room: **{selected_room_val}**")
                        c_res1, c_res2 = st.columns(2)
                        with c_res1: st.metric("XGBoost Price", f"{p_xgb:,.0f} THB", f"Floor: {floor_p:,.0f}")
                        with c_res2: st.metric("Linear Price", f"{p_lr:,.0f} THB")

                    except Exception as e:
                        st.error(f"Prediction Error: {e} (Try Retraining)")

    def show_model_insight_page():
        st.title("🧠 Model Insight")
        st.info("Feature Importance Display")

    def show_about_page():
        st.title("ℹ️ About")
        st.info("System Info")

    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2933/2933116.png", width=80)
        st.markdown(f"### User: {st.session_state['username']}")
        page = st.radio("เมนูใช้งาน:", ["📊 แดชบอร์ด", "📥 จัดการข้อมูล", "🔮 พยากรณ์ราคา", "🧠 วิเคราะห์โมเดล", "ℹ️ เกี่ยวกับระบบ"])
        st.divider()
        if st.button("Logout"): st.session_state['logged_in'] = False; st.rerun()

    if "แดชบอร์ด" in page: show_dashboard_page()
    elif "จัดการข้อมูล" in page: show_manage_data_page()
    elif "พยากรณ์ราคา" in page: show_pricing_page()
    elif "วิเคราะห์โมเดล" in page: show_model_insight_page()
    elif "เกี่ยวกับระบบ" in page: show_about_page()
