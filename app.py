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
    page_title="Hotel Price Forecasting System",
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

# ค่าเริ่มต้น (ใช้กรณีไม่มีไฟล์)
DEFAULT_BASE_PRICES = {
    'Grand Suite Room': 2700,
    'Villa Suite (Garden)': 2700,
    'Executive Room': 2500,
    'Executive Room with Balcony': 2400,
    'Villa Suite (Bathtub)': 2000,
    'Deluxe Room': 1500,
    'Standard Room': 1000
}

# ไม่ Fix ตายตัวแล้ว แต่เป็นค่าตั้งต้นเฉยๆ (User แก้ได้)
DEFAULT_CHANNELS = ["Agoda", "Booking.com", "Traveloka", "Walk-in", "Direct", "Expedia", "Trip.com", "Direct Booking"]

DEFAULT_METRICS = {
    'xgb': {'mae': 1112.79, 'r2': 0.7256},
    'lr':  {'mae': 1162.27, 'r2': 0.7608},
    'importance': {}
}

if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'username' not in st.session_state: st.session_state['username'] = ""

# ==========================================================
# 2. HELPER FUNCTIONS (ส่วนจัดการ Master Data)
# ==========================================================
def load_base_prices():
    if not os.path.exists(BASE_PRICE_FILE):
        with open(BASE_PRICE_FILE, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_BASE_PRICES, f, ensure_ascii=False, indent=4)
        return DEFAULT_BASE_PRICES
    try:
        with open(BASE_PRICE_FILE, 'r', encoding='utf-8') as f: return json.load(f)
    except: return DEFAULT_BASE_PRICES

def save_base_prices(price_dict):
    with open(BASE_PRICE_FILE, 'w', encoding='utf-8') as f:
        json.dump(price_dict, f, ensure_ascii=False, indent=4)

def load_channels():
    if not os.path.exists(CHANNELS_FILE):
        with open(CHANNELS_FILE, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_CHANNELS, f, ensure_ascii=False, indent=4)
        return DEFAULT_CHANNELS
    try:
        with open(CHANNELS_FILE, 'r', encoding='utf-8') as f: return json.load(f)
    except: return DEFAULT_CHANNELS

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
# 4. BACKEND SYSTEM
# ==========================================================

@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE):
        try: gdown.download("https://drive.google.com/uc?id=1dxgKIvSTelLaJvAtBSCMCU5K4FuJvfri", DATA_FILE, quiet=True)
        except: return pd.DataFrame()

    try:
        df = pd.read_csv(DATA_FILE)
        
        # 1. Date Processing
        if 'Date' in df.columns:
            # พยายามอ่าน Date ให้ได้มากที่สุด
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            df = df.dropna(subset=['Date'])
            df['is_weekend'] = df['Date'].dt.weekday.isin([5, 6]).astype(int)
            df['Year'] = df['Date'].dt.year.astype(int)
            df['month'] = df['Date'].dt.month
            
        if 'Room' in df.columns:
            df['Room'] = df['Room'].astype(str)

        # 2. Room Type Mapping (สำคัญมาก: แปลงเลขห้องเป็นชื่อห้อง)
        if os.path.exists(ROOM_FILE):
            room_type = pd.read_csv(ROOM_FILE)
            if 'Room' in room_type.columns: room_type['Room'] = room_type['Room'].astype(str)
            
            # Logic การ Merge แบบเดิมที่ถูกต้อง
            if 'Room_Type' in room_type.columns:
                df = df.merge(room_type, on='Room', how='left')
                if 'Room_Type' in df.columns: df = df.rename(columns={'Room_Type': 'Target_Room_Type'})
                elif 'Room_Type_y' in df.columns: df = df.rename(columns={'Room_Type_y': 'Target_Room_Type'})
            elif 'Target_Room_Type' in room_type.columns:
                df = df.merge(room_type[['Room', 'Target_Room_Type']], on='Room', how='left')
        
        # 3. Filter Outlier
        if 'Target_Room_Type' in df.columns:
            df = df.dropna(subset=['Target_Room_Type'])
            df['Target_Room_Type'] = df['Target_Room_Type'].fillna(df['Room'])
        else:
            df['Target_Room_Type'] = df['Room']
        
        df['Reservation'] = df['Reservation'].fillna('Unknown')
        
        return df
    except: return pd.DataFrame()

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

# 🔥 VALIDATOR SYSTEM (แก้ไขใหม่: ต้อง Map ก่อนตรวจ)
def validate_data_only(df_to_check):
    # 1. โหลดข้อมูลอ้างอิง
    valid_rooms_names = set(load_base_prices().keys()) 
    valid_channels = set(load_channels())             
    
    # 2. สร้าง Mapper (เลขห้อง -> ชื่อห้อง)
    room_mapper = {}
    if os.path.exists(ROOM_FILE):
        try:
            rt = pd.read_csv(ROOM_FILE)
            if 'Room' in rt.columns:
                target_col = 'Target_Room_Type' if 'Target_Room_Type' in rt.columns else 'Room_Type'
                if target_col in rt.columns:
                    rt['Room'] = rt['Room'].astype(str)
                    room_mapper = pd.Series(rt[target_col].values, index=rt['Room']).to_dict()
        except: pass

    df_clean = df_to_check.copy()
    
    # 3. แปลงค่า (Mapping)
    if 'Room' in df_clean.columns:
        df_clean['Room'] = df_clean['Room'].astype(str)
        # เปลี่ยนเลขเป็นชื่อ (ถ้ามีใน Mapper)
        df_clean['Room'] = df_clean['Room'].map(lambda x: room_mapper.get(x, x))
        
    if 'Reservation' in df_clean.columns:
        df_clean['Reservation'] = df_clean['Reservation'].astype(str)

    # 4. ตรวจสอบวันที่
    if 'Date' in df_clean.columns:
        df_clean['Date_Parsed'] = pd.to_datetime(df_clean['Date'], dayfirst=True, errors='coerce')
    else:
        df_clean['Date_Parsed'] = pd.NaT

    # 5. ตรวจสอบสิทธิ์ (Gatekeeper)
    mask_date = df_clean['Date_Parsed'].notna()
    mask_room = df_clean['Room'].isin(valid_rooms_names)
    mask_channel = df_clean['Reservation'].isin(valid_channels)

    # 6. รายงานสาเหตุ
    df_clean['Error_Reason'] = ""
    df_clean.loc[~mask_date, 'Error_Reason'] += "Date Invalid; "
    df_clean.loc[~mask_room, 'Error_Reason'] += "Room Unknown (Update Base Price); "
    df_clean.loc[~mask_channel, 'Error_Reason'] += "Channel Unknown (Update Channels); "

    mask_valid = mask_date & mask_room & mask_channel
    
    df_good = df_clean[mask_valid].copy()
    df_bad = df_clean[~mask_valid].copy()
    
    return df_good, df_bad

def save_dataframe_to_file(df_good):
    if not df_good.empty:
        save_cols = ['Date', 'Room', 'Price', 'Reservation', 'Name', 'Night', 'Adults', 'Children', 'Infants', 'Extra Person']
        
        # Save เป็น YYYY-MM-DD
        if 'Date_Parsed' in df_good.columns:
             df_good['Date'] = df_good['Date_Parsed'].dt.strftime('%Y-%m-%d')
        
        final_cols = [c for c in save_cols if c in df_good.columns]
        df_good[final_cols].to_csv(DATA_FILE, index=False)
        st.cache_data.clear()
        return True
    return False

def retrain_system():
    status_text = st.empty()
    progress_bar = st.progress(0)
    try:
        status_text.text("⏳ Reading & Cleaning data...")
        df = load_data() # ใช้ load_data ตัวเดิมที่เสถียร
        
        if df.empty: return False, 0
        df = df.dropna(subset=['Price', 'Night'])
        
        df['Night'] = df['Night'].fillna(1)
        df['Adults'] = df['Adults'].fillna(2)
        df['Children'] = df['Children'].fillna(0)
        df['Infants'] = df['Infants'].fillna(0)
        df['Extra Person'] = df['Extra Person'].fillna(0)
        
        if not os.path.exists("thai_holidays.csv"):
             try: gdown.download("https://drive.google.com/uc?id=1L-pciKEeRce1gzuhdtpIGcLs0fYHnbZw", "thai_holidays.csv", quiet=True)
             except: pass
        if os.path.exists("thai_holidays.csv"):
            holidays_csv = pd.read_csv("thai_holidays.csv")
            holidays_csv['Holiday_Date'] = pd.to_datetime(holidays_csv['Holiday_Date'], dayfirst=True, errors='coerce')
            df['is_holiday'] = df['Date'].isin(holidays_csv['Holiday_Date']).astype(int)
        else: df['is_holiday'] = 0

        df['is_weekend'] = df['Date'].dt.weekday.isin([5, 6]).astype(int)
        df['total_guests'] = df[['Adults', 'Children', 'Infants', 'Extra Person']].sum(axis=1)
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
        
        fi_raw = xgb_new.feature_importances_
        col_mapping = {'Night': 'Night', 'total_guests': 'Guests', 'is_holiday': 'Is Holiday', 'is_weekend': 'Is Weekend', 'month': 'Month', 'weekday': 'Weekday', 'RoomType_encoded': 'Room Type', 'Reservation_encoded': 'Reservation'}
        new_importance = {col_mapping.get(col, col): float(val) for col, val in zip(feature_cols, fi_raw)}

        lr_new = LinearRegression()
        lr_new.fit(X_train, y_train)
        pred_lr = lr_new.predict(X_test)
        new_lr_mae = mean_absolute_error(y_test, pred_lr)
        new_lr_r2 = r2_score(y_test, pred_lr)
        
        progress_bar.progress(80)
        joblib.dump(xgb_new, MODEL_FILES['xgb'])
        joblib.dump(lr_new, MODEL_FILES['lr'])
        joblib.dump(le_room_new, MODEL_FILES['le_room'])
        joblib.dump(le_res_new, MODEL_FILES['le_res'])
        
        new_metrics = {
            'xgb': {'mae': new_xgb_mae, 'r2': new_xgb_r2},
            'lr':  {'mae': new_lr_mae, 'r2': new_lr_r2},
            'importance': new_importance
        }
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
        u = st.text_input("Username"); p = st.text_input("Password", type="password")
        if st.button("Login", type="primary", use_container_width=True):
            if login_user(u, p): st.session_state['logged_in'] = True; st.session_state['username'] = u; st.rerun()
            else: st.error("Invalid Login")

if not st.session_state['logged_in']:
    login_page()
else:
    df_raw = load_data() 
    xgb_model, lr_model, le_room, le_res, metrics = load_system_models()
    
    def show_dashboard_page():
        # [KEEPING ORIGINAL DASHBOARD CODE EXACTLY AS IS]
        st.title("📊 Financial Executive Dashboard")
        if df_raw.empty: st.warning("No Data Found"); return

        with st.expander("🔎 Filter Data (ตัวกรองข้อมูล)", expanded=True):
            f_col1, f_col2, f_col3 = st.columns(3)
            all_years = sorted(df_raw['Year'].unique().tolist())
            year_opts = ['All'] + [str(y) for y in all_years]
            with f_col1: sel_year = st.selectbox("📅 Select Year (เลือกปี)", year_opts)
            
            all_months = sorted(df_raw['month'].unique().tolist())
            month_opts = ['All'] + [datetime(2024, m, 1).strftime('%B') for m in all_months]
            with f_col2: sel_month_str = st.selectbox("🗓️ Select Month (เลือกเดือน)", month_opts)

            df_filtered = df_raw.copy()
            if sel_year != 'All': df_filtered = df_filtered[df_filtered['Year'] == int(sel_year)]
            if sel_month_str != 'All':
                sel_month_num = datetime.strptime(sel_month_str, "%B").month
                df_filtered = df_filtered[df_filtered['month'] == sel_month_num]

        if df_filtered.empty: st.warning("⚠️ No data available for the selected filters."); return

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
                st.subheader("Revenue vs Nights")
                room_perf = df_filtered.groupby(group_col).agg({'Price': 'sum', 'Night': 'sum'}).reset_index().sort_values('Price', ascending=False)
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Bar(x=room_perf[group_col], y=room_perf['Price'], name="Revenue", marker_color='#1f77b4'), secondary_y=False)
                fig.add_trace(go.Scatter(x=room_perf[group_col], y=room_perf['Night'], name="Nights", mode='lines+markers', marker_color='#ff7f0e'), secondary_y=True)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.subheader("Booking Trend")
                monthly = df_filtered.groupby('month').agg({'Price': 'sum', 'Room': 'count'}).reset_index().sort_values('month')
                monthly['M_Name'] = monthly['month'].apply(lambda x: datetime(2024, int(x), 1).strftime('%b'))
                fig2 = make_subplots(specs=[[{"secondary_y": True}]])
                fig2.add_trace(go.Scatter(x=monthly['M_Name'], y=monthly['Price'], name="Revenue", line=dict(color='green', width=3)), secondary_y=False)
                fig2.add_trace(go.Scatter(x=monthly['M_Name'], y=monthly['Room'], name="Bookings", line=dict(color='blue', dash='dot')), secondary_y=True)
                st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            st.markdown("### 2. Channel Strategy")
            c3, c4 = st.columns(2)
            with c3:
                st.subheader("Revenue Share")
                res_rev = df_filtered.groupby('Reservation')['Price'].sum().reset_index()
                st.plotly_chart(px.pie(res_rev, values='Price', names='Reservation', hole=0.4), use_container_width=True)
            with c4:
                st.subheader("Monthly Channel")
                m_res = df_filtered.groupby(['month', 'Reservation']).size().reset_index(name='Count')
                m_res['M_Name'] = m_res['month'].apply(lambda x: datetime(2024, int(x), 1).strftime('%b'))
                st.plotly_chart(px.bar(m_res, x='M_Name', y='Count', color='Reservation'), use_container_width=True)

        with tab3:
            st.markdown("### 3. Product & Behavior")
            c5, c6 = st.columns(2)
            with c5:
                st.subheader("Room Revenue")
                mt_room = df_filtered.groupby(['month', group_col])['Price'].sum().reset_index()
                mt_room['M_Name'] = mt_room['month'].apply(lambda x: datetime(2024, int(x), 1).strftime('%b'))
                st.plotly_chart(px.bar(mt_room, x='M_Name', y='Price', color=group_col), use_container_width=True)
            with c6:
                st.subheader("Weekday vs Weekend")
                df_filtered['DayType'] = df_filtered['is_weekend'].map({1: 'Weekend', 0: 'Weekday'})
                day_rev = df_filtered.groupby('DayType')['Price'].sum().reset_index()
                st.plotly_chart(px.pie(day_rev, values='Price', names='DayType', hole=0.4), use_container_width=True)

        st.divider()
        st.subheader("📋 Raw Data Explorer")
        with st.expander("คลิกเพื่อดูตารางข้อมูล"): st.dataframe(df_filtered)

    def show_manage_data_page():
        st.title("📥 ระบบจัดการฐานข้อมูล (Management Center)")
        
        tab_trans, tab_master, tab_channel, tab_train = st.tabs([
            "📝 ข้อมูลการจอง", "⚙️ ราคาฐาน", "📢 ช่องทางการจอง", "🚀 Retrain"
        ])

        # TAB 1: Transactions (กู้คืนหน้า Add/Edit/Delete แต่ Logic ปลอดภัย)
        with tab_trans:
            st.subheader("1. นำเข้าข้อมูล (Import)")
            up_file = st.file_uploader("เลือกไฟล์ CSV", type=['csv'])
            if up_file is not None:
                if st.button("➕ Merge File", type="secondary"):
                    try:
                        new_df = pd.read_csv(up_file)
                        if os.path.exists(DATA_FILE):
                            old_df = pd.read_csv(DATA_FILE)
                            merged_df = pd.concat([old_df, new_df], ignore_index=True)
                        else: merged_df = new_df
                        
                        # Validate ก่อน Save
                        good_df, bad_df = validate_data_only(merged_df)
                        
                        if not bad_df.empty:
                            st.warning(f"⚠️ พบข้อมูลที่ไม่ถูกต้อง {len(bad_df)} รายการ")
                            with st.expander("🔴 ดูรายการที่ถูกตัดออก"): 
                                st.dataframe(bad_df[['Date', 'Room', 'Reservation', 'Error_Reason']])
                            if st.button("🚨 ยืนยันบันทึกเฉพาะข้อมูลที่ถูกต้อง (ข้อมูลผิดจะหายไป)"):
                                save_dataframe_to_file(good_df)
                                st.success("บันทึกเรียบร้อย!"); time.sleep(1); st.rerun()
                        else:
                            save_dataframe_to_file(good_df)
                            st.success(f"✅ บันทึกข้อมูล {len(good_df)} รายการ เรียบร้อย!"); time.sleep(1); st.rerun()
                    except Exception as e: st.error(f"Error: {e}")

            st.divider()
            st.subheader("2. แก้ไขข้อมูล (Safe Mode)")
            
            # โหลด Raw String เพื่อให้แก้ได้อิสระ
            df_current = pd.read_csv(DATA_FILE) if os.path.exists(DATA_FILE) else pd.DataFrame()
            
            edited_df = st.data_editor(
                df_current,
                num_rows="dynamic",
                use_container_width=True,
                key="booking_editor_final"
            )

            col_save, col_reset = st.columns([1, 4])
            with col_save:
                if st.button("🔍 ตรวจสอบและบันทึก", type="primary"):
                    good_df, bad_df = validate_data_only(edited_df)
                    
                    if not bad_df.empty:
                        st.error(f"⛔ พบข้อผิดพลาด {len(bad_df)} รายการ")
                        st.dataframe(bad_df[['Date', 'Room', 'Reservation', 'Error_Reason']])
                        st.warning("กดปุ่มยืนยันด้านล่างเพื่อบันทึกเฉพาะข้อมูลที่ถูกต้อง")
                        st.session_state['pending_save'] = True
                        st.session_state['pending_good_df'] = good_df
                    else:
                        save_dataframe_to_file(good_df)
                        st.success("✅ บันทึกสมบูรณ์"); time.sleep(1); st.rerun()

                if st.session_state.get('pending_save'):
                    if st.button("🚨 ยืนยันบันทึก (ทิ้งข้อมูลที่ผิด)"):
                        save_dataframe_to_file(st.session_state['pending_good_df'])
                        del st.session_state['pending_save']
                        st.success("บันทึกเรียบร้อย"); time.sleep(1); st.rerun()
            
            with col_reset:
                if st.button("🧨 ล้างข้อมูลทั้งหมด"):
                     if os.path.exists(DATA_FILE): os.remove(DATA_FILE); st.cache_data.clear(); st.rerun()

        # TAB 2: Base Price
        with tab_master:
            st.subheader("⚙️ ราคาฐาน (และชื่อห้อง)")
            current_prices = load_base_prices()
            df_prices = pd.DataFrame(list(current_prices.items()), columns=['Room Type', 'Base Price'])
            edited_prices_df = st.data_editor(df_prices, num_rows="dynamic", use_container_width=True)
            if st.button("💾 บันทึกราคาฐาน"):
                new_prices_dict = {row['Room Type']: row['Base Price'] for i, row in edited_prices_df.iterrows() if str(row['Room Type']).strip()}
                save_base_prices(new_prices_dict)
                st.success("✅ อัปเดตเรียบร้อย")

        # TAB 3: Channels (แก้ปัญหา Agoda หาย)
        with tab_channel:
            st.subheader("📢 จัดการช่องทางการจอง")
            current_channels = load_channels()
            df_channels = pd.DataFrame(current_channels, columns=['Channel Name'])
            edited_channels_df = st.data_editor(df_channels, num_rows="dynamic", use_container_width=True)
            if st.button("💾 บันทึกช่องทาง"):
                new_channel_list = [row['Channel Name'] for i, row in edited_channels_df.iterrows() if str(row['Channel Name']).strip()]
                save_channels(new_channel_list)
                st.success("✅ อัปเดตเรียบร้อย")

        # TAB 4: Retrain
        with tab_train:
            st.subheader("🚀 Retrain Model")
            if st.button("Start Retraining"):
                success, count = retrain_system()
                if success: st.success(f"Done! ({count} rows)"); time.sleep(1); st.rerun()

    def show_pricing_page():
        st.title("🔮 ระบบพยากรณ์ราคา (Price Forecasting)")
        if xgb_model is None: st.error("❌ Model not found"); return

        def calculate_clamped_price(model, input_df, room_name_selected, n_nights):
            predicted_price = model.predict(input_df)[0]
            base_p = get_base_price(room_name_selected)
            # Floor Price = Base * Nights (แบบเดิม)
            floor_price = base_p * n_nights
            final_price = max(predicted_price, floor_price)
            return final_price, predicted_price, floor_price

        with st.container(border=True):
            st.subheader("🛠️ กำหนดเงื่อนไขการจอง")
            c1, c2 = st.columns(2)
            with c1: date_range = st.date_input("Select Dates", value=[], min_value=None)
            
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
            elif len(date_range) == 1: checkin_date = date_range[0]
            
            with c2:
                col_night, col_hol = st.columns(2)
                with col_night: st.number_input("Nights", value=nights, disabled=True)
                with col_hol:
                    manual_holiday = st.checkbox("Holiday (วันหยุด)", value=is_h)
                    final_is_holiday = 1 if manual_holiday else 0

            c3, c4, c5 = st.columns(3)
            with c3:
                # ใช้ห้องจาก Model ที่เทรนมา
                room_list = list(le_room.classes_)
                selected_room_val = st.selectbox("Room Type", ["All"] + room_list)
            
            with c4:
                # 📢 กู้คืน Logic คนเกิน (+500)
                base_guests = 2
                guests = st.number_input(f"Guests", min_value=1, max_value=10, value=2)
                extra_charge = 0
                if guests > base_guests: extra_charge = (guests - base_guests) * 500

            with c5:
                # ใช้ช่องทางจากไฟล์ channels.json (ไม่ใช่ Hardcode)
                channels = load_channels()
                selected_res_val = st.selectbox("Channel", ["All"] + channels)

            if st.button("🚀 คำนวณราคา (Predict)", type="primary"):
                if selected_room_val == "All" or selected_res_val == "All":
                    st.warning("Batch Predict Mode Not Implemented in this simplified version")
                else:
                    try:
                        r_code = le_room.transform([selected_room_val])[0]
                        # แก้ปัญหาถ้า Channel ในไฟล์ JSON ไม่มีใน Model ให้ใช้ Mode แทน
                        try: res_code = le_res.transform([selected_res_val])[0]
                        except: res_code = 0 
                        
                        inp_norm = pd.DataFrame([{
                            'Night': nights, 'total_guests': guests, 
                            'is_holiday': final_is_holiday, 'is_weekend': 1 if checkin_date.weekday() in [5,6] else 0,
                            'month': checkin_date.month, 'weekday': checkin_date.weekday(),
                            'RoomType_encoded': r_code, 'Reservation_encoded': res_code
                        }])
                        
                        p_xgb, raw_xgb, floor_p = calculate_clamped_price(xgb_model, inp_norm, selected_room_val, nights)
                        p_lr, raw_lr, _ = calculate_clamped_price(lr_model, inp_norm, selected_room_val, nights)

                        # 📢 กู้คืนตัวคูณ (Multipliers)
                        base_p = get_base_price(selected_room_val)
                        multiplier = 1.0
                        if final_is_holiday: multiplier = 1.5
                        elif checkin_date.weekday() in [4, 5]: multiplier = 1.2
                        
                        standard_price = (base_p * multiplier * nights) + (extra_charge * nights)
                        
                        # Compare
                        final_xgb = max(p_xgb, standard_price)

                        st.divider()
                        st.markdown(f"### 🏨 Room: **{selected_room_val}**")
                        c1, c2, c3 = st.columns(3)
                        with c1: st.metric("🤖 AI Price", f"{final_xgb:,.0f} THB")
                        with c2: st.metric("📏 Standard Price", f"{standard_price:,.0f} THB")
                        with c3: 
                            if extra_charge > 0: st.warning(f"+Extra Guest: {extra_charge}")
                            if multiplier > 1: st.info(f"Multiplier: x{multiplier}")

                    except Exception as e: st.error(f"Error: {e}")

    def show_model_insight_page():
        st.title("🧠 วิเคราะห์ปัจจัยโมเดล")
        imp_data = metrics.get('importance', DEFAULT_METRICS['importance'])
        fi_df = pd.DataFrame(list(imp_data.items()), columns=['Feature', 'Importance']).sort_values('Importance', ascending=True)
        st.plotly_chart(px.bar(fi_df, x='Importance', y='Feature', orientation='h'), use_container_width=True)

    def show_about_page():
        st.title("ℹ️ เกี่ยวกับระบบ"); st.info("Hotel Price Forecasting System")

    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2933/2933116.png", width=80)
        page = st.radio("เมนูใช้งาน:", ["📊 แดชบอร์ด", "📥 จัดการข้อมูล", "🔮 พยากรณ์ราคา", "🧠 วิเคราะห์โมเดล", "ℹ️ เกี่ยวกับระบบ"])
        if st.button("Logout"): st.session_state['logged_in'] = False; st.rerun()

    if "แดชบอร์ด" in page: show_dashboard_page()
    elif "จัดการข้อมูล" in page: show_manage_data_page()
    elif "พยากรณ์ราคา" in page: show_pricing_page()
    elif "วิเคราะห์โมเดล" in page: show_model_insight_page()
    elif "เกี่ยวกับระบบ" in page: show_about_page()
