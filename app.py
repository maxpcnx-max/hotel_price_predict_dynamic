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
    page_title="Hotel Price Forecasting System (Master Data)",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_FILE = "users.db"
DATA_FILE = "check_in_report.csv"
ROOM_FILE = "room_type.csv" # ไฟล์จับคู่เลขห้อง -> ชื่อห้อง
METRICS_FILE = "model_metrics.json"
BASE_PRICE_FILE = "base_prices.json" # ไฟล์ราคาฐาน
CHANNELS_FILE = "channels.json" # ไฟล์ช่องทาง

MODEL_FILES = {
    'xgb': 'xgb_hotel_model.joblib',
    'lr': 'lr_hotel_model.joblib',
    'le_room': 'le_room.joblib',
    'le_res': 'le_res.joblib'
}

# --- ค่า Default ห้องพัก ---
DEFAULT_BASE_PRICES = {
    'Grand Suite Room': 2700,
    'Villa Suite (Garden)': 2700,
    'Executive Room': 2500,
    'Executive Room with Balcony': 2400,
    'Villa Suite (Bathtub)': 2000,
    'Deluxe Room': 1500,
    'Standard Room': 1000
}

# --- ค่า Default ช่องทาง ---
DEFAULT_CHANNELS = [
    "Direct Booking", "Trip.com", "Expedia", "Booking.com", "B2B", "Traveloka.com"
]

DEFAULT_METRICS = {
    'xgb': {'mae': 0, 'r2': 0},
    'lr':  {'mae': 0, 'r2': 0},
    'importance': {}
}

if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'username' not in st.session_state: st.session_state['username'] = ""
if 'historical_avg' not in st.session_state: st.session_state['historical_avg'] = {}

# ==========================================================
# 2. HELPER FUNCTIONS (Master Data)
# ==========================================================
def load_base_prices():
    if not os.path.exists(BASE_PRICE_FILE):
        return DEFAULT_BASE_PRICES.copy()
    try:
        with open(BASE_PRICE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return DEFAULT_BASE_PRICES.copy()

def save_base_prices(price_dict):
    with open(BASE_PRICE_FILE, 'w', encoding='utf-8') as f:
        json.dump(price_dict, f, ensure_ascii=False, indent=4)

def get_base_price(room_text):
    if not isinstance(room_text, str): return 0
    prices = load_base_prices()
    if room_text in prices: return prices[room_text]
    for key in prices:
        if key in room_text: return prices[key]
    return 0

def load_channels():
    if not os.path.exists(CHANNELS_FILE):
        return DEFAULT_CHANNELS.copy()
    try:
        with open(CHANNELS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return DEFAULT_CHANNELS.copy()

def save_channels(channel_list):
    with open(CHANNELS_FILE, 'w', encoding='utf-8') as f:
        json.dump(channel_list, f, ensure_ascii=False, indent=4)

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
# 4. DATA HANDLING (Core Logic)
# ==========================================================

def parse_dates_smart(date_series):
    def convert_dt(val):
        if pd.isna(val) or val == '': return pd.NaT
        val_str = str(val).strip()
        try:
            if '-' in val_str and val_str[0:4].isdigit():
                return pd.to_datetime(val_str, yearfirst=True) 
            return pd.to_datetime(val_str, dayfirst=True)
        except:
            return pd.NaT
    return date_series.apply(convert_dt)

def normalize_room_id(val):
    """แปลงเลขห้องให้เป็น format มาตรฐาน (แก้ 1.0 เป็น 1)"""
    try:
        val_float = float(val)
        if val_float.is_integer():
            return str(int(val_float))
        return str(val_float)
    except:
        return str(val).strip()

@st.cache_data
def load_data():
    """โหลดข้อมูล + กรองห้องผีออก (Strict Filter)"""
    # 1. โหลดไฟล์ Transaction
    if not os.path.exists(DATA_FILE):
        try: gdown.download("https://drive.google.com/uc?id=1dxgKIvSTelLaJvAtBSCMCU5K4FuJvfri", DATA_FILE, quiet=True)
        except: return pd.DataFrame()

    try:
        df = pd.read_csv(DATA_FILE)
        
        if 'Date' in df.columns:
            df['Date'] = parse_dates_smart(df['Date'])
            df['is_weekend'] = df['Date'].dt.weekday.isin([5, 6]).astype(int)
            df['Year'] = df['Date'].dt.year
            df['month'] = df['Date'].dt.month
            
        if 'Room' in df.columns:
            df['Room'] = df['Room'].apply(normalize_room_id)

        # 2. โหลดไฟล์ Mapping (room_type.csv)
        # นี่คือหัวใจสำคัญ: ถ้าไม่มีไฟล์นี้ หรือเลขห้องไม่อยู่ในไฟล์นี้ จะถือว่าเป็นห้องผี
        if os.path.exists(ROOM_FILE):
            try:
                room_master = pd.read_csv(ROOM_FILE)
                # Normalize Room ID ใน Master ให้ตรงกัน
                if 'Room' in room_master.columns:
                    room_master['Room'] = room_master['Room'].apply(normalize_room_id)
                
                # เตรียมชื่อ Column เป้าหมาย
                target_col = 'Room_Type' if 'Room_Type' in room_master.columns else 'Target_Room_Type'
                
                if target_col in room_master.columns:
                    # เลือกเฉพาะ Column ที่จำเป็น
                    room_master = room_master[['Room', target_col]]
                    
                    # Merge ข้อมูล: เอา Transaction เป็นตัวตั้ง, เอาชื่อห้องมาแปะ
                    df = df.merge(room_master, on='Room', how='left')
                    
                    # เปลี่ยนชื่อ Column ให้เป็นมาตรฐาน 'Target_Room_Type'
                    if target_col != 'Target_Room_Type':
                        df = df.rename(columns={target_col: 'Target_Room_Type'})
            except: pass

        # 3. STRICT FILTER: กรองข้อมูลขยะ
        if 'Target_Room_Type' in df.columns:
            # ลบแถวที่ Target_Room_Type เป็น NaN (แปลว่าหาใน Master ไม่เจอ)
            df = df.dropna(subset=['Target_Room_Type'])
            # ลบแถวที่เป็น Unknown (เผื่อหลุดมา)
            df = df[df['Target_Room_Type'] != 'Unknown']
        else:
            # ถ้าไม่มีการ Map เลย ให้คืนค่าว่างไปเลย เพื่อความปลอดภัย
            return pd.DataFrame()

        df['Reservation'] = df['Reservation'].fillna('Unknown')
        return df
    except Exception as e:
        print(f"Error: {e}") 
        return pd.DataFrame()

def save_data_robust(new_df, mode='append'):
    try:
        if 'Date' in new_df.columns:
            new_df['Date'] = parse_dates_smart(new_df['Date'])
            new_df['Date'] = new_df['Date'].dt.strftime('%Y-%m-%d')
            
        if mode == 'append':
            if os.path.exists(DATA_FILE):
                current_df = pd.read_csv(DATA_FILE)
                if 'Date' in current_df.columns:
                    current_df['Date'] = parse_dates_smart(current_df['Date']).dt.strftime('%Y-%m-%d')
                updated_df = pd.concat([current_df, new_df], ignore_index=True)
            else:
                updated_df = new_df
        else: 
            updated_df = new_df

        cols_to_keep = ['Date', 'Room', 'Price', 'Reservation', 'Name', 'Night', 'Adults', 'Children', 'Infants', 'Extra Person']
        existing_cols = [c for c in cols_to_keep if c in updated_df.columns]
        
        updated_df[existing_cols].to_csv(DATA_FILE, index=False)
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"Save Error: {e}")
        return False

def calculate_historical_avg(df):
    if df.empty: return {}
    df_clean = df.copy()
    if 'Night' not in df_clean.columns: df_clean['Night'] = 1
    df_clean = df_clean.dropna(subset=['Price', 'Night'])
    df_clean = df_clean[df_clean['Night'] > 0]
    df_clean['ADR_Actual'] = df_clean['Price'] / df_clean['Night']
    
    if 'Target_Room_Type' in df_clean.columns:
        avg_map = df_clean.groupby('Target_Room_Type')['ADR_Actual'].mean().to_dict()
        return avg_map
    return {}

@st.cache_resource
def load_system_models():
    try:
        xgb = joblib.load(MODEL_FILES['xgb'])
        lr = joblib.load(MODEL_FILES['lr'])
        le_room = joblib.load(MODEL_FILES['le_room'])
        le_res = joblib.load(MODEL_FILES['le_res'])
    except:
        xgb, lr, le_room, le_res = None, None, None, None

    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, 'r') as f: metrics = json.load(f)
        except:
            metrics = DEFAULT_METRICS
    else:
        metrics = DEFAULT_METRICS
        
    return xgb, lr, le_room, le_res, metrics

# ==========================================================
# 5. RETRAIN SYSTEM
# ==========================================================
def retrain_system():
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        status_text.text("⏳ Reading data from storage...")
        df = load_data()
        
        if df.empty:
            st.error("ไม่พบข้อมูลสำหรับเทรนโมเดล (กรุณาตรวจสอบ Master Data ว่าจับคู่ห้องครบถ้วนหรือไม่)")
            return False, 0
            
        status_text.text("🧹 Preparing data...")
        df_clean = df.dropna(subset=['Price', 'Night', 'Date'])
        
        # กรองเอาเฉพาะช่องทางที่มีใน Master Data
        if 'Reservation' in df_clean.columns:
             valid_channels = set(load_channels())
             df_clean = df_clean[df_clean['Reservation'].isin(valid_channels)]

        if df_clean.empty:
            st.error("ไม่เหลือข้อมูลหลังจากกรอง (ตรวจสอบช่องทางการขาย)")
            return False, 0

        df_clean['Night'] = df_clean['Night'].fillna(1)
        df_clean['Adults'] = df_clean['Adults'].fillna(2)
        df_clean['Children'] = df_clean['Children'].fillna(0)
        df_clean['Infants'] = df_clean['Infants'].fillna(0)
        df_clean['Extra Person'] = df_clean['Extra Person'].fillna(0)
        
        if not os.path.exists("thai_holidays.csv"):
             try: gdown.download("https://drive.google.com/uc?id=1L-pciKEeRce1gzuhdtpIGcLs0fYHnbZw", "thai_holidays.csv", quiet=True)
             except: pass
        if os.path.exists("thai_holidays.csv"):
            holidays_csv = pd.read_csv("thai_holidays.csv")
            holidays_csv['Holiday_Date'] = parse_dates_smart(holidays_csv['Holiday_Date'])
            df_clean['is_holiday'] = df_clean['Date'].isin(holidays_csv['Holiday_Date']).astype(int)
        else: df_clean['is_holiday'] = 0

        df_clean['is_weekend'] = df_clean['Date'].dt.weekday.isin([5, 6]).astype(int)
        df_clean['total_guests'] = df_clean[['Adults', 'Children', 'Infants', 'Extra Person']].sum(axis=1)
        df_clean['month'] = df_clean['Date'].dt.month
        df_clean['weekday'] = df_clean['Date'].dt.weekday
        
        le_room_new = LabelEncoder()
        df_clean['RoomType_encoded'] = le_room_new.fit_transform(df_clean['Target_Room_Type'].astype(str))
        le_res_new = LabelEncoder()
        df_clean['Reservation_encoded'] = le_res_new.fit_transform(df_clean['Reservation'].astype(str))
        
        feature_cols = ['Night', 'total_guests', 'is_holiday', 'is_weekend', 'month', 'weekday', 'RoomType_encoded', 'Reservation_encoded']
        X = df_clean[feature_cols]
        X = X.fillna(0)
        y = df_clean['Price']
        
        progress_bar.progress(40)
        status_text.text(f"🏋️‍♂️ Training Model on {len(df_clean)} valid rows...")
        
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
        status_text.text("💾 Saving updated intelligence...")
        
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
            
        st.session_state['historical_avg'] = calculate_historical_avg(df_clean)
            
        st.cache_resource.clear()
        progress_bar.progress(100)
        status_text.success(f"✅ Retraining Complete! Cleaned Data: {len(df_clean)} rows. New R²: {new_xgb_r2:.4f}")
        return True, len(df_clean)
        
    except Exception as e:
        st.error(f"Retrain Error: {e}")
        return False, 0

# ==========================================================
# 6. MAIN UI PAGES
# ==========================================================

def login_page():
    st.markdown("""<style>.stTextInput > div > div > input {text-align: center;}</style>""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/2933/2933116.png", width=120)
        st.title("🔒 Login System")
        st.markdown("ระบบการพยากรณ์ราคาห้องพัก (Hotel Price Forecasting System)")
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
    df_raw = load_data() 
    
    if not df_raw.empty and not st.session_state['historical_avg']:
        st.session_state['historical_avg'] = calculate_historical_avg(df_raw)

    xgb_model, lr_model, le_room, le_res, metrics = load_system_models()
    
    def show_dashboard_page():
        st.title("📊 Financial Executive Dashboard")
        
        # เช็คก่อนว่ามีข้อมูลหรือไม่ (ถ้า load_data กรองออกหมด ก็จะว่าง)
        if df_raw.empty: 
            st.warning("⚠️ ไม่พบข้อมูลสำหรับแสดงผล (อาจเป็นเพราะยังไม่ได้จับคู่เลขห้อง หรือข้อมูลถูกกรองออกหมด)")
            st.info("👉 กรุณาไปที่เมนู **'จัดการข้อมูล' -> 'จัดการ Master Data' -> 'จับคู่เลขห้อง'** เพื่อเพิ่มเลขห้องเข้าระบบ")
            return

        with st.expander("🔎 Filter Data (ตัวกรองข้อมูล)", expanded=True):
            f_col1, f_col2, f_col3 = st.columns(3)
            
            valid_years = df_raw['Year'].unique()
            all_years = sorted(valid_years.tolist())
            year_opts = ['All'] + [str(int(y)) for y in all_years]
            with f_col1: sel_year = st.selectbox("📅 Select Year (เลือกปี)", year_opts)
            
            valid_months = df_raw['month'].unique()
            all_months = sorted(valid_months.tolist())
            month_opts = ['All'] + [datetime(2024, int(m), 1).strftime('%B') for m in all_months]
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
        # ใช้ Target_Room_Type (ชื่อห้อง) ในการ Group เสมอ
        group_col = 'Target_Room_Type'

        with tab1:
            st.markdown("### 1. Financial Overview (ภาพรวมการเงิน)")
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Revenue vs Nights")
                room_perf = df_filtered.groupby(group_col).agg({'Price': 'sum', 'Night': 'sum'}).reset_index().sort_values('Price', ascending=False)
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Bar(x=room_perf[group_col], y=room_perf['Price'], name="Revenue", marker_color='#1f77b4'), secondary_y=False)
                fig.add_trace(go.Scatter(x=room_perf[group_col], y=room_perf['Night'], name="Nights", mode='lines+markers', marker_color='#ff7f0e'), secondary_y=True)
                fig.update_layout(legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.subheader("Revenue vs Booking Trend")
                monthly = df_filtered.groupby('month').agg({'Price': 'sum', 'Room': 'count'}).reset_index().sort_values('month')
                monthly['M_Name'] = monthly['month'].apply(lambda x: datetime(2024, int(x), 1).strftime('%b'))
                fig2 = make_subplots(specs=[[{"secondary_y": True}]])
                fig2.add_trace(go.Scatter(x=monthly['M_Name'], y=monthly['Price'], name="Revenue", line=dict(color='green', width=3)), secondary_y=False)
                fig2.add_trace(go.Scatter(x=monthly['M_Name'], y=monthly['Room'], name="Bookings", line=dict(color='blue', dash='dot')), secondary_y=True)
                fig2.update_layout(legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig2, use_container_width=True)
            st.subheader("ADR Trend Analysis (Average Daily Rate)")
            monthly_adr = df_filtered.groupby('month').apply(lambda x: x['Price'].sum() / x['Night'].sum()).reset_index(name='ADR')
            monthly_adr['M_Name'] = monthly_adr['month'].apply(lambda x: datetime(2024, int(x), 1).strftime('%b'))
            fig_adr = px.line(monthly_adr, x='M_Name', y='ADR', markers=True, title="ADR per Month")
            st.plotly_chart(fig_adr, use_container_width=True)

        with tab2:
            st.markdown("### 2. Channel Strategy (เจาะลึกช่องทางการขาย)")
            c3, c4 = st.columns(2)
            with c3:
                st.subheader("Revenue Share by Channel")
                res_rev = df_filtered.groupby('Reservation')['Price'].sum().reset_index()
                st.plotly_chart(px.pie(res_rev, values='Price', names='Reservation', hole=0.4), use_container_width=True)
            with c4:
                st.subheader("Monthly Booking by Channel")
                m_res = df_filtered.groupby(['month', 'Reservation']).size().reset_index(name='Count')
                m_res['M_Name'] = m_res['month'].apply(lambda x: datetime(2024, int(x), 1).strftime('%b'))
                st.plotly_chart(px.bar(m_res, x='M_Name', y='Count', color='Reservation'), use_container_width=True)
            st.subheader("High-Value Customer Channel (ADR)")
            chan_adr = df_filtered.groupby('Reservation').apply(lambda x: x['Price'].sum() / x['Night'].sum()).reset_index(name='ADR').sort_values('ADR', ascending=False)
            st.plotly_chart(px.bar(chan_adr, x='Reservation', y='ADR', color='ADR', color_continuous_scale='Greens'), use_container_width=True)

        with tab3:
            st.markdown("### 3. Product & Behavior (พฤติกรรมลูกค้า)")
            c5, c6 = st.columns(2)
            with c5:
                st.subheader("Monthly Revenue by Room")
                mt_room = df_filtered.groupby(['month', group_col])['Price'].sum().reset_index()
                mt_room['M_Name'] = mt_room['month'].apply(lambda x: datetime(2024, int(x), 1).strftime('%b'))
                st.plotly_chart(px.bar(mt_room, x='M_Name', y='Price', color=group_col), use_container_width=True)
            with c6:
                st.subheader("Channel Preference by Room")
                heatmap_data = df_filtered.groupby([group_col, 'Reservation']).size().unstack(fill_value=0)
                fig_heat = px.imshow(heatmap_data, text_auto=True, aspect="auto", color_continuous_scale='Blues')
                st.plotly_chart(fig_heat, use_container_width=True)
            st.subheader("Weekday vs Weekend Revenue")
            df_filtered['DayType'] = df_filtered['is_weekend'].map({1: 'Weekend', 0: 'Weekday'})
            day_rev = df_filtered.groupby('DayType')['Price'].sum().reset_index()
            c7, c8 = st.columns(2)
            with c7: st.plotly_chart(px.pie(day_rev, values='Price', names='DayType', hole=0.4, title="Revenue Share"), use_container_width=True)
            with c8:
                day_avg = df_filtered.groupby('DayType')['Price'].mean().reset_index()
                st.plotly_chart(px.bar(day_avg, x='DayType', y='Price', title="Avg Booking Value", color='DayType'), use_container_width=True)

        st.divider()
        st.subheader("📋 Raw Data Explorer (Cleaned for Dashboard)")
        with st.expander("คลิกเพื่อดูตารางข้อมูลที่ผ่านการกรองแล้ว"): st.dataframe(df_filtered)

    def show_manage_data_page():
        st.title("📥 ระบบจัดการฐานข้อมูล (Master Data Management)")
        
        tab_trans, tab_master, tab_train = st.tabs(["📝 ข้อมูลการจอง (Transactions)", "⚙️ ข้อมูลหลัก (Master Data)", "🚀 อัปเดตโมเดล (Retrain)"])

        with tab_trans:
            # PART A: Import
            st.subheader("1. นำเข้าข้อมูลใหม่ (Import)")
            st.caption("นำข้อมูลมาต่อท้าย (Append) โดยไม่มีการกรอง")
            up_file = st.file_uploader("เลือกไฟล์ Booking CSV", type=['csv'])
            if up_file is not None:
                if st.button("💾 บันทึกข้อมูลเข้าระบบ"):
                    try:
                        up_file.seek(0)
                        new_data = pd.read_csv(up_file)
                        if save_data_robust(new_data, mode='append'):
                            st.success(f"✅ บันทึกเรียบร้อย! ({len(new_data)} รายการ)"); time.sleep(1); st.rerun()
                    except Exception as e: st.error(f"Error: {e}")

            st.divider()

            # PART B: Edit
            st.subheader("2. แก้ไขข้อมูล (Edit Mode)")
            st.caption("แก้ไขข้อมูลในถังข้อมูลโดยตรง")

            df_current = load_data() # โหลดข้อมูลมาแสดง (อันนี้ผ่านการกรองแล้ว)
            
            # ถ้าอยากโชว์ข้อมูลดิบทั้งหมดให้ user แก้ (รวมถึงอันที่ยังไม่ได้ map) ต้องโหลดแบบ raw
            # แต่เพื่อความปลอดภัย ให้โชว์เท่าที่ load_data เห็นจะดีกว่า หรือไม่ก็ต้องเขียน load_raw แยก
            if df_current.empty:
                st.info("📭 ยังไม่มีข้อมูลที่แสดงผลได้ (ลองตรวจสอบ Master Data)")
            else:
                df_current.columns = df_current.columns.astype(str)
                edited_df = st.data_editor(
                    df_current,
                    num_rows="dynamic",
                    use_container_width=True,
                    key="trans_editor",
                    column_config={
                        "Date": st.column_config.DateColumn("Date", format="DD/MM/YYYY"),
                        "Price": st.column_config.NumberColumn("Price", format="%d THB"),
                        # ซ่อนคอลัมน์ที่คำนวณมา
                        "Target_Room_Type": None 
                    }
                )

                if st.button("💾 บันทึกการเปลี่ยนแปลง (Save All)"):
                    if save_data_robust(edited_df, mode='overwrite'):
                        st.success("✅ บันทึกข้อมูลทั้งหมดเรียบร้อย!"); time.sleep(1); st.rerun()

            st.divider()
            with st.expander("🧨 พื้นที่อันตราย (Danger Zone)"):
                st.warning("⚠️ **คำเตือน:** การกดปุ่มนี้จะลบข้อมูลการจองทั้งหมด, ลบโมเดลที่เคยเทรนมา, และรีเซ็ตค่า Config ทั้งหมดกลับเป็นค่าโรงงาน")
                if st.button("🔥 Factory Reset (ล้างทุกอย่างและคืนค่าเดิม)", type="primary"):
                     if os.path.exists(DATA_FILE): os.remove(DATA_FILE)
                     for key, file_path in MODEL_FILES.items():
                         if os.path.exists(file_path): os.remove(file_path)
                     
                     # Reset Channels & Rooms & Base Prices
                     if os.path.exists(ROOM_FILE): os.remove(ROOM_FILE) # ลบ mapping ด้วย
                     with open(BASE_PRICE_FILE, 'w', encoding='utf-8') as f: json.dump(DEFAULT_BASE_PRICES, f, indent=4)
                     with open(CHANNELS_FILE, 'w', encoding='utf-8') as f: json.dump(DEFAULT_CHANNELS, f, indent=4)
                     with open(METRICS_FILE, 'w', encoding='utf-8') as f: json.dump(DEFAULT_METRICS, f, indent=4)

                     st.cache_data.clear()
                     st.cache_resource.clear()
                     st.success("✅ Factory Reset Complete! ระบบกลับสู่ค่าเริ่มต้นแล้ว")
                     time.sleep(2); st.rerun()

        with tab_master:
            # Layout: 3 Columns
            c1, c2, c3 = st.columns(3)
            
            # --- Column 1: จับคู่เลขห้อง (Room Mapping) ---
            with c1:
                st.subheader("1. จับคู่เลขห้อง")
                st.caption("Map: เลขห้อง (CSV) -> ชื่อห้อง")
                
                # โหลดไฟล์ room_type.csv
                if os.path.exists(ROOM_FILE):
                    df_room_map = pd.read_csv(ROOM_FILE)
                else:
                    df_room_map = pd.DataFrame(columns=['Room', 'Room_Type'])
                
                df_room_map = df_room_map.astype(str) # แปลงเป็น string ให้หมดกัน error

                edited_room_map = st.data_editor(
                    df_room_map,
                    num_rows="dynamic",
                    use_container_width=True,
                    key="room_map_editor",
                    column_config={
                        "Room": st.column_config.TextColumn("เลขห้อง (ID)", required=True),
                        "Room_Type": st.column_config.TextColumn("ชื่อห้อง (Name)", required=True)
                    }
                )
                
                if st.button("💾 บันทึกการจับคู่"):
                    edited_room_map.to_csv(ROOM_FILE, index=False)
                    st.cache_data.clear() # สำคัญ! ต้องเคลียร์ cache เพื่อให้ dashboard เห็นค่าใหม่
                    st.success("✅ บันทึกเรียบร้อย!")
                    time.sleep(0.5); st.rerun()

            # --- Column 2: ราคาฐาน (Base Prices) ---
            with c2:
                st.subheader("2. ราคาฐาน")
                st.caption("กำหนดราคาสำหรับ AI")
                current_prices = load_base_prices()
                df_prices = pd.DataFrame(list(current_prices.items()), columns=['Room Type', 'Base Price'])
                
                edited_prices_df = st.data_editor(
                    df_prices,
                    num_rows="dynamic", 
                    use_container_width=True,
                    column_config={
                        "Room Type": st.column_config.TextColumn("ชื่อห้องพัก (ต้องตรงกับข้อ 1)", required=True),
                        "Base Price": st.column_config.NumberColumn("ราคาฐาน (THB)", format="%d THB", min_value=0, required=True)
                    },
                    key="base_price_editor"
                )
                if st.button("💾 บันทึกราคาฐาน"):
                    new_prices_dict = {row['Room Type']: row['Base Price'] for index, row in edited_prices_df.iterrows() if row['Room Type']}
                    save_base_prices(new_prices_dict)
                    st.success("✅ อัปเดตราคาฐานเรียบร้อย!")

            # --- Column 3: ช่องทาง (Channels) ---
            with c3:
                st.subheader("3. ช่องทางการขาย")
                st.caption("เพิ่ม/ลบ ช่องทาง")
                current_channels = load_channels()
                df_channels = pd.DataFrame(current_channels, columns=['Channel Name'])
                
                edited_channels_df = st.data_editor(
                    df_channels,
                    num_rows="dynamic",
                    use_container_width=True,
                    column_config={
                        "Channel Name": st.column_config.TextColumn("ชื่อช่องทาง", required=True)
                    },
                    key="channel_editor"
                )
                if st.button("💾 บันทึกช่องทาง"):
                    new_channels_list = [row['Channel Name'] for index, row in edited_channels_df.iterrows() if row['Channel Name']]
                    save_channels(new_channels_list)
                    st.success("✅ อัปเดตช่องทางเรียบร้อย!")

        with tab_train:
            st.subheader("🧠 สั่งให้โมเดลเรียนรู้ใหม่ (Retrain Model)")
            st.info("ระบบจะดึงข้อมูลจากถังข้อมูลมา **คัดกรองเฉพาะข้อมูลที่มีคุณภาพ** เพื่อสอน AI")
            col_m1, col_m2 = st.columns(2)
            with col_m1: st.metric("Current Accuracy (R²)", f"{metrics['xgb']['r2']*100:.2f}%")
            if st.button("🚀 เริ่มกระบวนการเรียนรู้ใหม่ (Start Retraining)", type="primary"):
                success, count = retrain_system()
                if success: st.success(f"🎉 เรียนรู้สำเร็จ! ใช้ข้อมูลคุณภาพ {count:,} รายการ"); time.sleep(2); st.rerun()

    def show_pricing_page():
        st.title("🔮 ระบบพยากรณ์ราคา (Price Forecasting)")
        
        # Load Master Data for Dropdowns
        master_rooms = list(load_base_prices().keys())
        master_channels = load_channels()

        def get_historical_avg_price(room_text):
            hist_map = st.session_state.get('historical_avg', {})
            if room_text in hist_map: return hist_map[room_text]
            return 0

        def predict_segmented_price(model, start_date, n_nights, guests, r_code, res_code):
            MAX_CHUNK = 7 
            total_predicted = 0
            remaining_nights = n_nights
            current_date = start_date
            
            while remaining_nights > 0:
                chunk_nights = min(remaining_nights, MAX_CHUNK)
                chunk_end_date = current_date + timedelta(days=chunk_nights)
                
                chunk_is_holiday = 0
                temp_date = current_date
                while temp_date < chunk_end_date:
                    if temp_date in holidays.Thailand():
                        chunk_is_holiday = 1
                        break
                    temp_date += timedelta(days=1)
                
                chunk_is_weekend = 1 if current_date.weekday() in [5, 6] else 0
                
                inp_chunk = pd.DataFrame([{
                    'Night': chunk_nights, 
                    'total_guests': guests, 
                    'is_holiday': chunk_is_holiday, 
                    'is_weekend': chunk_is_weekend,
                    'month': current_date.month, 
                    'weekday': current_date.weekday(),
                    'RoomType_encoded': r_code, 
                    'Reservation_encoded': res_code
                }])
                
                try:
                    chunk_price = model.predict(inp_chunk)[0]
                except:
                    chunk_price = 0 
                
                total_predicted += chunk_price
                remaining_nights -= chunk_nights
                current_date = chunk_end_date
            return total_predicted

        def calculate_rule_based_price(base_per_night, start_date, n_nights, use_holiday, use_weekend):
            th_holidays = holidays.Thailand()
            total_price = 0
            current_date = start_date
            for _ in range(n_nights):
                multiplier = 1.0
                is_weekend = current_date.weekday() in [5, 6]
                is_holiday = current_date in th_holidays
                
                is_near_holiday = False
                for i in range(1, 4):
                    if (current_date + timedelta(days=i)) in th_holidays:
                        is_near_holiday = True; break
                
                if is_holiday and use_holiday:
                    multiplier = 1.7 if (is_weekend and use_weekend) else 1.5
                elif is_weekend and use_weekend:
                    multiplier = 1.56 if (is_near_holiday and use_holiday) else 1.2
                elif is_near_holiday and use_holiday:
                    multiplier = 1.3
                
                total_price += (base_per_night * multiplier)
                current_date += timedelta(days=1)
            return total_price

        def calculate_clamped_price(model, start_date, n_nights, guests, r_code, res_code, room_name_selected, use_h, use_w):
            base_per_night = get_base_price(room_name_selected) 
            rule_price = calculate_rule_based_price(base_per_night, start_date, n_nights, use_h, use_w)
            
            # AI Prediction
            if model is not None:
                raw_predicted = predict_segmented_price(model, start_date, n_nights, guests, r_code, res_code)
            else:
                raw_predicted = 0

            if raw_predicted == 0:
                final_price = rule_price
            else:
                hist_avg = get_historical_avg_price(room_name_selected)
                if hist_avg > 0:
                    hist_total = hist_avg * n_nights
                    offset = raw_predicted - hist_total
                    final_price = rule_price + offset
                else:
                    final_price = rule_price

            total_base = base_per_night * n_nights
            final_price = max(final_price, total_base)
            
            return final_price, raw_predicted, rule_price

        with st.container(border=True):
            col_head, col_status_placeholder = st.columns([1.5, 1]) 
            with col_head: st.subheader("🛠️ กำหนดเงื่อนไขการจอง")
            status_container = col_status_placeholder.container()

            c1, c2 = st.columns([2, 1])
            with c1:
                date_range = st.date_input("Select Dates (Check-in - Check-out)", value=[], min_value=None)
            
            nights = 1
            checkin_date = datetime.now()
            auto_holiday = False
            auto_weekend = False
            
            if len(date_range) == 2:
                checkin_date = date_range[0]
                checkout_date = date_range[1]
                nights = (checkout_date - checkin_date).days
                if nights < 1: nights = 1
                curr = checkin_date
                while curr < checkout_date:
                    if curr in holidays.Thailand(): auto_holiday = True
                    if curr.weekday() in [5, 6]: auto_weekend = True
                    curr += timedelta(days=1)
            elif len(date_range) == 1:
                checkin_date = date_range[0]
            
            with c2: st.number_input("Nights", value=nights, disabled=True)

            with status_container:
                st.write("") 
                sc1, sc2 = st.columns(2)
                with sc1: st.checkbox("หยุดนักขัตฤกษ์", value=auto_holiday, disabled=True)
                with sc2: st.checkbox("เสาร์-อาทิตย์", value=auto_weekend, disabled=True)

            c3, c4, c5 = st.columns(3)
            with c3:
                # Dropdown from Master Data
                room_display_map = {"All (เลือกทั้งหมด)": "All"}
                for r in master_rooms:
                    bp = get_base_price(r) 
                    display_text = f"{r} (Base: {bp:,.0f})"
                    room_display_map[display_text] = r
                selected_room_display = st.selectbox("Room Type", list(room_display_map.keys()))
                selected_room_val = room_display_map[selected_room_display]

            with c4:
                max_g = 4
                if selected_room_val != "All":
                    if "Standard" in str(selected_room_val) or "Deluxe" in str(selected_room_val): max_g = 2
                guests = st.number_input(f"Guests (Max {max_g})", min_value=1, max_value=max_g, value=min(2, max_g))

            with c5:
                # Dropdown from Master Channels
                res_options = ["All (เลือกทั้งหมด)"] + master_channels
                selected_res = st.selectbox("Channel", res_options)
                selected_res_val = "All" if "All" in selected_res else selected_res

            if st.button("🚀 คำนวณราคา (Predict)", type="primary", use_container_width=True):
                use_holiday_val = auto_holiday
                use_weekend_val = auto_weekend

                try:
                    le_room_enc = joblib.load(MODEL_FILES['le_room'])
                    le_res_enc = joblib.load(MODEL_FILES['le_res'])
                except:
                    le_room_enc, le_res_enc = None, None

                if selected_room_val == "All" or selected_res_val == "All":
                    st.info(f"📊 รายงานผลการพยากรณ์รวม (Batch Report)")
                    target_rooms = master_rooms if selected_room_val == "All" else [selected_room_val]
                    target_res = master_channels if selected_res_val == "All" else [selected_res_val]
                    
                    results = []
                    for r_type in target_rooms:
                        base_per_night = get_base_price(r_type)
                        
                        r_code = 0
                        if le_room_enc:
                            try: r_code = le_room_enc.transform([r_type])[0]
                            except: r_code = -1

                        for ch_type in target_res:
                            res_code = 0
                            if le_res_enc:
                                try: res_code = le_res_enc.transform([ch_type])[0]
                                except: res_code = -1

                            final_xgb, _, _ = calculate_clamped_price(xgb_model, checkin_date, nights, guests, r_code, res_code, r_type, use_holiday_val, use_weekend_val)
                            final_lr, _, _ = calculate_clamped_price(lr_model, checkin_date, nights, guests, r_code, res_code, r_type, use_holiday_val, use_weekend_val)
                            
                            results.append({
                                "Room": r_type, "Channel": ch_type, "Guests": guests,
                                "Base Price (Total)": base_per_night * nights, 
                                "XGB Price": final_xgb, "LR Price": final_lr
                            })
                    st.dataframe(pd.DataFrame(results).style.format("{:,.0f}", subset=["Base Price (Total)", "XGB Price", "LR Price"]), use_container_width=True, height=500)

                else:
                    # Single Prediction
                    r_code = 0
                    if le_room_enc:
                        try: r_code = le_room_enc.transform([selected_room_val])[0]
                        except: r_code = -1
                    
                    res_code = 0
                    if le_res_enc:
                        try: res_code = le_res_enc.transform([selected_res_val])[0]
                        except: res_code = -1
                    
                    p_xgb_norm, raw_xgb, _ = calculate_clamped_price(xgb_model, checkin_date, nights, guests, r_code, res_code, selected_room_val, use_holiday_val, use_weekend_val)
                    p_lr_norm, raw_lr, _ = calculate_clamped_price(lr_model, checkin_date, nights, guests, r_code, res_code, selected_room_val, use_holiday_val, use_weekend_val)
                    std_base = get_base_price(selected_room_val) * nights

                    st.divider()
                    st.markdown(f"### 🏨 ผลการวิเคราะห์ราคาห้อง: **{selected_room_val}**")
                    st.caption(f"เงื่อนไข: {nights} คืน | {guests} ท่าน | ช่องทาง {selected_res_val} | Standard Base: {std_base:,.0f} THB")
                    
                    if r_code == -1 or res_code == -1:
                        st.warning("⚠️ ห้องหรือช่องทางนี้ยังไม่เคยผ่านการเทรน AI (ระบบใช้ราคาฐานคำนวณแทน)")

                    r1c1, r1c2 = st.columns(2)
                    with r1c1:
                        diff_xgb = p_xgb_norm - std_base
                        st.container(border=True).metric(
                            label=f"⚡ XGBoost (ปกติ: {guests} ท่าน)",
                            value=f"{p_xgb_norm:,.0f} THB",
                            delta=f"{diff_xgb:+,.0f} THB (vs Base)",
                            delta_color="normal"
                        )
                    
                    with r1c2:
                        diff_lr = p_lr_norm - std_base
                        st.container(border=True).metric(
                            label=f"📉 Linear Regression (ปกติ: {guests} ท่าน)",
                            value=f"{p_lr_norm:,.0f} THB",
                            delta=f"{diff_lr:+,.0f} THB (vs Base)",
                            delta_color="normal"
                        )

                    extra_guests = guests + 1
                    r2c1, r2c2 = st.columns(2)
                    if extra_guests <= max_g:
                        extra_charge = 500 * nights
                        p_xgb_extra = p_xgb_norm + extra_charge
                        p_lr_extra = p_lr_norm + extra_charge
                        
                        with r2c1:
                            st.container(border=True).metric(
                                label=f"👥 XGBoost (เพิ่มแขก: {extra_guests} ท่าน)",
                                value=f"{p_xgb_extra:,.0f} THB",
                                delta=f"+{extra_charge:,.0f} THB (Add-on)",
                                delta_color="normal"
                            )
                        with r2c2:
                            st.container(border=True).metric(
                                label=f"👥 Linear (เพิ่มแขก: {extra_guests} ท่าน)",
                                value=f"{p_lr_extra:,.0f} THB",
                                delta=f"+{extra_charge:,.0f} THB (Add-on)",
                                delta_color="normal"
                            )
                    else:
                        st.warning(f"🚫 ไม่สามารถเพิ่มผู้เข้าพักเป็น {extra_guests} ท่านได้ (Max {max_g})")

    def show_model_insight_page():
        st.title("🧠 วิเคราะห์ปัจจัยโมเดล (Model Factor Analysis)")
        st.markdown("แสดงค่าความสำคัญของตัวแปร (Feature Importance Scores) จากการเรียนรู้ของ AI")

        imp_data = metrics.get('importance', {})
        if not imp_data: imp_data = DEFAULT_METRICS['importance']

        name_mapping = {
            'Night': 'Night (จำนวนคืน)',
            'Reservation': 'Reservation (ช่องทางการจอง)',
            'Month': 'Month (เดือนที่เข้าพัก)',
            'Is Weekend': 'Is Weekend (วันหยุดสุดสัปดาห์)',
            'Room Type': 'Room Type (ประเภทห้องพัก)',
            'Weekday': 'Weekday (วันในสัปดาห์)',
            'Guests': 'Total Guests (ผู้เข้าพักรวม)',
            'Is Holiday': 'Is Holiday (วันหยุดนักขัตฤกษ์)'
        }

        data_list = []
        for key, value in imp_data.items():
            th_name = name_mapping.get(key, key) 
            data_list.append({'Feature': th_name, 'Importance': value})

        fi_df = pd.DataFrame(data_list)
        if fi_df.empty or 'Importance' not in fi_df.columns:
            st.warning("⚠️ ยังไม่มีข้อมูลโมเดล (กรุณากด Retrain Model ที่เมนู 'จัดการข้อมูล' เพื่อเริ่มเรียนรู้)")
            return

        fi_df = fi_df.sort_values('Importance', ascending=True) 

        st.divider()
        st.subheader("กราฟแสดงน้ำหนักความสำคัญของตัวแปร (Dynamic)")

        fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h', 
                     title='Feature Importance Score (อัปเดตล่าสุด)',
                     text_auto='.4f', 
                     color='Importance', 
                     color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("ดูข้อมูลแบบตาราง (Table View)", expanded=True):
            display_df = fi_df.sort_values('Importance', ascending=False)
            display_df['Percentage'] = (display_df['Importance'] * 100).map('{:.2f}%'.format)
            st.dataframe(display_df, use_container_width=True)

        if not display_df.empty:
            top_1 = display_df.iloc[0]
            top_2 = display_df.iloc[1] if len(display_df) > 1 else display_df.iloc[0]
            st.info(f"""
            **💡 ข้อสังเกตจาก AI:**
            * **{top_1['Feature']}:** มีผลต่อราคามากที่สุด ({top_1['Percentage']})
            * **{top_2['Feature']}:** มีผลรองลงมา ({top_2['Percentage']})
            """)

    def show_about_page():
        st.title("ℹ️ เกี่ยวกับระบบ / ผู้จัดทำ")
        st.divider()
        c1, c2 = st.columns([1, 2])
        with c1: 
            if os.path.exists("my_profile.jpg"):
                st.image("my_profile.jpg", width=250)
            else:
                st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=200)
        with c2:
            st.header("ผู้จัดทำ")
            st.markdown("**ว่าที่ร้อยตรีพรพินิต วิรัตน์สกุลชัย** สาขาวิทยาการข้อมูล และ นวัตกรรมดิจิทัล\n\nคณะ นวัตกรรม เทคโนโลยีและการสร้างสรรค์ **มหาวิทยาลัยฟาร์อีสเทอร์น**")
            st.divider()
            st.info("วิทยานิพนธ์: การพัฒนาระบบสนับสนุนการตัดสินใจเพื่อการพยากรณ์ราคาแบบพลวัต")

    # ==========================================================
    # UI Sidebar ใหม่ (แบ่งหมวดหมู่ + มีเส้นคั่น)
    # ==========================================================
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2933/2933116.png", width=80)
        st.markdown(f"### User: {st.session_state['username']}")
        
        # กำหนดค่าเริ่มต้นของหน้า (State) ถ้ายังไม่มี
        if 'current_page' not in st.session_state:
            st.session_state['current_page'] = "📊 แดชบอร์ด"
        
        def set_page(page_name):
            st.session_state['current_page'] = page_name

        st.divider() # เส้นคั่นแรก

        # --- หมวด 1: หน้าแรก ---
        st.caption("หน้าแรก") 
        if st.button("📊 หน้าแดชบอร์ด", use_container_width=True, 
                     type="primary" if st.session_state['current_page'] == "📊 แดชบอร์ด" else "secondary"):
            set_page("📊 แดชบอร์ด")
            st.rerun()

        st.divider() # เส้นคั่น

        # --- หมวด 2: จัดการข้อมูล ---
        st.caption("จัดการข้อมูล") 
        if st.button("📥 จัดการข้อมูล", use_container_width=True,
                     type="primary" if st.session_state['current_page'] == "📥 จัดการข้อมูล" else "secondary"):
            set_page("📥 จัดการข้อมูล")
            st.rerun()
            
        if st.button("🔮 พยากรณ์ราคา", use_container_width=True,
                     type="primary" if st.session_state['current_page'] == "🔮 พยากรณ์ราคา" else "secondary"):
            set_page("🔮 พยากรณ์ราคา")
            st.rerun()

        if st.button("🧠 วิเคราะห์โมเดล", use_container_width=True,
                     type="primary" if st.session_state['current_page'] == "🧠 วิเคราะห์โมเดล" else "secondary"):
            set_page("🧠 วิเคราะห์โมเดล")
            st.rerun()

        st.divider() # เส้นคั่น

        # --- หมวด 3: อื่นๆ ---
        st.caption("อื่น ๆ") 
        if st.button("ℹ️ เกี่ยวกับระบบ", use_container_width=True,
                     type="primary" if st.session_state['current_page'] == "ℹ️ เกี่ยวกับระบบ" else "secondary"):
            set_page("ℹ️ เกี่ยวกับระบบ")
            st.rerun()
            
        st.divider()
        if st.button("Log out", type="secondary"): 
            st.session_state['logged_in'] = False
            st.rerun()

    # ==========================================================
    # ส่วนการเรียกหน้า (Page Routing) - ปรับมาใช้ State
    # ==========================================================
    page = st.session_state['current_page']

    if "แดชบอร์ด" in page: show_dashboard_page()
    elif "จัดการข้อมูล" in page: show_manage_data_page()
    elif "พยากรณ์ราคา" in page: show_pricing_page()
    elif "วิเคราะห์โมเดล" in page: show_model_insight_page()
    elif "เกี่ยวกับระบบ" in page: show_about_page()
