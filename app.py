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
    page_title="Hotel Price Forecasting System (Gatekeeper)",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_FILE = "users.db"
DATA_FILE = "check_in_report.csv"
ROOM_FILE = "room_type.csv" 
METRICS_FILE = "model_metrics.json"
BASE_PRICE_FILE = "base_prices.json" 
CHANNELS_FILE = "channels.json" # 🔥 ไฟล์ใหม่สำหรับเก็บรายชื่อช่องทาง

MODEL_FILES = {
    'xgb': 'xgb_hotel_model.joblib',
    'lr': 'lr_hotel_model.joblib',
    'le_room': 'le_room.joblib',
    'le_res': 'le_res.joblib'
}

# --- ค่า Default ---
DEFAULT_BASE_PRICES = {
    'Grand Suite Room': 2700,
    'Villa Suite (Garden)': 2700,
    'Executive Room': 2500,
    'Executive Room with Balcony': 2400,
    'Villa Suite (Bathtub)': 2000,
    'Deluxe Room': 1500,
    'Standard Room': 1000
}

DEFAULT_CHANNELS = ["Agoda", "Booking.com", "Traveloka", "Walk-in", "Direct", "Expedia"]

DEFAULT_METRICS = {'xgb': {'mae': 0, 'r2': 0}, 'lr': {'mae': 0, 'r2': 0}, 'importance': {}}

if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'username' not in st.session_state: st.session_state['username'] = ""

# ==========================================================
# 2. HELPER FUNCTIONS (Load/Save Configs)
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
    # 🔥 โหลดรายชื่อช่องทาง (ถ้าไม่มีไฟล์ ให้สร้างจาก Default)
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
# 4. BACKEND SYSTEM
# ==========================================================

@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE):
        try: gdown.download("https://drive.google.com/uc?id=1dxgKIvSTelLaJvAtBSCMCU5K4FuJvfri", DATA_FILE, quiet=True)
        except: return pd.DataFrame()

    try:
        df = pd.read_csv(DATA_FILE)
        
        if 'Date' in df.columns:
            # เก็บ Raw Date ไว้ (ไม่ dropna ที่นี่ เพื่อให้หน้า Edit เห็นทั้งหมด)
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            
            # คำนวณ Feature เฉพาะแถวที่ดี (เพื่อไม่ให้ error)
            mask = df['Date'].notna()
            df.loc[mask, 'is_weekend'] = df.loc[mask, 'Date'].dt.weekday.isin([5, 6]).astype(int)
            df.loc[mask, 'Year'] = df.loc[mask, 'Date'].dt.year.astype(int)
            df.loc[mask, 'month'] = df.loc[mask, 'Date'].dt.month
            
        if 'Room' in df.columns:
            df['Room'] = df['Room'].astype(str)

        # Mapping (ถ้ามี)
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
        df = df.loc[:, ~df.columns.duplicated()]
        return df
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()

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

# 🔥 VALIDATION LOGIC: หัวใจสำคัญของ Gatekeeper
def validate_and_save_data(df_to_check, save_to_file=True):
    """
    ตรวจสอบข้อมูลก่อนบันทึก:
    1. ต้องเป็นห้องที่มีใน Base Price (หรือ Room File)
    2. ต้องเป็น Channel ที่มีในระบบ
    3. วันที่ต้องไม่เป็น NaT
    """
    # 1. เตรียมรายการที่ถูกต้อง (Whitelist)
    valid_rooms = set(load_base_prices().keys()) # เอามาจาก Base Price เลย ง่ายดี
    valid_channels = set(load_channels())
    
    # เพิ่มห้องจาก room_type.csv ด้วย (เผื่อมีชื่อเดิมที่ยังไม่ได้ map)
    if os.path.exists(ROOM_FILE):
        try:
            rt = pd.read_csv(ROOM_FILE)
            if 'Room' in rt.columns: valid_rooms.update(rt['Room'].astype(str))
        except: pass

    # 2. ตรวจสอบข้อมูล (Cleaning)
    df_clean = df_to_check.copy()
    
    if 'Date' in df_clean.columns:
        df_clean['Date'] = pd.to_datetime(df_clean['Date'], dayfirst=True, errors='coerce')
    
    if 'Room' in df_clean.columns:
        df_clean['Room'] = df_clean['Room'].astype(str)
        
    if 'Reservation' in df_clean.columns:
        df_clean['Reservation'] = df_clean['Reservation'].astype(str)

    # 3. แยกน้ำแยกเนื้อ
    # กฎ 1: วันที่ต้องสมบูรณ์
    mask_date = df_clean['Date'].notna()
    # กฎ 2: ห้องต้องรู้จัก (อยู่ใน Base Price หรือ Master File)
    mask_room = df_clean['Room'].isin(valid_rooms)
    # กฎ 3: Channel ต้องรู้จัก (ถ้าจะเคร่ง) - หรือจะปล่อยก็ได้ แต่ user ขอให้เพิ่ม tab channel ก็น่าจะอยากคุม
    # เพื่อความยืดหยุ่น ถ้า Channel ไม่มี เราจะ Warn แต่ไม่ตัดทิ้งดีไหม? 
    # เอาแบบเคร่งตามโจทย์ "ระบบยอมรับไหม" -> ถ้ามี tab ให้เพิ่ม แสดงว่าต้องตรงกัน
    mask_channel = df_clean['Reservation'].isin(valid_channels)

    # รวมกฎ (ตัดเฉพาะที่แย่จริงๆ คือ วันที่พัง หรือ ห้องมั่ว)
    # Channel มั่ว อาจจะแค่เตือนแต่เก็บไว้ หรือตัดเลย? 
    # ตัดเลยตาม Concept "Gatekeeper"
    mask_valid = mask_date & mask_room & mask_channel
    
    df_good = df_clean[mask_valid].copy()
    df_bad = df_clean[~mask_valid].copy()
    
    # 4. บันทึก (เฉพาะของดี)
    if save_to_file and not df_good.empty:
        # ตัดคอลัมน์ขยะทิ้ง
        save_cols = ['Date', 'Room', 'Price', 'Reservation', 'Name', 'Night', 'Adults', 'Children', 'Infants', 'Extra Person']
        final_cols = [c for c in save_cols if c in df_good.columns]
        
        # ถ้าไฟล์มีอยู่แล้ว ให้เขียนทับ หรือ ต่อท้าย?
        # ถ้าเรียกจากปุ่ม Save (Edit Table) คือเขียนทับ (Full Replace)
        # ถ้าเรียกจาก Upload คือต่อท้าย (Append)
        # ปรับ logic ให้รับ mode ได้ แต่ในที่นี้ขอเขียนทับถาวรเฉพาะ Save Button
        
        df_good[final_cols].to_csv(DATA_FILE, index=False)
        st.cache_data.clear()
        
    return df_good, df_bad

def retrain_system():
    status_text = st.empty()
    progress_bar = st.progress(0)
    try:
        status_text.text("⏳ Reading & Cleaning data...")
        df = load_data() 
        if df.empty: return False, 0
        df = df.dropna(subset=['Price', 'Night', 'Date'])
        
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
    df_raw = load_data() 
    xgb_model, lr_model, le_room, le_res, metrics = load_system_models()
    
    def show_dashboard_page():
        st.title("📊 Financial Executive Dashboard")
        if df_raw.empty: st.warning("No Data Found"); return

        df_filtered = df_raw.dropna(subset=['Date']).copy()

        with st.expander("🔎 Filter Data (ตัวกรองข้อมูล)", expanded=True):
            f_col1, f_col2, f_col3 = st.columns(3)
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
                room_perf = df_filtered.groupby(group_col).agg({'Price': 'sum', 'Night': 'sum'}).reset_index().sort_values('Price', ascending=False)
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Bar(x=room_perf[group_col], y=room_perf['Price'], name="Revenue", marker_color='#1f77b4'), secondary_y=False)
                fig.add_trace(go.Scatter(x=room_perf[group_col], y=room_perf['Night'], name="Nights", mode='lines+markers', marker_color='#ff7f0e'), secondary_y=True)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                monthly = df_filtered.groupby('month').agg({'Price': 'sum', 'Room': 'count'}).reset_index().sort_values('month')
                monthly['M_Name'] = monthly['month'].astype(int).apply(lambda x: datetime(2024, x, 1).strftime('%b'))
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
        st.subheader("📋 Raw Data Explorer")
        with st.expander("คลิกเพื่อดูตารางข้อมูลที่ผ่านการกรองแล้ว"): st.dataframe(df_filtered)

    def show_manage_data_page():
        st.title("📥 ระบบจัดการฐานข้อมูล (Data Management)")
        
        # เพิ่ม Tabs ใหม่ตามสั่ง
        tab_trans, tab_master, tab_channel, tab_train = st.tabs([
            "📝 ข้อมูลการจอง (Transactions)", 
            "⚙️ ราคาฐาน (Base Price)", 
            "📢 ช่องทางการจอง (Channels)", 
            "🚀 อัปเดตโมเดล (Retrain)"
        ])

        # ----------------------------------------
        # TAB 1: Transactions (Gatekeeper Save)
        # ----------------------------------------
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
                        
                        # 🔥 ผ่าน Gatekeeper
                        good_df, bad_df = validate_and_save_data(merged_df, save_to_file=True)
                        
                        if not bad_df.empty:
                            st.warning(f"⚠️ พบข้อมูลที่ไม่ถูกต้อง {len(bad_df)} รายการ (ชื่อห้อง/ช่องทางไม่ตรง หรือวันที่ผิด) ระบบข้ามข้อมูลเหล่านี้ไป")
                            with st.expander("ดูรายการที่ถูกตัดออก"): st.dataframe(bad_df)
                        
                        st.success(f"✅ บันทึกข้อมูลที่ถูกต้อง {len(good_df)} รายการ เรียบร้อย!")
                        time.sleep(1); st.rerun()
                    except Exception as e: st.error(f"Error: {e}")

            st.divider()
            st.subheader("2. แก้ไขข้อมูล (Gatekeeper Mode)")
            st.info("💡 ข้อมูลที่คุณแก้ไขจะถูกตรวจสอบความถูกต้องก่อนบันทึก (ห้องและช่องทางต้องมีในระบบ)")
            
            # โหลด Raw Data เพื่อแสดงใน Editor
            df_current = load_data()
            
            # แปลง Date ใน memory เพื่อให้ Editor สวย
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

            # ปุ่ม Save พร้อมระบบกรอง (Gatekeeper)
            if st.button("💾 บันทึกการเปลี่ยนแปลง (Save & Filter)", type="primary"):
                try:
                    # 🔥 เรียกใช้ฟังก์ชันตรวจสอบ
                    good_df, bad_df = validate_and_save_data(edited_df, save_to_file=True)
                    
                    if not bad_df.empty:
                        st.error(f"🚫 ระบบลบข้อมูลที่ไม่ถูกต้องออก {len(bad_df)} รายการ (ตรวจสอบชื่อห้อง/ช่องทาง)")
                        with st.expander("ดูรายการที่ถูกลบ"): st.dataframe(bad_df)
                    
                    st.success(f"✅ บันทึกข้อมูลที่ถูกต้อง {len(good_df)} รายการ สำเร็จ!")
                    time.sleep(1); st.rerun()
                    
                except Exception as e:
                    st.error(f"Save Error: {e}")
            
            with st.columns([1, 4])[0]:
                if st.button("🧨 ล้างข้อมูลทั้งหมด"):
                     if os.path.exists(DATA_FILE):
                        os.remove(DATA_FILE)
                        st.cache_data.clear()
                        st.rerun()

        # ----------------------------------------
        # TAB 2: Base Price (Source of Truth for Rooms)
        # ----------------------------------------
        with tab_master:
            st.subheader("⚙️ กำหนดราคาฐาน (รายชื่อห้องที่ระบบยอมรับ)")
            st.caption("ℹ️ ชื่อห้องที่ปรากฏในตารางนี้ คือรายชื่อห้องที่ระบบจะยอมรับในหน้า 'ข้อมูลการจอง' (เพิ่มห้องใหม่ที่นี่ได้เลย)")
            
            current_prices = load_base_prices()
            df_prices = pd.DataFrame(list(current_prices.items()), columns=['Room Type', 'Base Price'])
            
            edited_prices_df = st.data_editor(
                df_prices,
                num_rows="dynamic",
                use_container_width=True,
                key="price_editor_tab",
                column_config={"Base Price": st.column_config.NumberColumn("Price", format="%d THB")}
            )
            
            if st.button("💾 บันทึกราคาฐาน"):
                new_prices_dict = {}
                for index, row in edited_prices_df.iterrows():
                    if row['Room Type'] and str(row['Room Type']).strip() != "":
                        new_prices_dict[row['Room Type']] = row['Base Price']
                save_base_prices(new_prices_dict)
                st.success("✅ อัปเดตราคาฐาน และ รายชื่อห้อง เรียบร้อย!")

        # ----------------------------------------
        # TAB 3: Channels (Source of Truth for Reservation)
        # ----------------------------------------
        with tab_channel:
            st.subheader("📢 จัดการช่องทางการจอง (Channels)")
            st.caption("ℹ️ รายชื่อช่องทางที่ระบบยอมรับ (เช่น Agoda, Booking.com)")
            
            current_channels = load_channels()
            df_channels = pd.DataFrame(current_channels, columns=['Channel Name'])
            
            edited_channels_df = st.data_editor(
                df_channels,
                num_rows="dynamic",
                use_container_width=True,
                key="channel_editor_tab"
            )
            
            if st.button("💾 บันทึกช่องทาง"):
                new_channel_list = []
                for index, row in edited_channels_df.iterrows():
                    if row['Channel Name'] and str(row['Channel Name']).strip() != "":
                        new_channel_list.append(row['Channel Name'])
                save_channels(new_channel_list)
                st.success("✅ อัปเดตรายชื่อช่องทางเรียบร้อย!")

        # ----------------------------------------
        # TAB 4: Retrain
        # ----------------------------------------
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
                # โหลดช่องทางจากไฟล์ Channels
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
