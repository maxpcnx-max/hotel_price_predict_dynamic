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
    page_title="Hotel Price Forecasting System (Smart Date Fixed)",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_FILE = "users.db"
DATA_FILE = "check_in_report.csv"
ROOM_FILE = "room_type.csv" # ไฟล์ Master Data (Mapping ชื่อห้อง)
RES_FILE = "reservation_master.csv" # ไฟล์ Master Data (Mapping ช่องทางการขาย)
METRICS_FILE = "model_metrics.json"
BASE_PRICE_FILE = "base_prices.json" # ไฟล์เก็บราคาฐาน

MODEL_FILES = {
    'xgb': 'xgb_hotel_model.joblib',
    'lr': 'lr_hotel_model.joblib',
    'le_room': 'le_room.joblib',
    'le_res': 'le_res.joblib'
}

DEFAULT_BASE_PRICES = {
    'Standard Room': 1000,
    'Deluxe Room': 1500,
    'Grand Suite Room': 2700
}

DEFAULT_METRICS = {
    'xgb': {'mae': 0, 'r2': 0},
    'lr':  {'mae': 0, 'r2': 0},
    'importance': {}
}

if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'username' not in st.session_state: st.session_state['username'] = ""
if 'historical_avg' not in st.session_state: st.session_state['historical_avg'] = {}
if 'current_page' not in st.session_state: st.session_state['current_page'] = "📊 แดชบอร์ด"

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

def get_base_price(room_text):
    if not isinstance(room_text, str): return 0
    prices = load_base_prices()
    # Exact match first
    if room_text in prices: return prices[room_text]
    # Partial match
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
# 4. DATA HANDLING (Strict Clean & Master Data)
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
    return pd.to_datetime(date_series.apply(convert_dt), errors='coerce')

def normalize_room_id(val):
    try:
        val_float = float(val)
        if val_float.is_integer():
            return str(int(val_float))
        return str(val_float)
    except:
        return str(val).strip()

@st.cache_data
def load_data():
    """โหลดข้อมูลโดยยึด Master Data เป็นหลัก (กำจัดข้อมูลขยะ & ห้องผี)"""
    # 1. Load Transactions
    if not os.path.exists(DATA_FILE):
        try: gdown.download("https://drive.google.com/uc?id=1dxgKIvSTelLaJvAtBSCMCU5K4FuJvfri", DATA_FILE, quiet=True)
        except: return pd.DataFrame()

    try:
        df = pd.read_csv(DATA_FILE)
        
        # Clean Dates
        if 'Date' in df.columns:
            df['Date'] = parse_dates_smart(df['Date'])
            df['is_weekend'] = df['Date'].dt.weekday.isin([5, 6]).astype(int)
            df['Year'] = df['Date'].dt.year
            df['month'] = df['Date'].dt.month
            
        # Clean Room IDs (แก้ 1.0 เป็น 1)
        if 'Room' in df.columns:
            df['Room'] = df['Room'].apply(normalize_room_id)

        # -------------------------------------------------------
        # STEP 2: Strict Room Mapping (หัวใจสำคัญของการกรอง)
        # -------------------------------------------------------
        valid_rooms_list = []
        if os.path.exists(ROOM_FILE):
            try:
                room_master = pd.read_csv(ROOM_FILE)
                if 'Room' in room_master.columns:
                    room_master['Room'] = room_master['Room'].apply(normalize_room_id)
                
                # เตรียม Master Table
                cols_to_use = ['Room']
                target_col = 'Room_Type' if 'Room_Type' in room_master.columns else 'Target_Room_Type'
                
                if target_col in room_master.columns:
                    cols_to_use.append(target_col)
                    room_master = room_master[cols_to_use]
                    
                    # Merge ข้อมูล
                    df = df.merge(room_master, on='Room', how='left')
                    
                    # เปลี่ยนชื่อ Column ให้เป็นมาตรฐาน
                    if target_col != 'Target_Room_Type':
                        df = df.rename(columns={target_col: 'Target_Room_Type'})
                    
                    # เก็บรายชื่อห้องที่ถูกต้องไว้ตรวจสอบ (Strict Filter Logic)
                    valid_rooms_list = room_master[target_col].unique().tolist()
            except: pass

        # ถ้า Merge แล้วไม่มีชื่อห้อง ให้ใส่ Unknown ไปก่อน
        if 'Target_Room_Type' not in df.columns:
            df['Target_Room_Type'] = 'Unknown'
        else:
            df['Target_Room_Type'] = df['Target_Room_Type'].fillna('Unknown')

        # *** STRICT FILTER ***: ตัดทิ้งแถวที่ชื่อห้องเป็น Unknown หรือไม่อยู่ใน Master File
        # นี่คือบรรทัดที่กำจัดห้องเลข 4 หรือ NaN ออกไปครับ
        if valid_rooms_list:
            df = df[df['Target_Room_Type'].isin(valid_rooms_list)]
        
        # -------------------------------------------------------
        # STEP 3: Channel Mapping (จัดการช่องทางการขาย)
        # -------------------------------------------------------
        if 'Reservation' not in df.columns: df['Reservation'] = 'Unknown'
        df['Reservation'] = df['Reservation'].astype(str)

        if os.path.exists(RES_FILE):
            try:
                res_master = pd.read_csv(RES_FILE)
                # สร้าง Dictionary Mapping จากไฟล์ Master
                res_map = {}
                # รองรับชื่อ column หลากหลาย
                col_id = 'Reservation_ID' if 'Reservation_ID' in res_master.columns else 'Reservation'
                col_name = 'Reservation_Name' if 'Reservation_Name' in res_master.columns else 'Reservation_Type'
                
                if col_id in res_master.columns and col_name in res_master.columns:
                     for _, row in res_master.iterrows():
                         res_map[str(row[col_id])] = str(row[col_name])
                
                # Map ค่าใหม่ลงไป (เช่น Agoda.com -> Agoda)
                if res_map:
                    df['Reservation'] = df['Reservation'].map(res_map).fillna(df['Reservation'])
            except: pass
            
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_system_models():
    # Load models gracefully (return None if missing)
    xgb = joblib.load(MODEL_FILES['xgb']) if os.path.exists(MODEL_FILES['xgb']) else None
    lr = joblib.load(MODEL_FILES['lr']) if os.path.exists(MODEL_FILES['lr']) else None
    le_room = joblib.load(MODEL_FILES['le_room']) if os.path.exists(MODEL_FILES['le_room']) else None
    le_res = joblib.load(MODEL_FILES['le_res']) if os.path.exists(MODEL_FILES['le_res']) else None
    
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, 'r') as f: metrics = json.load(f)
    else: metrics = DEFAULT_METRICS
        
    return xgb, lr, le_room, le_res, metrics

# ==========================================================
# 5. RETRAIN SYSTEM
# ==========================================================
def retrain_system():
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        status_text.text("⏳ Reading data...")
        df = load_data()
        
        if df.empty:
            st.error("ไม่พบข้อมูลสำหรับเทรนโมเดล")
            return False, 0
            
        status_text.text("🧹 Preparing data...")
        df_clean = df.dropna(subset=['Price', 'Night', 'Date'])
        
        # เติมค่าว่าง
        df_clean['Night'] = df_clean['Night'].fillna(1)
        df_clean['Adults'] = df_clean['Adults'].fillna(2)
        df_clean['Children'] = df_clean['Children'].fillna(0)
        df_clean['Infants'] = df_clean['Infants'].fillna(0)
        df_clean['Extra Person'] = df_clean['Extra Person'].fillna(0)
        
        # Holiday & Weekend
        if not os.path.exists("thai_holidays.csv"):
             try: gdown.download("https://drive.google.com/uc?id=1L-pciKEeRce1gzuhdtpIGcLs0fYHnbZw", "thai_holidays.csv", quiet=True)
             except: pass
        
        df_clean['is_holiday'] = 0
        if os.path.exists("thai_holidays.csv"):
            holidays_csv = pd.read_csv("thai_holidays.csv")
            holidays_csv['Holiday_Date'] = parse_dates_smart(holidays_csv['Holiday_Date'])
            df_clean['is_holiday'] = df_clean['Date'].isin(holidays_csv['Holiday_Date']).astype(int)

        df_clean['is_weekend'] = df_clean['Date'].dt.weekday.isin([5, 6]).astype(int)
        df_clean['total_guests'] = df_clean[['Adults', 'Children', 'Infants', 'Extra Person']].sum(axis=1)
        df_clean['month'] = df_clean['Date'].dt.month
        df_clean['weekday'] = df_clean['Date'].dt.weekday
        
        # Encoders
        le_room_new = LabelEncoder()
        df_clean['RoomType_encoded'] = le_room_new.fit_transform(df_clean['Target_Room_Type'].astype(str))
        le_res_new = LabelEncoder()
        df_clean['Reservation_encoded'] = le_res_new.fit_transform(df_clean['Reservation'].astype(str))
        
        feature_cols = ['Night', 'total_guests', 'is_holiday', 'is_weekend', 'month', 'weekday', 'RoomType_encoded', 'Reservation_encoded']
        X = df_clean[feature_cols].fillna(0)
        y = df_clean['Price']
        
        progress_bar.progress(40)
        status_text.text("🏋️‍♂️ Training Models...")
        
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
        status_text.text("💾 Saving models...")
        
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
        status_text.success(f"✅ Retraining Complete! R²: {new_xgb_r2:.4f}")
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
        if df_raw.empty: st.warning("No Valid Data Found (ตรวจสอบ Master Data)"); return

        with st.expander("🔎 Filter Data", expanded=True):
            f_col1, f_col2 = st.columns(2)
            valid_years = df_raw['Year'].dropna().unique()
            year_opts = ['All'] + [str(int(y)) for y in sorted(valid_years.tolist())]
            with f_col1: sel_year = st.selectbox("📅 Select Year", year_opts)
            
            valid_months = df_raw['month'].dropna().unique()
            month_opts = ['All'] + [datetime(2024, int(m), 1).strftime('%B') for m in sorted(valid_months.tolist())]
            with f_col2: sel_month_str = st.selectbox("🗓️ Select Month", month_opts)

            df_filtered = df_raw.copy()
            if sel_year != 'All': df_filtered = df_filtered[df_filtered['Year'] == int(sel_year)]
            if sel_month_str != 'All':
                sel_month_num = datetime.strptime(sel_month_str, "%B").month
                df_filtered = df_filtered[df_filtered['month'] == sel_month_num]

        if df_filtered.empty: st.warning("⚠️ No data available."); return

        st.divider()
        k1, k2, k3 = st.columns(3)
        with k1: st.metric("💰 Total Revenue", f"{df_filtered['Price'].sum()/1e6:.2f} M THB")
        with k2: st.metric("📦 Total Bookings", f"{len(df_filtered):,} รายการ")
        with k3: st.metric("🏷️ Avg. Booking Value", f"{df_filtered['Price'].mean():,.0f} THB")
        
        st.divider()
        tab1, tab2, tab3 = st.tabs(["💰 Financial", "📢 Channels", "🛌 Product"])
        group_col = 'Target_Room_Type' 

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Revenue by Room Type")
                room_perf = df_filtered.groupby(group_col).agg({'Price': 'sum'}).reset_index().sort_values('Price', ascending=False)
                st.plotly_chart(px.bar(room_perf, x=group_col, y='Price', color=group_col), use_container_width=True)
            with c2:
                st.subheader("Monthly Revenue")
                monthly = df_filtered.groupby('month')['Price'].sum().reset_index()
                monthly['M_Name'] = monthly['month'].apply(lambda x: datetime(2024, int(x), 1).strftime('%b'))
                st.plotly_chart(px.line(monthly, x='M_Name', y='Price', markers=True), use_container_width=True)

        with tab2:
            c3, c4 = st.columns(2)
            with c3:
                st.subheader("Revenue by Channel")
                res_rev = df_filtered.groupby('Reservation')['Price'].sum().reset_index()
                st.plotly_chart(px.pie(res_rev, values='Price', names='Reservation', hole=0.4), use_container_width=True)
            with c4:
                st.subheader("Bookings by Channel")
                res_cnt = df_filtered['Reservation'].value_counts().reset_index()
                res_cnt.columns = ['Reservation', 'Count']
                st.plotly_chart(px.bar(res_cnt, x='Reservation', y='Count', color='Reservation'), use_container_width=True)

        with tab3:
            st.subheader("Revenue Heatmap (Room vs Channel)")
            heatmap = df_filtered.pivot_table(index=group_col, columns='Reservation', values='Price', aggfunc='sum', fill_value=0)
            st.plotly_chart(px.imshow(heatmap, text_auto=".2s", aspect="auto", color_continuous_scale='Blues'), use_container_width=True)

    def show_manage_data_page():
        st.title("📥 ระบบจัดการฐานข้อมูล (Master Data)")
        
        tab_trans, tab_master, tab_train = st.tabs(["📝 ข้อมูลการจอง (Transactions)", "⚙️ ข้อมูลหลัก (Master Data)", "🚀 อัปเดตโมเดล"])

        with tab_trans:
            st.subheader("1. นำเข้าข้อมูลใหม่")
            up_file = st.file_uploader("เลือกไฟล์ Booking CSV", type=['csv'])
            if up_file is not None:
                if st.button("💾 บันทึกข้อมูล"):
                    try:
                        new_data = pd.read_csv(up_file)
                        if save_data_robust(new_data, mode='append'):
                            st.success("✅ บันทึกเรียบร้อย!"); time.sleep(1); st.rerun()
                    except Exception as e: st.error(f"Error: {e}")

            st.divider()
            st.subheader("2. ดูข้อมูลปัจจุบัน (100 รายการล่าสุด)")
            df_curr = load_data()
            st.dataframe(df_curr.tail(100), use_container_width=True)
            
            with st.expander("🧨 Danger Zone"):
                if st.button("Clear All Data"):
                    if os.path.exists(DATA_FILE): os.remove(DATA_FILE); st.rerun()

        with tab_master:
            # --- ROOM MANAGEMENT ---
            st.container(border=True)
            st.markdown("#### 1. จัดการห้องพัก (Room Master)")
            st.caption("ห้องที่ไม่อยู่ในรายการนี้ จะไม่แสดงใน Dashboard")
            if os.path.exists(ROOM_FILE):
                df_room = pd.read_csv(ROOM_FILE)
            else:
                df_room = pd.DataFrame(columns=['Room', 'Room_Type'])
                
            df_room = df_room.astype(str)

            edited_room = st.data_editor(
                df_room,
                num_rows="dynamic",
                use_container_width=True,
                key="room_editor",
                column_config={
                    "Room": st.column_config.TextColumn("Room ID (เลขห้อง)", required=True),
                    "Room_Type": st.column_config.TextColumn("Room Name (ชื่อห้อง)", required=True)
                }
            )

            if st.button("💾 บันทึกห้องพัก"):
                edited_room.to_csv(ROOM_FILE, index=False)
                st.cache_data.clear() # เคลียร์ Cache เพื่อให้ Dashboard อัปเดตทันที
                st.success("✅ บันทึกห้องพักเรียบร้อย!")
                time.sleep(0.5); st.rerun()

            # --- PRICE MANAGEMENT ---
            st.divider()
            st.markdown("#### 2. จัดการราคาฐาน (Base Price)")
            prices = load_base_prices()
            df_prices = pd.DataFrame(list(prices.items()), columns=['Room_Type', 'Price'])
            
            edited_prices = st.data_editor(
                df_prices,
                num_rows="dynamic",
                use_container_width=True,
                key="price_editor",
                column_config={
                    "Room_Type": st.column_config.TextColumn("Room Name"),
                    "Price": st.column_config.NumberColumn("Base Price (THB)")
                }
            )
            
            if st.button("💾 บันทึกราคา"):
                new_p = {row['Room_Type']: row['Price'] for _, row in edited_prices.iterrows() if row['Room_Type']}
                save_base_prices(new_p)
                st.success("✅ บันทึกราคาเรียบร้อย!")

            # --- CHANNEL MANAGEMENT ---
            st.divider()
            st.markdown("#### 3. จัดการช่องทางการขาย (Channel Master)")
            st.caption("เพิ่มช่องทางใหม่ได้ที่นี่ (เช่น TikTok) แล้วจะไปโผล่ที่หน้าพยากรณ์")
            
            if os.path.exists(RES_FILE):
                df_res = pd.read_csv(RES_FILE)
            else:
                df_res = pd.DataFrame(columns=['Reservation_ID', 'Reservation_Name'])

            df_res = df_res.astype(str)

            edited_res = st.data_editor(
                df_res,
                num_rows="dynamic",
                use_container_width=True,
                key="res_editor",
                column_config={
                    "Reservation_ID": st.column_config.TextColumn("Original ID (ชื่อในไฟล์/รหัส)", required=True),
                    "Reservation_Name": st.column_config.TextColumn("Display Name (ชื่อที่แสดง)", required=True)
                }
            )
            
            if st.button("💾 บันทึกช่องทาง"):
                edited_res.to_csv(RES_FILE, index=False)
                st.cache_data.clear()
                st.success("✅ บันทึกช่องทางเรียบร้อย!")
                time.sleep(0.5); st.rerun()

        with tab_train:
            st.subheader("🚀 Retrain Model")
            st.info("กดเมื่อมีการเพิ่มข้อมูลห้องหรือช่องทางใหม่ เพื่อให้ AI เรียนรู้")
            if st.button("Start Training", type="primary"):
                success, count = retrain_system()
                if success: st.success("✅ เรียบร้อย!"); time.sleep(1); st.rerun()

    def show_pricing_page():
        st.title("🔮 ระบบพยากรณ์ราคา (Price Forecasting)")

        # ----------------------------------------------------
        # DYNAMIC DROPDOWNS (ดึงจาก Master Data โดยตรง)
        # ----------------------------------------------------
        # 1. Room List: ดึงจาก Base Price 
        base_prices = load_base_prices()
        room_options = list(base_prices.keys())
        
        # 2. Channel List: ดึงจาก Channel Master File ที่คุณเพิ่งแก้
        channel_options = []
        if os.path.exists(RES_FILE):
            try:
                res_df = pd.read_csv(RES_FILE)
                # พยายามหา column ชื่อ
                col_name = 'Reservation_Name' if 'Reservation_Name' in res_df.columns else 'Reservation_Type'
                if col_name in res_df.columns:
                    channel_options = res_df[col_name].dropna().unique().tolist()
            except: pass
        
        # Fallback: ถ้ายังไม่มีไฟล์ Master ให้ดึงจากข้อมูลดิบ
        if not channel_options:
            if not df_raw.empty:
                channel_options = df_raw['Reservation'].unique().tolist()
            else:
                channel_options = ["Walk-in", "Agoda", "Booking.com"] # ค่าเริ่มต้น

        with st.container(border=True):
            st.subheader("🛠️ กำหนดเงื่อนไข")
            c1, c2, c3 = st.columns(3)
            
            with c1:
                sel_room = st.selectbox("เลือกห้องพัก", room_options)
            with c2:
                sel_res = st.selectbox("เลือกช่องทาง", channel_options)
            with c3:
                guests = st.number_input("จำนวนแขก", 1, 4, 2)

            date_range = st.date_input("เลือกวันที่เข้าพัก", [])
            
            if st.button("🚀 คำนวณราคา", type="primary", use_container_width=True):
                if len(date_range) < 1: st.error("กรุณาเลือกวันที่"); return
                
                checkin = date_range[0]
                nights = (date_range[1] - checkin).days if len(date_range) > 1 else 1
                if nights < 1: nights = 1

                # --- PREDICTION LOGIC WITH FALLBACK ---
                base_p = get_base_price(sel_room)
                total_base = base_p * nights
                
                predicted_price = total_base # Default start

                if xgb_model and le_room and le_res:
                    try:
                        # ลองแปลงชื่อห้อง/ช่องทาง เป็นรหัสที่ AI รู้จัก
                        r_code = le_room.transform([sel_room])[0]
                        res_code = le_res.transform([sel_res])[0]
                        
                        # สร้างตัวแปรส่งให้ AI
                        is_we = 1 if checkin.weekday() in [5,6] else 0
                        inp = pd.DataFrame([{
                            'Night': nights, 'total_guests': guests, 
                            'is_holiday': 0, 'is_weekend': is_we,
                            'month': checkin.month, 'weekday': checkin.weekday(),
                            'RoomType_encoded': r_code, 'Reservation_encoded': res_code
                        }])
                        
                        raw_pred = xgb_model.predict(inp)[0]
                        
                        # Apply Logic: ราคาต้องไม่ต่ำกว่าราคาฐาน
                        predicted_price = max(raw_pred, total_base)
                        
                        st.success("✅ คำนวณด้วย AI เรียบร้อย")
                    except Exception as e:
                        # กรณีเจอช่องทางใหม่ที่ AI ไม่รู้จัก -> ใช้ราคาฐาน + 20% (ถ้าเป็นวันหยุด)
                        st.warning(f"⚠️ ข้อมูลใหม่ AI ยังไม่เคยเรียนรู้ (ใช้ราคามาตรฐานคำนวณแทน)")
                        multiplier = 1.0
                        if checkin.weekday() in [5,6]: multiplier += 0.2
                        predicted_price = total_base * multiplier
                else:
                    st.error("❌ ไม่พบ Model (กรุณากด Retrain)")

                st.metric("ราคาแนะนำ (Total)", f"{predicted_price:,.0f} THB", f"Base: {total_base:,.0f}")


    def show_model_insight_page():
        st.title("🧠 Model Insights")
        if not metrics['importance']: st.warning("No Data"); return
        
        imp = pd.DataFrame(list(metrics['importance'].items()), columns=['Feature', 'Score'])
        st.plotly_chart(px.bar(imp, x='Score', y='Feature', orientation='h'), use_container_width=True)

    def show_about_page():
        st.title("ℹ️ About")
        st.info("System Version: 3.0 (Strict Master Data Integration)")

    # ==========================================================
    # 7. ROUTING
    # ==========================================================
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2933/2933116.png", width=80)
        st.markdown(f"### User: {st.session_state['username']}")
        
        def set_page(p): st.session_state['current_page'] = p
        
        st.divider()
        st.caption("Main Menu")
        if st.button("📊 แดชบอร์ด", use_container_width=True): set_page("📊 แดชบอร์ด"); st.rerun()
        
        st.divider()
        st.caption("Management")
        if st.button("📥 จัดการข้อมูล", use_container_width=True): set_page("📥 จัดการข้อมูล"); st.rerun()
        if st.button("🔮 พยากรณ์ราคา", use_container_width=True): set_page("🔮 พยากรณ์ราคา"); st.rerun()
        if st.button("🧠 วิเคราะห์โมเดล", use_container_width=True): set_page("🧠 วิเคราะห์โมเดล"); st.rerun()
        
        st.divider()
        st.caption("Other")
        if st.button("ℹ️ เกี่ยวกับระบบ", use_container_width=True): set_page("ℹ️ เกี่ยวกับระบบ"); st.rerun()
        if st.button("Log out"): st.session_state['logged_in'] = False; st.rerun()

    pg = st.session_state['current_page']
    if "แดชบอร์ด" in pg: show_dashboard_page()
    elif "จัดการข้อมูล" in pg: show_manage_data_page()
    elif "พยากรณ์ราคา" in pg: show_pricing_page()
    elif "วิเคราะห์โมเดล" in pg: show_model_insight_page()
    elif "เกี่ยวกับระบบ" in pg: show_about_page()

