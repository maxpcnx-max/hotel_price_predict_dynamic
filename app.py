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
    page_title="Hotel Price Forecasting System (Final Fix)",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_FILE = "users.db"
DATA_FILE = "check_in_report.csv"
ROOM_FILE = "room_type.csv" 
METRICS_FILE = "model_metrics.json"
BASE_PRICE_FILE = "base_prices.json" 

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

# 🔥 กำหนดโครงสร้างคอลัมน์มาตรฐานของไฟล์ CSV (Schema)
# เพื่อป้องกันคอลัมน์ขยะ (เช่น Year, month, Target_Room_Type) ปนเปื้อนลงไฟล์
RAW_COLS = [
    'Date', 'Room', 'Price', 'Reservation', 'Name', 
    'Night', 'Adults', 'Children', 'Infants', 'Extra Person'
]

DEFAULT_METRICS = {'xgb': {'mae': 0, 'r2': 0}, 'lr': {'mae': 0, 'r2': 0}, 'importance': {}}

if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'username' not in st.session_state: st.session_state['username'] = ""

# ==========================================================
# 2. DATABASE
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
# 3. HELPER FUNCTIONS (แยกส่วน Load, Save, Process ชัดเจน)
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
    for key in prices:
        if key in room_text: return prices[key]
    return 0

@st.cache_data
def load_raw_data_for_edit():
    """
    ✅ ฟังก์ชันนี้: อ่าน CSV ดิบๆ เพื่อเอาไปใส่ใน Editor
    - ห้าม Merge
    - ห้ามคำนวณ Year/Month
    - ห้าม Dropna (เดี๋ยว User แก้เอง)
    """
    if not os.path.exists(DATA_FILE):
        try: gdown.download("https://drive.google.com/uc?id=1dxgKIvSTelLaJvAtBSCMCU5K4FuJvfri", DATA_FILE, quiet=True)
        except: return pd.DataFrame(columns=RAW_COLS)

    try:
        df = pd.read_csv(DATA_FILE)
        
        # กรองเอาเฉพาะคอลัมน์มาตรฐาน (กันไฟล์บวม)
        cols_exist = [c for c in RAW_COLS if c in df.columns]
        df = df[cols_exist]

        # แปลง Date เป็น datetime เพื่อให้ Editor แสดง DatePicker ได้
        # ใช้ errors='coerce' เพื่อให้ค่าผิดกลายเป็น NaT (ว่างๆ) แต่อย่าเพิ่งลบทิ้ง ให้ User เห็นและแก้
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        
        if 'Room' in df.columns:
            df['Room'] = df['Room'].astype(str)
            
        return df
    except Exception as e:
        # st.error(f"Error loading raw data: {e}")
        return pd.DataFrame(columns=RAW_COLS)

def process_data_for_analytics(df_raw):
    """
    ✅ ฟังก์ชันนี้: รับข้อมูลดิบ -> ปรุงรส (Merge, สร้าง Feature)
    - ใช้สำหรับ Dashboard และ Retrain เท่านั้น
    - ไม่กระทบไฟล์จริง
    """
    if df_raw.empty: return df_raw
    df = df_raw.copy()
    
    # 1. จัดการ Date
    if 'Date' in df.columns:
        # สำหรับการคำนวณ เราต้องตัดแถวที่วันที่เสียทิ้ง
        df = df.dropna(subset=['Date'])
        df['is_weekend'] = df['Date'].dt.weekday.isin([5, 6]).astype(int)
        df['Year'] = df['Date'].dt.year.astype(int)
        df['month'] = df['Date'].dt.month
        df['weekday'] = df['Date'].dt.weekday
    
    # 2. Merge Master Data (Room Type)
    if os.path.exists(ROOM_FILE):
        try:
            room_type = pd.read_csv(ROOM_FILE)
            if 'Room' in room_type.columns: room_type['Room'] = room_type['Room'].astype(str)
            
            # Merge
            if 'Target_Room_Type' in room_type.columns:
                df = df.merge(room_type[['Room', 'Target_Room_Type']], on='Room', how='left')
            elif 'Room_Type' in room_type.columns:
                room_type = room_type.rename(columns={'Room_Type': 'Target_Room_Type'})
                df = df.merge(room_type[['Room', 'Target_Room_Type']], on='Room', how='left')
        except: pass
    
    # 3. Fill Missing Target Room
    if 'Target_Room_Type' in df.columns:
        df['Target_Room_Type'] = df['Target_Room_Type'].fillna(df['Room'])
    else:
        if 'Room' in df.columns: df['Target_Room_Type'] = df['Room']
        
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

# ฟังก์ชัน Save แบบ "Dumb Save" (เหมือน Base Price)
def dumb_save_csv(df_to_save):
    try:
        # ตัดคอลัมน์ส่วนเกินออก (พวก Year, Month ที่อาจติดมา) ให้เหลือแค่ RAW_COLS
        final_cols = [c for c in RAW_COLS if c in df_to_save.columns]
        df_clean = df_to_save[final_cols]
        
        # บันทึกทับไฟล์เดิม
        df_clean.to_csv(DATA_FILE, index=False)
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"Save Error: {e}")
        return False

# ฟังก์ชัน Merge แบบ "Dumb Merge" (ต่อตูดไฟล์)
def dumb_merge_csv(uploaded_file):
    try:
        new_df = pd.read_csv(uploaded_file)
        # Clean colums ก่อน
        valid_cols = [c for c in RAW_COLS if c in new_df.columns]
        new_df = new_df[valid_cols]
        
        if os.path.exists(DATA_FILE):
            old_df = pd.read_csv(DATA_FILE)
            merged_df = pd.concat([old_df, new_df], ignore_index=True)
        else:
            merged_df = new_df
            
        return dumb_save_csv(merged_df)
    except Exception as e:
        st.error(f"Merge Error: {e}")
        return False

def retrain_system():
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        status_text.text("⏳ Reading & Preparing Data...")
        # 1. อ่าน Raw -> Process
        df_raw = load_raw_data_for_edit()
        if df_raw.empty: return False, 0
        
        df = process_data_for_analytics(df_raw) # ได้ Year, Month, Target_Room_Type มาแล้ว
        
        # 2. Filter Clean Data for Training
        df = df.dropna(subset=['Price', 'Night', 'Date'])
        
        # ... (Feature Engineering Logic เดิม) ...
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
        
        # Encoders
        le_room_new = LabelEncoder()
        df['Target_Room_Type'] = df['Target_Room_Type'].astype(str)
        df['RoomType_encoded'] = le_room_new.fit_transform(df['Target_Room_Type'])
        
        le_res_new = LabelEncoder()
        df['Reservation'] = df['Reservation'].astype(str)
        df['Reservation_encoded'] = le_res_new.fit_transform(df['Reservation'])
        
        feature_cols = ['Night', 'total_guests', 'is_holiday', 'is_weekend', 'month', 'weekday', 'RoomType_encoded', 'Reservation_encoded']
        X = df[feature_cols].fillna(0)
        y = df['Price']
        
        progress_bar.progress(40)
        status_text.text("🏋️‍♂️ Training XGBoost...")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        xgb_new = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        xgb_new.fit(X_train, y_train)
        pred_xgb = xgb_new.predict(X_test)
        new_xgb_r2 = r2_score(y_test, pred_xgb)
        
        fi_raw = xgb_new.feature_importances_
        col_mapping = {'Night': 'Night', 'total_guests': 'Guests', 'is_holiday': 'Is Holiday', 'is_weekend': 'Is Weekend', 'month': 'Month', 'weekday': 'Weekday', 'RoomType_encoded': 'Room Type', 'Reservation_encoded': 'Reservation'}
        new_importance = {col_mapping.get(col, col): float(val) for col, val in zip(feature_cols, fi_raw)}
        
        lr_new = LinearRegression()
        lr_new.fit(X_train, y_train)
        
        progress_bar.progress(80)
        status_text.text("💾 Saving models...")
        
        joblib.dump(xgb_new, MODEL_FILES['xgb'])
        joblib.dump(lr_new, MODEL_FILES['lr'])
        joblib.dump(le_room_new, MODEL_FILES['le_room'])
        joblib.dump(le_res_new, MODEL_FILES['le_res'])
        
        with open(METRICS_FILE, 'w') as f:
            json.dump({'xgb': {'mae': 0, 'r2': new_xgb_r2}, 'lr': {'mae': 0, 'r2': 0}, 'importance': new_importance}, f)
            
        st.cache_resource.clear()
        progress_bar.progress(100)
        status_text.success(f"✅ Training Done! R²: {new_xgb_r2:.4f}")
        return True, len(df)
    except Exception as e:
        st.error(f"Retrain Error: {e}")
        return False, 0

# ==========================================================
# 4. MAIN UI PAGES
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
    # โหลดโมเดล
    xgb_model, lr_model, le_room, le_res, metrics = load_system_models()
    
    # ----------------------------------------
    # PAGE 1: DASHBOARD
    # ----------------------------------------
    def show_dashboard_page():
        st.title("📊 Financial Executive Dashboard")
        
        # 1. โหลดข้อมูลดิบ
        df_raw = load_raw_data_for_edit()
        if df_raw.empty: st.warning("No Data Found"); return
        
        # 2. ปรุงรสข้อมูล (Process) เพื่อใช้กราฟ
        df_dash = process_data_for_analytics(df_raw)
        
        # 3. กรองข้อมูลแสดงผล (เฉพาะที่มี Date)
        df_filtered = df_dash.dropna(subset=['Date']).copy()

        with st.expander("🔎 Filter Data (ตัวกรองข้อมูล)", expanded=True):
            f_col1, f_col2, f_col3 = st.columns(3)
            years = sorted(df_filtered['Year'].unique().tolist())
            year_opts = ['All'] + [str(int(y)) for y in years]
            with f_col1: sel_year = st.selectbox("📅 Select Year (เลือกปี)", year_opts)
            
            months = sorted(df_filtered['month'].unique().tolist())
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
        st.dataframe(df_filtered)

    # ----------------------------------------
    # PAGE 2: MANAGE DATA (แก้ไขจุดนี้ตามสั่ง)
    # ----------------------------------------
    def show_manage_data_page():
        st.title("📥 จัดการข้อมูล (Raw Data Editor)")
        st.caption("โหมดแก้ไขข้อมูลดิบ: Add, Edit, Delete ได้อิสระ แล้วกด Save")
        
        tab_trans, tab_master, tab_train = st.tabs(["📝 ข้อมูลการจอง (Booking)", "⚙️ ราคาฐาน (Master Data)", "🚀 เทรนโมเดล (Retrain)"])

        # TAB 1: BOOKING EDITOR
        with tab_trans:
            st.markdown("#### 1. นำเข้าไฟล์ (Merge)")
            up_file = st.file_uploader("เลือกไฟล์ CSV ที่จะนำเข้า", type=['csv'])
            if up_file is not None:
                if st.button("➕ Merge File (ต่อท้าย)", type="secondary"):
                    if dumb_merge_csv(up_file):
                        st.success("Merge สำเร็จ! รีโหลดหน้าใหม่...")
                        time.sleep(1); st.rerun()

            st.divider()
            st.markdown("#### 2. แก้ไขตาราง (CRUD)")
            
            # 🔥 1. โหลดข้อมูลดิบ (ไม่ปรุงแต่ง) มาใส่ Editor
            df_current = load_raw_data_for_edit()
            
            # 🔥 2. แสดง Data Editor ให้ User แก้ไขได้เต็มที่
            edited_df = st.data_editor(
                df_current,
                num_rows="dynamic", # อนุญาตให้ Add/Delete Row ได้
                use_container_width=True,
                key="booking_editor_crud",
                column_config={
                    "Date": st.column_config.DateColumn("Check-in Date", format="DD/MM/YYYY"),
                    "Price": st.column_config.NumberColumn("Price (THB)", format="%d"),
                }
            )

            # 🔥 3. ปุ่ม Save แบบ Dumb Save (รับค่าจาก Editor -> เขียนลงไฟล์เลย)
            if st.button("💾 บันทึกข้อมูลทั้งหมด (Save All)", type="primary"):
                try:
                    if dumb_save_csv(edited_df):
                        st.success(f"✅ บันทึกข้อมูลเรียบร้อย ({len(edited_df)} รายการ)")
                        time.sleep(1); st.rerun()
                except Exception as e:
                    st.error(f"Save Error: {e}")

        # TAB 2: MASTER DATA EDITOR (Logic เดิมที่ดีอยู่แล้ว)
        with tab_master:
            st.subheader("⚙️ กำหนดราคาฐาน (Base Prices)")
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
                st.success("✅ อัปเดตราคาฐานเรียบร้อย!")

        # TAB 3: RETRAIN
        with tab_train:
            st.subheader("🧠 สั่งให้โมเดลเรียนรู้ใหม่ (Retrain)")
            st.write(f"Current Accuracy (R²): {metrics['xgb']['r2']:.4f}")
            if st.button("🚀 Start Retraining", type="primary"):
                success, count = retrain_system()
                if success: st.success(f"Done! Trained on {count} rows."); time.sleep(1); st.rerun()

    # ----------------------------------------
    # PAGE 3: PREDICTION
    # ----------------------------------------
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
                res_options = ["All (เลือกทั้งหมด)"] + list(le_res.classes_)
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
                        
                        extra_guests = guests + 1
                        r2c1, r2c2 = st.columns(2)
                        if extra_guests <= max_g:
                            p_xgb_ex = p_xgb + 500
                            p_lr_ex = p_lr + 500
                            with r2c1: st.metric(f"XGBoost (+1 Guest: {extra_guests})", f"{p_xgb_ex:,.0f} THB", "+500")
                            with r2c2: st.metric(f"Linear (+1 Guest: {extra_guests})", f"{p_lr_ex:,.0f} THB", "+500")

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
