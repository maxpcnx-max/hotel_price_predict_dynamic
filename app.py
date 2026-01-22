import streamlit as st
import pandas as pd
import joblib
import sqlite3
import os
import json  # <--- [NEW] ใช้เก็บค่าคะแนน MAE/R2 ล่าสุด
import holidays
import plotly.express as px
import gdown
from datetime import datetime

# [NEW] Import Library สำหรับการ Retrain โมเดลในตัว
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

# ชื่อไฟล์ต่างๆ
DB_FILE = "users.db"
DATA_FILE = "check_in_report.csv"
METRICS_FILE = "model_metrics.json" # <--- [NEW] ไฟล์เก็บคะแนนล่าสุด

MODEL_FILES = {
    'xgb': 'xgb_hotel_model.joblib',
    'lr': 'lr_hotel_model.joblib',
    'le_room': 'le_room.joblib',
    'le_res': 'le_res.joblib'
}

# [NEW] ค่า Default (Thesis Baseline) ใช้กรณีเริ่มระบบครั้งแรก
DEFAULT_METRICS = {
    'xgb': {'mae': 1112.79, 'r2': 0.7256}, # ค่าจากเล่ม Thesis
    'lr':  {'mae': 1162.27, 'r2': 0.7608}
}

# Check Login State
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ""

# ==========================================================
# 2. DATABASE (SQLite)
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

def register_user(username, password):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('INSERT INTO users VALUES (?,?)', (username, password))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

init_db()

# ==========================================================
# 3. BACKEND SYSTEM (Dynamic Engine)
# ==========================================================

# --- A. Data Loader ---
@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE):
        # Fallback: Download default data
        url_main = "https://drive.google.com/uc?id=1dxgKIvSTelLaJvAtBSCMCU5K4FuJvfri"
        try: gdown.download(url_main, DATA_FILE, quiet=True)
        except: return pd.DataFrame()

    try:
        df = pd.read_csv(DATA_FILE)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        
        # Merge Room Type if exists
        if os.path.exists("room_type.csv"):
            room_type = pd.read_csv("room_type.csv")
            if 'Room_Type' in room_type.columns:
                 room_type = room_type.rename(columns={'Room_Type': 'Target_Room_Type'})
            df = df.merge(room_type, on='Room', how='left')
            df['Target_Room_Type'] = df['Target_Room_Type'].fillna('Standard Room')
        else:
            df['Target_Room_Type'] = df.get('Room', 'Unknown')

        df['Reservation'] = df['Reservation'].fillna('Unknown')
        df['month'] = df['Date'].dt.month
        return df
    except Exception as e:
        return pd.DataFrame()

# --- B. Model Loader (With Dynamic Metrics) ---
@st.cache_resource
def load_system_models():
    # 1. โหลดโมเดล
    for name, file in MODEL_FILES.items():
        if not os.path.exists(file):
            return None, None, None, None, None

    xgb = joblib.load(MODEL_FILES['xgb'])
    lr = joblib.load(MODEL_FILES['lr'])
    le_room = joblib.load(MODEL_FILES['le_room'])
    le_res = joblib.load(MODEL_FILES['le_res'])
    
    # 2. [NEW] โหลดคะแนน (Metrics)
    # ถ้ามีไฟล์ json (จากการ Retrain ล่าสุด) ให้ใช้ค่านั้น
    # ถ้าไม่มี ให้ใช้ค่า Default (Thesis)
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, 'r') as f:
            metrics = json.load(f)
    else:
        metrics = DEFAULT_METRICS
        
    return xgb, lr, le_room, le_res, metrics

# --- C. Save Data (With Seek Fix) ---
def save_uploaded_data(uploaded_file):
    try:
        # [FIX] ต้อง Reset cursor กลับไปจุดเริ่มต้นไฟล์ก่อนอ่าน
        uploaded_file.seek(0) 
        
        new_data = pd.read_csv(uploaded_file)
        
        if os.path.exists(DATA_FILE):
            current_df = pd.read_csv(DATA_FILE)
            updated_df = pd.concat([current_df, new_data], ignore_index=True)
        else:
            updated_df = new_data
            
        updated_df.to_csv(DATA_FILE, index=False)
        st.cache_data.clear() # Clear Cache Data
        return True
    except Exception as e:
        st.error(f"Save failed: {e}")
        return False

# --- D. [NEW] Retrain Function (ฟังก์ชันเทรนโมเดลใหม่สดๆ) ---
def retrain_system():
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        status_text.text("⏳ Reading all data...")
        df = pd.read_csv(DATA_FILE)
        
        # --- Preprocessing (เหมือน Training Pipeline) ---
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        
        # Handle Room Type
        if os.path.exists("room_type.csv"):
            room_type = pd.read_csv("room_type.csv")
            if 'Room_Type' in room_type.columns:
                 room_type = room_type.rename(columns={'Room_Type': 'Target_Room_Type'})
            df = df.merge(room_type, on='Room', how='left')
            df['Target_Room_Type'] = df['Target_Room_Type'].fillna('Standard Room')
        else:
            df['Target_Room_Type'] = df['Room']

        df = df.dropna(subset=['Date'])
        df['Reservation'] = df['Reservation'].fillna('Unknown')
        
        # Download Holidays if missing
        if not os.path.exists("thai_holidays.csv"):
             try: gdown.download("https://drive.google.com/uc?id=1L-pciKEeRce1gzuhdtpIGcLs0fYHnbZw", "thai_holidays.csv", quiet=True)
             except: pass
        
        if os.path.exists("thai_holidays.csv"):
            holidays_csv = pd.read_csv("thai_holidays.csv")
            holidays_csv['Holiday_Date'] = pd.to_datetime(holidays_csv['Holiday_Date'], dayfirst=True, errors='coerce')
            df['is_holiday'] = df['Date'].isin(holidays_csv['Holiday_Date']).astype(int)
        else:
            df['is_holiday'] = 0

        df['is_weekend'] = df['Date'].dt.weekday.isin([5, 6]).astype(int)
        df['total_guests'] = df[['Adults', 'Children', 'Infants', 'Extra Person']].fillna(0).sum(axis=1)
        df['month'] = df['Date'].dt.month
        df['weekday'] = df['Date'].dt.weekday
        
        # Create New Encoders
        le_room_new = LabelEncoder()
        df['RoomType_encoded'] = le_room_new.fit_transform(df['Target_Room_Type'].astype(str))
        
        le_res_new = LabelEncoder()
        df['Reservation_encoded'] = le_res_new.fit_transform(df['Reservation'].astype(str))
        
        # Select Features
        feature_cols = ['Night', 'total_guests', 'is_holiday', 'is_weekend', 'month', 'weekday', 'RoomType_encoded', 'Reservation_encoded']
        X = df[feature_cols]
        y = df['Price']
        
        # --- Training & Evaluation ---
        progress_bar.progress(30)
        status_text.text("🏋️‍♂️ Training new models...")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 1. XGBoost
        xgb_new = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        xgb_new.fit(X_train, y_train)
        pred_xgb = xgb_new.predict(X_test)
        new_xgb_mae = mean_absolute_error(y_test, pred_xgb)
        new_xgb_r2 = r2_score(y_test, pred_xgb)
        
        # 2. Linear Regression
        lr_new = LinearRegression()
        lr_new.fit(X_train, y_train)
        pred_lr = lr_new.predict(X_test)
        new_lr_mae = mean_absolute_error(y_test, pred_lr)
        new_lr_r2 = r2_score(y_test, pred_lr)
        
        progress_bar.progress(80)
        status_text.text("💾 Saving updated intelligence...")
        
        # --- Save Models & Metrics ---
        joblib.dump(xgb_new, MODEL_FILES['xgb'])
        joblib.dump(lr_new, MODEL_FILES['lr'])
        joblib.dump(le_room_new, MODEL_FILES['le_room'])
        joblib.dump(le_res_new, MODEL_FILES['le_res'])
        
        # [NEW] Save Dynamic Metrics to JSON
        new_metrics = {
            'xgb': {'mae': new_xgb_mae, 'r2': new_xgb_r2},
            'lr':  {'mae': new_lr_mae, 'r2': new_lr_r2}
        }
        with open(METRICS_FILE, 'w') as f:
            json.dump(new_metrics, f)
            
        st.cache_resource.clear() # Clear Model Cache
        
        progress_bar.progress(100)
        status_text.success(f"✅ Retraining Complete! New Accuracy (R²): {new_xgb_r2:.4f}")
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
        st.markdown("ระบบการพยากรณ์ราคาห้องพัก (Hotel Price Forecasting System)")
        
        tab_log, tab_reg = st.tabs(["Login", "Register"])
        with tab_log:
            username = st.text_input("Username", placeholder="Username")
            password = st.text_input("Password", type="password", placeholder="Password")
            if st.button("เข้าสู่ระบบ (Login)", type="primary", use_container_width=True):
                if login_user(username, password):
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    st.rerun()
                else:
                    st.error("Invalid Username or Password")
        with tab_reg:
            new_u = st.text_input("New User")
            new_p = st.text_input("New Pass", type="password")
            if st.button("สมัครสมาชิก", use_container_width=True):
                if register_user(new_u, new_p): st.success("Success! Please Login")
                else: st.error("User already exists")
        st.divider()

if not st.session_state['logged_in']:
    login_page()
else:
    # Load Resources
    df = load_data()
    xgb_model, lr_model, le_room, le_res, metrics = load_system_models()
    
    # Download Holidays if missing (Utility)
    if not os.path.exists("thai_holidays.csv"):
        try: gdown.download("https://drive.google.com/uc?id=1L-pciKEeRce1gzuhdtpIGcLs0fYHnbZw", "thai_holidays.csv", quiet=True)
        except: pass

    # --- Page Functions ---

    def show_home_page():
        if os.path.exists("cover.jpg"): st.image("cover.jpg", use_container_width=True)
        else: st.image("https://images.unsplash.com/photo-1566073771259-6a8506099945?q=80", use_container_width=True)
        st.title("ระบบการพยากรณ์ราคาห้องพัก 👋")
        st.subheader("(Hotel Price Forecasting System)")
        st.markdown("""
        ยินดีต้อนรับเข้าสู่ระบบสนับสนุนการตัดสินใจ (Decision Support System) 
        **ความสามารถของระบบ:**
        * **📊 Data Analytics:** วิเคราะห์ข้อมูลการจองย้อนหลัง (Dynamic)
        * **🔮 Price Forecasting:** พยากรณ์ราคาที่เหมาะสม (AI-Powered)
        * **🔄 Adaptive Learning:** ระบบสามารถเรียนรู้ข้อมูลใหม่ได้ (Retrain)
        """)
        st.info("👈 กรุณาเลือกเมนูจากแถบด้านซ้ายเพื่อเริ่มใช้งาน")

    def show_dashboard_page():
        st.title("📊 แดชบอร์ดสรุปข้อมูล (Executive Dashboard)")
        st.markdown("### สรุปภาพรวมผลประกอบการ")
        if df.empty:
            st.warning("ไม่พบข้อมูล กรุณาไปที่เมนู 'จัดการข้อมูล' เพื่อนำเข้าไฟล์")
            return
        st.divider()
        k1, k2, k3, k4 = st.columns(4)
        with k1: st.metric("📦 Booking ทั้งหมด", f"{len(df):,} รายการ")
        with k2: st.metric("💰 รายได้รวม", f"{df['Price'].sum()/1e6:.2f} M THB")
        with k3: st.metric("🏷️ ราคาเฉลี่ย", f"{df['Price'].mean():,.0f} THB")
        with k4: st.metric("🌙 คืนเฉลี่ย", f"{df['Night'].mean():.1f} คืน")
        st.divider()
        
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("**🏆 ยอดจองตามประเภทห้อง**")
            if 'Target_Room_Type' in df.columns:
                rc = df['Target_Room_Type'].value_counts().reset_index()
                rc.columns = ['Room', 'Count']
                st.plotly_chart(px.bar(rc, x='Count', y='Room', orientation='h', text='Count', color='Count', color_continuous_scale='Viridis'), use_container_width=True)
        with c2:
            st.markdown("**💸 สัดส่วนรายได้**")
            if 'Target_Room_Type' in df.columns:
                rev = df.groupby('Target_Room_Type')['Price'].sum().reset_index()
                st.plotly_chart(px.pie(rev, values='Price', names='Target_Room_Type', hole=0.4), use_container_width=True)
        
        st.divider()
        c3, c4 = st.columns([2, 3])
        with c3:
            st.markdown("**🌐 ช่องทางการจอง**")
            res = df['Reservation'].value_counts().reset_index()
            res.columns = ['Channel', 'Count']
            st.plotly_chart(px.pie(res, values='Count', names='Channel', hole=0.4, color_discrete_sequence=px.colors.sequential.Magma), use_container_width=True)
        with c4:
            st.markdown("**📈 แนวโน้มรายเดือน**")
            mt = df.groupby('month')['Price'].sum().reset_index()
            mt['M_Name'] = mt['month'].apply(lambda x: datetime(2024, int(x), 1).strftime('%B'))
            mt = mt.sort_values('month')
            st.plotly_chart(px.area(mt, x='M_Name', y='Price', markers=True, color_discrete_sequence=['#00CC96']), use_container_width=True)

    def show_manage_data_page():
        st.title("📥 จัดการข้อมูล & อัปเดตโมเดล")
        
        # [NEW] แยก Tab: นำเข้าข้อมูล vs สั่งเทรนโมเดล
        tab1, tab2 = st.tabs(["1. นำเข้าข้อมูลใหม่", "2. อัปเดตความฉลาด (Retrain)"])
        
        with tab1:
            st.markdown("### Import New Bookings")
            st.info("อัปโหลดไฟล์ CSV เพื่อเพิ่มข้อมูลเข้าระบบ Dashboard (ข้อมูลจะถูกต่อท้าย)")
            up_file = st.file_uploader("เลือกไฟล์ CSV", type=['csv'])
            if up_file is not None:
                st.write("Preview:")
                st.dataframe(pd.read_csv(up_file).head())
                if st.button("บันทึกข้อมูลเข้าระบบ", type="primary"):
                    if save_uploaded_data(up_file):
                        st.success("✅ บันทึกเรียบร้อย!")
                        st.rerun()
                        
        with tab2:
            st.markdown("### 🔄 On-Demand Model Retraining")
            st.warning("⚠️ ระบบจะนำข้อมูลทั้งหมดที่มี (CSV) มาสร้างโมเดลใหม่ เพื่อให้ค่าพยากรณ์แม่นยำขึ้น")
            col_metrics1, col_metrics2 = st.columns(2)
            with col_metrics1: st.metric("Current XGB Accuracy (R²)", f"{metrics['xgb']['r2']*100:.2f}%")
            
            if st.button("🚀 เริ่มกระบวนการเรียนรู้ใหม่ (Start Retraining)", type="secondary"):
                success, count = retrain_system()
                if success:
                    st.balloons()
                    st.success(f"🎉 โมเดลเรียนรู้ข้อมูลครบ {count:,} รายการเรียบร้อยแล้ว!")
                    st.rerun()

    def show_pricing_page():
        st.title("🔮 ระบบพยากรณ์ราคา (Price Forecasting)")
        if xgb_model is None: st.error("❌ Model not found"); return

        with st.container(border=True):
            st.subheader("🛠️ กำหนดตัวแปร")
            c1, c2, c3 = st.columns(3)
            with c1:
                checkin = st.date_input("Check-in", datetime.now())
                nights = st.number_input("Nights", 1, 30, 1)
            with c2:
                room = st.selectbox("Room Type", le_room.classes_)
                guests = st.number_input("Guests", 1, 10, 2)
            with c3:
                res = st.selectbox("Channel", le_res.classes_)
                is_h = checkin in holidays.Thailand()
                st.info(f"Holiday: {'✅ Yes' if is_h else '❌ No'}")

            if st.button("🚀 คำนวณราคา", type="primary", use_container_width=True):
                # Prepare Input
                r_code = le_room.transform([room])[0]
                res_code = le_res.transform([res])[0]
                inp = pd.DataFrame([{
                    'Night': nights, 'total_guests': guests, 
                    'is_holiday': 1 if is_h else 0, 'is_weekend': 1 if checkin.weekday() in [5,6] else 0,
                    'month': checkin.month, 'weekday': checkin.weekday(),
                    'RoomType_encoded': r_code, 'Reservation_encoded': res_code
                }])
                
                # Predict
                p_xgb = xgb_model.predict(inp)[0]
                p_lr = lr_model.predict(inp)[0]
                
                # [NEW] ใช้ค่า Metrics จริง (Dynamic) แทนค่า Hardcode
                st.divider()
                cr1, cr2 = st.columns(2)
                with cr1:
                    st.success("### ⚡ XGBoost (Recommended)")
                    st.metric("ราคาแนะนำ", f"{p_xgb:,.0f} THB")
                    st.caption(f"MAE: ±{metrics['xgb']['mae']:,.0f} | R²: {metrics['xgb']['r2']:.4f}")
                with cr2:
                    st.warning("### 📉 Linear Regression")
                    st.metric("ราคาประเมิน", f"{p_lr:,.0f} THB")
                    st.caption(f"MAE: ±{metrics['lr']['mae']:,.0f} | R²: {metrics['lr']['r2']:.4f}")

    def show_model_insight_page():
        st.title("🧠 วิเคราะห์ปัจจัยโมเดล (Model Factor Analysis)")
        # Static Thesis Data as requested
        data = {'Feature': ['Night', 'Reservation', 'Month', 'Is Weekend', 'Room Type', 'Weekday', 'Guests', 'Is Holiday'],
                'Importance': [0.4364, 0.1742, 0.1315, 0.0643, 0.0640, 0.0512, 0.0508, 0.0275]}
        fi_df = pd.DataFrame(data).sort_values('Importance', ascending=True)
        st.plotly_chart(px.bar(fi_df, x='Importance', y='Feature', orientation='h', title='Feature Importance (Thesis Baseline)', color='Importance'), use_container_width=True)

    def show_about_page():
        st.title("ℹ️ เกี่ยวกับระบบ / ผู้จัดทำ")
        st.divider()
        c1, c2 = st.columns([1, 2])
        with c1: st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=200)
        with c2:
            st.header("ผู้จัดทำ")
            st.markdown("**ว่าที่ร้อยตรีพรพินิต วิรัตน์สกุลชัย** สาขาวิทยาการข้อมูล และ นวัตกรรมดิจิทัล\n\nคณะ นวัตกรรม เทคโนโลยีและการสร้างสรรค์ **มหาวิทยาลัยฟาร์อีสเทอร์น**")
            st.divider()
            st.info("วิทยานิพนธ์: การพัฒนาระบบสนับสนุนการตัดสินใจเพื่อการพยากรณ์ราคาแบบพลวัต")

    # --- Sidebar ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2933/2933116.png", width=80)
        st.markdown(f"### User: {st.session_state['username']}")
        
        page = st.radio("เมนูใช้งาน:", ["🏠 หน้าหลัก", "📊 แดชบอร์ด", "📥 จัดการข้อมูล", "🔮 พยากรณ์ราคา", "🧠 วิเคราะห์โมเดล", "ℹ️ เกี่ยวกับระบบ"])
        
        st.divider()
        st.markdown("#### ⚙️ Real-time Performance")
        # [NEW] แสดงค่า Dynamic Metrics ใน Sidebar
        st.progress(metrics['xgb']['r2'], text=f"XGBoost: {metrics['xgb']['r2']*100:.1f}%")
        st.progress(metrics['lr']['r2'], text=f"Linear Reg: {metrics['lr']['r2']*100:.1f}%")
        
        st.divider()
        if st.button("Logout"):
            st.session_state['logged_in'] = False
            st.rerun()

    # --- Routing ---
    if "หน้าหลัก" in page: show_home_page()
    elif "แดชบอร์ด" in page: show_dashboard_page()
    elif "จัดการข้อมูล" in page: show_manage_data_page()
    elif "พยากรณ์ราคา" in page: show_pricing_page()
    elif "วิเคราะห์โมเดล" in page: show_model_insight_page()
    elif "เกี่ยวกับระบบ" in page: show_about_page()
