import streamlit as st
import pandas as pd
import joblib
import sqlite3
import os
import json
import holidays
import plotly.express as px
import gdown
from datetime import datetime

# Import Library สำหรับการ Retrain
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================================
# 1. SETUP & CONSTANTS (ตั้งค่าระบบ)
# ==========================================================
st.set_page_config(
    page_title="Hotel Price Forecasting System",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_FILE = "users.db"
DATA_FILE = "check_in_report.csv"
ROOM_FILE = "room_type.csv" # <--- [ปรับ] ไฟล์สำหรับแปลงเลขห้องเป็นชื่อห้อง
METRICS_FILE = "model_metrics.json"

MODEL_FILES = {
    'xgb': 'xgb_hotel_model.joblib',
    'lr': 'lr_hotel_model.joblib',
    'le_room': 'le_room.joblib',
    'le_res': 'le_res.joblib'
}

# [ปรับ] ราคา Base Price ตามที่คุณระบุ
BASE_PRICES = {
    'Grand Suite Room': 2700,
    'Villa Suite (Garden)': 2700,
    'Executive Room': 2500,
    'Executive Room with Balcony': 2400,
    'Villa Suite (Bathtub)': 2000,
    'Deluxe Room': 1500,
    'Standard Room': 1000
}

# ค่า Default (Thesis Baseline) ใช้กรณีเริ่มระบบครั้งแรกหรือหาไฟล์ไม่เจอ
DEFAULT_METRICS = {
    'xgb': {'mae': 1112.79, 'r2': 0.7256},
    'lr':  {'mae': 1162.27, 'r2': 0.7608},
    'importance': {
        'Night': 0.4364, 'Reservation': 0.1742, 'Month': 0.1315, 
        'Is Weekend': 0.0643, 'Room Type': 0.0640, 'Weekday': 0.0512, 
        'Guests': 0.0508, 'Is Holiday': 0.0275
    }
}

if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'username' not in st.session_state: st.session_state['username'] = ""

# ==========================================================
# 2. DATABASE (ระบบจัดการ User)
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
    except sqlite3.IntegrityError: return False

init_db()

# ==========================================================
# 3. BACKEND SYSTEM (ระบบหลังบ้าน)
# ==========================================================

@st.cache_data
def load_data():
    # 1. โหลดข้อมูล Booking
    if not os.path.exists(DATA_FILE):
        try: gdown.download("https://drive.google.com/uc?id=1dxgKIvSTelLaJvAtBSCMCU5K4FuJvfri", DATA_FILE, quiet=True)
        except: return pd.DataFrame()

    try:
        df = pd.read_csv(DATA_FILE)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        
        # 2. [ปรับ] โหลดข้อมูล Room Type Mapping และ Merge ทันที
        # เพื่อให้ทุกหน้าเห็นเป็น "ชื่อห้อง" ไม่ใช่ "เลขห้อง"
        if os.path.exists(ROOM_FILE):
            try:
                room_type = pd.read_csv(ROOM_FILE)
                if 'Room_Type' in room_type.columns:
                    # Merge โดยใช้คอลัมน์ Room เป็นตัวเชื่อม
                    df = df.merge(room_type, on='Room', how='left')
                    
                    # จัดการชื่อคอลัมน์หลัง Merge
                    if 'Room_Type_y' in df.columns: 
                        df = df.rename(columns={'Room_Type_y': 'Target_Room_Type'})
                    elif 'Room_Type' in df.columns:
                        df = df.rename(columns={'Room_Type': 'Target_Room_Type'})
            except:
                pass

        # 3. [ปรับ] Fallback: ถ้ายังไม่มีชื่อห้อง ให้พยายามใช้คอลัมน์เดิม
        if 'Target_Room_Type' not in df.columns:
            if 'Room_Type' in df.columns:
                df['Target_Room_Type'] = df['Room_Type']
            else:
                df['Target_Room_Type'] = df['Room'].astype(str) # ถ้าไม่มีจริงๆ ให้ใช้เลขห้องไปก่อน

        df['Target_Room_Type'] = df['Target_Room_Type'].fillna('Unknown')
        df['Reservation'] = df['Reservation'].fillna('Unknown')
        df['month'] = df['Date'].dt.month
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
    
    # [ปรับ] โหลดค่า Metrics ล่าสุดจากไฟล์ JSON
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, 'r') as f: metrics = json.load(f)
    else: metrics = DEFAULT_METRICS
        
    return xgb, lr, le_room, le_res, metrics

def save_uploaded_data(uploaded_file, is_room_file=False):
    try:
        uploaded_file.seek(0)
        new_data = pd.read_csv(uploaded_file)
        
        if is_room_file:
            # กรณีอัปโหลดไฟล์ชื่อห้อง ให้เซฟทับไฟล์ room_type.csv เลย
            new_data.to_csv(ROOM_FILE, index=False)
        else:
            # กรณีอัปโหลด Booking ให้ต่อท้ายไฟล์เดิม
            if os.path.exists(DATA_FILE):
                current_df = pd.read_csv(DATA_FILE)
                updated_df = pd.concat([current_df, new_data], ignore_index=True)
            else:
                updated_df = new_data
            updated_df.to_csv(DATA_FILE, index=False)
            
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"Save failed: {e}")
        return False

# --- RETRAIN FUNCTION (ระบบเทรนโมเดลใหม่) ---
def retrain_system():
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        status_text.text("⏳ Reading all data...")
        df = pd.read_csv(DATA_FILE)
        
        # --- Preprocessing Pipeline ---
        if 'Date' in df.columns: df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        
        # Merge Room Type (ทำเหมือน load_data เพื่อให้ข้อมูลตรงกัน)
        if os.path.exists(ROOM_FILE):
            room_type = pd.read_csv(ROOM_FILE)
            if 'Room_Type' in room_type.columns:
                 df = df.merge(room_type, on='Room', how='left')
                 if 'Room_Type_y' in df.columns: df = df.rename(columns={'Room_Type_y': 'Target_Room_Type'})
                 elif 'Room_Type' in df.columns: df = df.rename(columns={'Room_Type': 'Target_Room_Type'})
        
        if 'Target_Room_Type' not in df.columns:
             df['Target_Room_Type'] = df['Room'].astype(str)

        df = df.dropna(subset=['Date'])
        df['Target_Room_Type'] = df['Target_Room_Type'].fillna('Standard Room')
        df['Reservation'] = df['Reservation'].fillna('Unknown')
        
        # Holiday Handling
        if not os.path.exists("thai_holidays.csv"):
             try: gdown.download("https://drive.google.com/uc?id=1L-pciKEeRce1gzuhdtpIGcLs0fYHnbZw", "thai_holidays.csv", quiet=True)
             except: pass
        
        if os.path.exists("thai_holidays.csv"):
            holidays_csv = pd.read_csv("thai_holidays.csv")
            holidays_csv['Holiday_Date'] = pd.to_datetime(holidays_csv['Holiday_Date'], dayfirst=True, errors='coerce')
            df['is_holiday'] = df['Date'].isin(holidays_csv['Holiday_Date']).astype(int)
        else: df['is_holiday'] = 0

        df['is_weekend'] = df['Date'].dt.weekday.isin([5, 6]).astype(int)
        df['total_guests'] = df[['Adults', 'Children', 'Infants', 'Extra Person']].fillna(0).sum(axis=1)
        df['month'] = df['Date'].dt.month
        df['weekday'] = df['Date'].dt.weekday
        
        # Encoders
        le_room_new = LabelEncoder()
        df['RoomType_encoded'] = le_room_new.fit_transform(df['Target_Room_Type'].astype(str))
        le_res_new = LabelEncoder()
        df['Reservation_encoded'] = le_res_new.fit_transform(df['Reservation'].astype(str))
        
        feature_cols = ['Night', 'total_guests', 'is_holiday', 'is_weekend', 'month', 'weekday', 'RoomType_encoded', 'Reservation_encoded']
        X = df[feature_cols]
        y = df['Price']
        
        progress_bar.progress(40)
        status_text.text("🏋️‍♂️ Training new models...")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 1. XGBoost
        xgb_new = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        xgb_new.fit(X_train, y_train)
        pred_xgb = xgb_new.predict(X_test)
        new_xgb_mae = mean_absolute_error(y_test, pred_xgb)
        new_xgb_r2 = r2_score(y_test, pred_xgb)
        
        # [ปรับ] คำนวณ Feature Importance จริงๆ จากข้อมูลใหม่
        fi_raw = xgb_new.feature_importances_
        col_mapping = {'Night': 'Night', 'total_guests': 'Guests', 'is_holiday': 'Is Holiday', 'is_weekend': 'Is Weekend', 'month': 'Month', 'weekday': 'Weekday', 'RoomType_encoded': 'Room Type', 'Reservation_encoded': 'Reservation'}
        new_importance = {col_mapping.get(col, col): float(val) for col, val in zip(feature_cols, fi_raw)}

        # 2. Linear Regression
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
        
        # [ปรับ] บันทึกทั้ง Scores และ Importance ลง JSON
        new_metrics = {
            'xgb': {'mae': new_xgb_mae, 'r2': new_xgb_r2},
            'lr':  {'mae': new_lr_mae, 'r2': new_lr_r2},
            'importance': new_importance
        }
        with open(METRICS_FILE, 'w') as f: json.dump(new_metrics, f)
            
        st.cache_resource.clear()
        progress_bar.progress(100)
        status_text.success(f"✅ Retraining Complete! New R²: {new_xgb_r2:.4f}")
        return True, len(df)
        
    except Exception as e:
        st.error(f"Retrain Error: {e}")
        return False, 0

# ==========================================================
# 4. MAIN UI PAGES (ส่วนแสดงผล)
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
            u = st.text_input("Username"); p = st.text_input("Password", type="password")
            if st.button("Login", type="primary", use_container_width=True):
                if login_user(u, p): st.session_state['logged_in'] = True; st.session_state['username'] = u; st.rerun()
                else: st.error("Invalid Login")
        with tab_reg:
            nu = st.text_input("New User"); np = st.text_input("New Pass", type="password")
            if st.button("Register", use_container_width=True):
                if register_user(nu, np): st.success("Success!")
                else: st.error("Exists")

if not st.session_state['logged_in']:
    login_page()
else:
    df = load_data()
    xgb_model, lr_model, le_room, le_res, metrics = load_system_models()
    if not os.path.exists("thai_holidays.csv"):
        try: gdown.download("https://drive.google.com/uc?id=1L-pciKEeRce1gzuhdtpIGcLs0fYHnbZw", "thai_holidays.csv", quiet=True)
        except: pass

    def show_home_page():
        if os.path.exists("cover.jpg"): st.image("cover.jpg", use_container_width=True)
        else: st.image("https://images.unsplash.com/photo-1566073771259-6a8506099945?q=80", use_container_width=True)
        st.title("ระบบการพยากรณ์ราคาห้องพัก 👋")
        st.markdown("""
        **ความสามารถของระบบ:**
        * **📊 Data Analytics:** วิเคราะห์ข้อมูลการจองย้อนหลัง (Dynamic)
        * **🔮 Price Forecasting:** พยากรณ์ราคาที่เหมาะสม (AI-Powered)
        * **🔄 Adaptive Learning:** ระบบสามารถเรียนรู้ข้อมูลใหม่ได้ (Retrain)
        """)

    def show_dashboard_page():
        st.title("📊 แดชบอร์ดสรุปข้อมูล")
        if df.empty: st.warning("No Data"); return
        st.divider()
        k1, k2, k3, k4 = st.columns(4)
        with k1: st.metric("📦 Booking", f"{len(df):,}")
        with k2: st.metric("💰 Revenue", f"{df['Price'].sum()/1e6:.2f} M THB")
        with k3: st.metric("🏷️ ADR", f"{df['Price'].mean():,.0f} THB")
        with k4: st.metric("🌙 LOS", f"{df['Night'].mean():.1f} คืน")
        st.divider()
        
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("**🏆 ยอดจองตามประเภทห้อง (Room Type)**")
            # [ปรับ] ใช้ Target_Room_Type (ชื่อห้อง) แทนเลขห้อง
            if 'Target_Room_Type' in df.columns:
                rc = df['Target_Room_Type'].value_counts().reset_index()
                rc.columns = ['Room', 'Count']
                st.plotly_chart(px.bar(rc, x='Count', y='Room', orientation='h', text='Count', color='Count', color_continuous_scale='Viridis'), use_container_width=True)
            else:
                st.warning("⚠️ แสดงเป็นเลขห้องเนื่องจากไม่พบไฟล์ Room Mapping")
                rc = df['Room'].value_counts().head(20).reset_index() 
                rc.columns = ['Room', 'Count']
                st.plotly_chart(px.bar(rc, x='Count', y='Room', orientation='h', text='Count'), use_container_width=True)
                
        with c2:
            st.markdown("**💸 สัดส่วนรายได้**")
            group_col = 'Target_Room_Type' if 'Target_Room_Type' in df.columns else 'Room'
            rev = df.groupby(group_col)['Price'].sum().reset_index()
            st.plotly_chart(px.pie(rev, values='Price', names=group_col, hole=0.4), use_container_width=True)
        
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
        # [ปรับ] เพิ่ม Tab สำหรับอัปโหลดชื่อห้อง
        tab1, tab2, tab3 = st.tabs(["1. นำเข้า Booking", "2. อัปเดตชื่อห้อง (Room Type)", "3. อัปเดตโมเดล (Retrain)"])
        
        with tab1:
            st.markdown("### Import New Bookings (CSV)")
            st.info("อัปโหลดไฟล์จองห้องพัก (ต้องมีคอลัมน์ Room, Price, Date, ...)")
            up_file = st.file_uploader("เลือกไฟล์ Booking CSV", type=['csv'], key="booking_up")
            if up_file is not None:
                if st.button("บันทึกข้อมูล Booking", type="primary"):
                    if save_uploaded_data(up_file, is_room_file=False):
                        # [ปรับ] แจ้งเตือนเมื่อนำเข้าสำเร็จ
                        st.success("✅ นำเข้าข้อมูล Booking เรียบร้อยแล้ว!")
                        st.balloons()
                        st.rerun()

        with tab2:
            st.markdown("### Update Room Mapping (CSV)")
            st.info("อัปโหลดไฟล์จับคู่เลขห้องกับชื่อห้อง (Columns: Room, Room_Type)")
            room_file = st.file_uploader("เลือกไฟล์ Room Type CSV", type=['csv'], key="room_up")
            if room_file is not None:
                if st.button("บันทึกข้อมูลชื่อห้อง", type="secondary"):
                    if save_uploaded_data(room_file, is_room_file=True):
                        st.success("✅ อัปเดตชื่อห้องเรียบร้อย! Dashboard จะแสดงชื่อห้องแทนเลขห้องแล้ว")
                        st.rerun()
                        
        with tab3:
            st.markdown("### 🔄 On-Demand Model Retraining")
            st.warning("⚠️ การกดปุ่มนี้จะทำให้โมเดลเรียนรู้ข้อมูลใหม่ และค่าความแม่นยำจะเปลี่ยนไป")
            if st.button("🚀 เริ่มกระบวนการเรียนรู้ใหม่ (Start Retraining)"):
                success, count = retrain_system()
                if success:
                    st.success(f"🎉 โมเดลเรียนรู้ครบ {count:,} รายการ! ค่า MAE/R2 และ Feature Importance ถูกอัปเดตแล้ว")
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
                r_code = le_room.transform([room])[0]
                res_code = le_res.transform([res])[0]
                inp = pd.DataFrame([{
                    'Night': nights, 'total_guests': guests, 
                    'is_holiday': 1 if is_h else 0, 'is_weekend': 1 if checkin.weekday() in [5,6] else 0,
                    'month': checkin.month, 'weekday': checkin.weekday(),
                    'RoomType_encoded': r_code, 'Reservation_encoded': res_code
                }])
                
                p_xgb = xgb_model.predict(inp)[0]
                p_lr = lr_model.predict(inp)[0]
                
                # [ปรับ] เทียบ Base Price
                base_price = 0
                for key in BASE_PRICES:
                    if key in room: base_price = BASE_PRICES[key]; break
                
                st.divider()
                c_base, c_xgb, c_lr = st.columns(3)
                with c_base:
                    st.info("### 🏷️ Base Price")
                    st.metric("ราคาตั้งต้น", f"{base_price:,.0f} THB")
                    st.caption("ราคามาตรฐานโรงแรม")
                with c_xgb:
                    st.success("### ⚡ XGBoost (AI)")
                    st.metric("ราคาแนะนำ", f"{p_xgb:,.0f} THB", delta=f"{p_xgb - base_price:,.0f} THB")
                    st.caption(f"MAE: ±{metrics['xgb']['mae']:,.0f} | R²: {metrics['xgb']['r2']:.4f}")
                with c_lr:
                    st.warning("### 📉 Linear Reg")
                    st.metric("ราคาประเมิน", f"{p_lr:,.0f} THB", delta=f"{p_lr - base_price:,.0f} THB")
                    st.caption(f"MAE: ±{metrics['lr']['mae']:,.0f} | R²: {metrics['lr']['r2']:.4f}")

    def show_model_insight_page():
        st.title("🧠 วิเคราะห์ปัจจัยโมเดล (Dynamic Insight)")
        # [ปรับ] ดึง Feature Importance จากไฟล์ Metrics (ไม่ใช่ Hardcode)
        imp_data = metrics.get('importance', DEFAULT_METRICS['importance'])
        fi_df = pd.DataFrame(list(imp_data.items()), columns=['Feature', 'Importance']).sort_values('Importance', ascending=True)
        
        st.divider()
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("📊 กราฟความสำคัญ")
            st.plotly_chart(px.bar(fi_df, x='Importance', y='Feature', orientation='h', text_auto='.4f', color='Importance', color_continuous_scale='Blues'), use_container_width=True)
        with c2:
            st.subheader("📋 ตารางข้อมูล")
            # [ปรับ] โชว์ตารางให้เห็นชัดๆ
            st.dataframe(fi_df.sort_values('Importance', ascending=False), use_container_width=True, height=400)

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

    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2933/2933116.png", width=80)
        st.markdown(f"### User: {st.session_state['username']}")
        page = st.radio("เมนูใช้งาน:", ["🏠 หน้าหลัก", "📊 แดชบอร์ด", "📥 จัดการข้อมูล", "🔮 พยากรณ์ราคา", "🧠 วิเคราะห์โมเดล", "ℹ️ เกี่ยวกับระบบ"])
        st.divider()
        st.markdown("#### ⚙️ Real-time Performance")
        st.progress(metrics['xgb']['r2'], text=f"XGBoost: {metrics['xgb']['r2']*100:.1f}%")
        st.progress(metrics['lr']['r2'], text=f"Linear Reg: {metrics['lr']['r2']*100:.1f}%")
        st.divider()
        if st.button("Logout"): st.session_state['logged_in'] = False; st.rerun()

    if "หน้าหลัก" in page: show_home_page()
    elif "แดชบอร์ด" in page: show_dashboard_page()
    elif "จัดการข้อมูล" in page: show_manage_data_page()
    elif "พยากรณ์ราคา" in page: show_pricing_page()
    elif "วิเคราะห์โมเดล" in page: show_model_insight_page()
    elif "เกี่ยวกับระบบ" in page: show_about_page()