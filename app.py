import streamlit as st
import pandas as pd
import joblib
import sqlite3
import os
import holidays
import plotly.express as px
import gdown
from datetime import datetime

# ==========================================================
# 1. SETUP PAGE CONFIG & SESSION STATE
# ==========================================================
st.set_page_config(
    page_title="Hotel Price Forecasting System",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DB_FILE = "users.db"
DATA_FILE = "check_in_report.csv"
MODEL_FILES = {
    'xgb': 'xgb_hotel_model.joblib',
    'lr': 'lr_hotel_model.joblib',
    'le_room': 'le_room.joblib',
    'le_res': 'le_res.joblib'
}

# Check Login State
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ""

# ==========================================================
# 2. DATABASE SYSTEM (SQLite)
# ==========================================================
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)')
    # Default Admin
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

init_db() # Run on start

# ==========================================================
# 3. LOGIN PAGE UI (Original Design)
# ==========================================================
def login_page():
    st.markdown("""
        <style>
            .stTextInput > div > div > input {text-align: center;}
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/2933/2933116.png", width=120)
        st.title("🔒 Login System")
        st.markdown("ระบบการพยากรณ์ราคาห้องพัก (Hotel Price Forecasting System)")
        
        tab_login, tab_register = st.tabs(["เข้าสู่ระบบ", "สมัครสมาชิก"])
        
        with tab_login:
            username = st.text_input("Username", placeholder="Username")
            password = st.text_input("Password", type="password", placeholder="Password")
            
            if st.button("เข้าสู่ระบบ (Login)", type="primary", use_container_width=True):
                if login_user(username, password):
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    st.rerun()
                else:
                    st.error("Username หรือ Password ไม่ถูกต้อง")

        with tab_register:
            new_user = st.text_input("New Username", placeholder="ตั้งชื่อผู้ใช้ใหม่")
            new_pass = st.text_input("New Password", type="password", placeholder="ตั้งรหัสผ่าน")
            if st.button("สมัครสมาชิก (Register)", use_container_width=True):
                if new_user and new_pass:
                    if register_user(new_user, new_pass):
                        st.success("สมัครสำเร็จ! กรุณากลับไปหน้า Login")
                    else:
                        st.error("ชื่อผู้ใช้นี้มีอยู่ในระบบแล้ว")
                else:
                    st.warning("กรุณากรอกข้อมูลให้ครบ")
        
        st.divider()

# ==========================================================
# 4. SYSTEM BACKEND (Dynamic Engine)
# ==========================================================

# A. Data Loader (Dynamic)
@st.cache_data
def load_data():
    # ถ้าไม่มีไฟล์ ให้โหลดจาก Google Drive เป็นค่าตั้งต้น
    if not os.path.exists(DATA_FILE):
        url_main = "https://drive.google.com/uc?id=1dxgKIvSTelLaJvAtBSCMCU5K4FuJvfri"
        try:
            gdown.download(url_main, DATA_FILE, quiet=True)
        except:
            return pd.DataFrame() # Return empty if failed

    try:
        df = pd.read_csv(DATA_FILE)
        # Preprocessing
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        
        # จัดการข้อมูลห้อง (ถ้ามีไฟล์ room_type.csv ก็โหลดมา merge ได้ ถ้าไม่มีก็ข้าม)
        if os.path.exists("room_type.csv"):
            room_type = pd.read_csv("room_type.csv")
            if 'Room_Type' in room_type.columns:
                 room_type = room_type.rename(columns={'Room_Type': 'Target_Room_Type'})
            df = df.merge(room_type, on='Room', how='left')
            df['Target_Room_Type'] = df['Target_Room_Type'].fillna('Standard Room')
        elif 'Target_Room_Type' not in df.columns:
             # กรณีไม่มีไฟล์ room_type ให้สร้างคอลัมน์หลอกๆ หรือดึงจาก Room
             df['Target_Room_Type'] = df.get('Room', 'Unknown')

        # Feature Engineering เบื้องต้นสำหรับ Dashboard
        df['Reservation'] = df['Reservation'].fillna('Unknown')
        df['month'] = df['Date'].dt.month
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# B. Model Loader (Joblib)
@st.cache_resource
def load_system_models():
    # Check files
    for name, file in MODEL_FILES.items():
        if not os.path.exists(file):
            return None, None, None, None

    xgb = joblib.load(MODEL_FILES['xgb'])
    lr = joblib.load(MODEL_FILES['lr'])
    le_room = joblib.load(MODEL_FILES['le_room'])
    le_res = joblib.load(MODEL_FILES['le_res'])
    return xgb, lr, le_room, le_res

# C. Save Data Function
def save_uploaded_data(uploaded_file):
    try:
        # 👇 [สำคัญ] เพิ่มบรรทัดนี้ เพื่อสั่งรีเซ็ตหัวอ่านกลับไปจุดเริ่มต้น
        uploaded_file.seek(0) 
        
        new_data = pd.read_csv(uploaded_file)
        
        if os.path.exists(DATA_FILE):
            current_df = pd.read_csv(DATA_FILE)
            # ใช้ concat เพื่อต่อท้าย (ignore_index สำคัญมาก)
            updated_df = pd.concat([current_df, new_data], ignore_index=True)
        else:
            updated_df = new_data
            
        updated_df.to_csv(DATA_FILE, index=False)
        st.cache_data.clear() # เคลียร์ Cache เพื่อให้ Dashboard อัปเดต
        return True
    except Exception as e:
        st.error(f"Save failed: {e}")
        return False

# ==========================================================
# 5. MAIN APP LOGIC & PAGES
# ==========================================================

if not st.session_state['logged_in']:
    login_page()
else:
    # ----------------------------------------
    # LOAD RESOURCES
    # ----------------------------------------
    df = load_data()
    xgb_model, lr_model, le_room, le_res = load_system_models()

    # Load Holidays for Utility
    if not os.path.exists("thai_holidays.csv"):
        try:
             gdown.download("https://drive.google.com/uc?id=1L-pciKEeRce1gzuhdtpIGcLs0fYHnbZw", "thai_holidays.csv", quiet=True)
        except: pass
    
    # ----------------------------------------
    # PAGE FUNCTIONS
    # ----------------------------------------

    def show_home_page():
        if os.path.exists("cover.jpg"):
            st.image("cover.jpg", use_container_width=True)
        else:
            st.image("https://images.unsplash.com/photo-1566073771259-6a8506099945?q=80", use_container_width=True)
        
        st.title("ระบบการพยากรณ์ราคาห้องพัก 👋")
        st.subheader("(Hotel Price Forecasting System)")
        st.markdown(f"""
        ยินดีต้อนรับเข้าสู่ระบบสนับสนุนการตัดสินใจ (Decision Support System) 
        สำหรับผู้บริหารและฝ่ายจัดการโรงแรม
        
        **ความสามารถของระบบ:**
        * **📊 Data Analytics:** วิเคราะห์ข้อมูลการจองย้อนหลัง (Dynamic Data)
        * **🔮 Price Forecasting:** พยากรณ์ราคาที่เหมาะสม (XGBoost & Linear Regression)
        * **🧠 Insight Analysis:** วิเคราะห์ปัจจัยที่มีผลต่อการตั้งราคา (Thesis Results)
        """)
        st.info("👈 กรุณาเลือกเมนูจากแถบด้านซ้ายเพื่อเริ่มใช้งาน")

    def show_dashboard_page():
        st.title("📊 แดชบอร์ดสรุปข้อมูล (Executive Dashboard)")
        st.markdown("### สรุปภาพรวมผลประกอบการและการวิเคราะห์แนวโน้ม")
        
        if df.empty:
            st.warning("ไม่พบข้อมูล กรุณาไปที่เมนู 'จัดการข้อมูล' เพื่อนำเข้าไฟล์")
            return

        st.divider()
        
        # KPI
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1: st.metric("📦 Booking ทั้งหมด", f"{len(df):,} รายการ", "Total Records")
        with kpi2: st.metric("💰 รายได้รวม", f"{df['Price'].sum()/1e6:.2f} M THB", "Gross Revenue")
        with kpi3: st.metric("🏷️ ราคาเฉลี่ย (ADR)", f"{df['Price'].mean():,.0f} THB", "Per Night")
        with kpi4: st.metric("🌙 จำนวนคืนเฉลี่ย", f"{df['Night'].mean():.1f} คืน", "LOS")
        
        st.divider()
        
        # ROW 1: ROOM & REVENUE
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("**🏆 ยอดจองแยกตามประเภทห้อง (Room Type Count)**")
            if 'Target_Room_Type' in df.columns:
                rc = df['Target_Room_Type'].value_counts().reset_index()
                rc.columns = ['Room', 'Count']
                fig = px.bar(rc, x='Count', y='Room', orientation='h', text='Count', color='Count', color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            st.markdown("**💸 สัดส่วนรายได้ (Revenue Share)**")
            if 'Target_Room_Type' in df.columns:
                rev = df.groupby('Target_Room_Type')['Price'].sum().reset_index()
                fig = px.pie(rev, values='Price', names='Target_Room_Type', hole=0.4)
                st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # ROW 2: RESERVATION & TREND
        c3, c4 = st.columns([2, 3])
        with c3:
            st.markdown("**🌐 ช่องทางการจอง (Reservation Channel)**")
            res_count = df['Reservation'].value_counts().reset_index()
            res_count.columns = ['Channel', 'Count']
            fig_res = px.pie(res_count, values='Count', names='Channel', hole=0.4, 
                             color_discrete_sequence=px.colors.sequential.Magma)
            st.plotly_chart(fig_res, use_container_width=True)

        with c4:
            st.markdown("**📈 แนวโน้มรายได้รายเดือน (Monthly Revenue Trend)**")
            mt = df.groupby('month')['Price'].sum().reset_index()
            mt['M_Name'] = mt['month'].apply(lambda x: datetime(2024, int(x), 1).strftime('%B'))
            mt = mt.sort_values('month')
            st.plotly_chart(px.area(mt, x='M_Name', y='Price', markers=True, color_discrete_sequence=['#00CC96']), use_container_width=True)

    def show_pricing_page():
        st.title("🔮 ระบบพยากรณ์ราคา (Price Forecasting)")
        st.markdown("เปรียบเทียบผลลัพธ์ระหว่าง **XGBoost** และ **Linear Regression**")
        
        if xgb_model is None:
            st.error("❌ ไม่พบไฟล์โมเดล (.joblib) กรุณาตรวจสอบไฟล์ในโฟลเดอร์")
            return

        with st.container(border=True):
            st.subheader("🛠️ กำหนดตัวแปรสำหรับการพยากรณ์")
            c1, c2, c3 = st.columns(3)
            with c1:
                checkin_date = st.date_input("วันที่เช็คอิน", datetime.now())
                nights = st.number_input("จำนวนคืน", 1, 30, 1)
            with c2:
                room_name = st.selectbox("ประเภทห้อง", le_room.classes_)
                guests = st.number_input("จำนวนผู้เข้าพัก", 1, 10, 2)
            with c3:
                res_name = st.selectbox("ช่องทางการจอง", le_res.classes_)
                is_h = checkin_date in holidays.Thailand()
                st.info(f"สถานะวันหยุด: {'✅ ใช่' if is_h else '❌ ไม่ใช่'}")

            if st.button("🚀 คำนวณราคาพยากรณ์", type="primary", use_container_width=True):
                # Prepare Input
                r_code = le_room.transform([room_name])[0]
                res_code = le_res.transform([res_name])[0]
                total_guests = guests # Simple mapping
                
                inp = pd.DataFrame([{
                    'Night': nights, 
                    'total_guests': total_guests, 
                    'is_holiday': 1 if is_h else 0,
                    'is_weekend': 1 if checkin_date.weekday() in [5, 6] else 0,
                    'month': checkin_date.month, 
                    'weekday': checkin_date.weekday(),
                    'RoomType_encoded': r_code, 
                    'Reservation_encoded': res_code
                }])
                
                # Predict
                p_xgb = xgb_model.predict(inp)[0]
                p_lr = lr_model.predict(inp)[0]
                
                # Metrics (Hardcoded from Thesis)
                m_xgb_mae = 1112.79
                m_lr_mae = 1162.27
                
                st.divider()
                cr1, cr2 = st.columns(2)
                with cr1:
                    st.success("### ⚡ XGBoost (แนะนำ)")
                    st.metric("ราคาที่เหมาะสม", f"{p_xgb:,.0f} THB")
                    st.caption(f"ความคลาดเคลื่อนเฉลี่ย (MAE): ±{m_xgb_mae:,.0f} บาท")
                with cr2:
                    st.warning("### 📉 Linear Regression")
                    st.metric("ราคาประเมินทั่วไป", f"{p_lr:,.0f} THB")
                    st.caption(f"ความคลาดเคลื่อนเฉลี่ย (MAE): ±{m_lr_mae:,.0f} บาท")

    def show_import_page():
        st.title("📥 จัดการข้อมูล (Data Management)")
        st.markdown("### นำเข้าข้อมูลการจองใหม่ (Import New Bookings)")
        st.info("อัปโหลดไฟล์ CSV ที่มีโครงสร้างเหมือนข้อมูลเดิม เพื่ออัปเดต Dashboard ให้เป็นปัจจุบัน")
        
        uploaded_file = st.file_uploader("เลือกไฟล์ CSV", type=['csv'])
        if uploaded_file is not None:
            st.write("ตัวอย่างข้อมูล:")
            st.dataframe(pd.read_csv(uploaded_file).head())
            
            if st.button("บันทึกข้อมูลเข้าระบบ", type="primary"):
                if save_uploaded_data(uploaded_file):
                    st.success("✅ บันทึกเรียบร้อย! ข้อมูลถูกรวมเข้ากับฐานข้อมูลหลักแล้ว")
                    st.rerun()

    def show_model_insight_page():
        # --- STATIC PAGE AS REQUESTED ---
        st.title("🧠 วิเคราะห์ปัจจัยโมเดล (Model Factor Analysis)")
        st.markdown("แสดงค่าความสำคัญของตัวแปร (Feature Importance Scores)")
        
        data_static = {
            'Feature': [
                'Night (จำนวนคืน)', 'Reservation (ช่องทางการจอง)', 'Month (เดือนที่เข้าพัก)',
                'Is Weekend (วันหยุดสุดสัปดาห์)', 'Room Type (ประเภทห้องพัก)',
                'Weekday (วันในสัปดาห์)', 'Total Guests (ผู้เข้าพักรวม)', 'Is Holiday (วันหยุดนักขัตฤกษ์)'
            ],
            'Importance': [0.4364, 0.1742, 0.1315, 0.0643, 0.0640, 0.0512, 0.0508, 0.0275]
        }
        
        fi_df = pd.DataFrame(data_static).sort_values('Importance', ascending=True)
        
        st.divider()
        st.subheader("กราฟแสดงน้ำหนักความสำคัญของตัวแปร (ตามตาราง)")
        
        fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h', 
                     title='Feature Importance Score', text_auto='.4f', 
                     color='Importance', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("ดูข้อมูลแบบตาราง (Table View)", expanded=True):
            display_df = fi_df.sort_values('Importance', ascending=False)
            display_df['Percentage'] = (display_df['Importance'] * 100).map('{:.2f}%'.format)
            st.dataframe(display_df, use_container_width=True)

        st.info("คำอธิบาย: Night (จำนวนคืน) มีผลต่อราคามากที่สุด (43.64%) ตามด้วย Reservation (17.42%)")

    def show_about_page():
        # --- STATIC PAGE AS REQUESTED ---
        st.title("ℹ️ เกี่ยวกับระบบ / ผู้จัดทำ")
        st.divider()
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=200)
        with col2:
            st.header("ผู้จัดทำ")
            st.markdown("""
            **ว่าที่ร้อยตรีพรพินิต วิรัตน์สกุลชัย** สาขาวิทยาการข้อมูล และ นวัตกรรมดิจิทัล  
            คณะ นวัตกรรม เทคโนโลยีและการสร้างสรรค์  
            **มหาวิทยาลัยฟาร์อีสเทอร์น**
            """)
            st.divider()
            st.header("รายละเอียดโครงการ")
            st.info("โปรแกรมนี้เป็นส่วนหนึ่งของวิทยานิพนธ์เรื่อง *การพัฒนาระบบสนับสนุนการตัดสินใจเพื่อการพยากรณ์ราคาแบบพลวัตสำหรับธุรกิจโรงแรม*")
        
        st.divider()
        st.header("📜 กิตติกรรมประกาศ")
        st.markdown("*ขอขอบพระคุณ ผู้ช่วยศาสตราจารย์ ดร.พงศ์กร จันทราช และคณาจารย์ทุกท่าน*")

    # ----------------------------------------
    # SIDEBAR NAVIGATION
    # ----------------------------------------
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2933/2933116.png", width=80)
        st.markdown("### Admin Panel")
        st.caption(f"User: **{st.session_state['username']}** | Status: **Online**")
        
        selected_page = st.radio("เมนูใช้งาน:", 
            ["🏠 หน้าหลัก", "📊 แดชบอร์ด", "📥 จัดการข้อมูล", "🔮 พยากรณ์ราคา", "🧠 วิเคราะห์โมเดล", "ℹ️ เกี่ยวกับระบบ"]
        )
        
        st.divider()
        st.markdown("#### ⚙️ ประสิทธิภาพโมเดล")
        # Static Metrics Display for Sidebar
        st.progress(0.7256, text="XGBoost Score: 72.6%")
        st.progress(0.7608, text="Linear Reg Score: 76.1%")
        
        st.divider()
        if st.button("ออกจากระบบ (Logout)", type="secondary"):
            st.session_state['logged_in'] = False
            st.rerun()

    # ----------------------------------------
    # PAGE ROUTING
    # ----------------------------------------
    if "หน้าหลัก" in selected_page: show_home_page()
    elif "แดชบอร์ด" in selected_page: show_dashboard_page()
    elif "จัดการข้อมูล" in selected_page: show_import_page() # New Dynamic Page
    elif "พยากรณ์ราคา" in selected_page: show_pricing_page()
    elif "วิเคราะห์โมเดล" in selected_page: show_model_insight_page()
    elif "เกี่ยวกับระบบ" in selected_page: show_about_page()