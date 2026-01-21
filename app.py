import streamlit as st
import pandas as pd

# Import Logic และ Backend จาก utils.py
from utils import (
    init_db, 
    login_user, 
    register_user, 
    load_data, 
    calculate_historical_avg, 
    load_system_models
)

# Import หน้า UI จากไฟล์แยก (ต้องตั้งชื่อไฟล์ให้ตรงกัน)
import dashboard_page
import manage_page
import pricing_page
import insight_page
import about_page

# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================
st.set_page_config(
    page_title="Hotel Price Forecasting System",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# 2. SESSION STATE MANAGEMENT
# ==========================================================
if 'logged_in' not in st.session_state: 
    st.session_state['logged_in'] = False
if 'username' not in st.session_state: 
    st.session_state['username'] = ""
if 'historical_avg' not in st.session_state: 
    st.session_state['historical_avg'] = {}

# Initialize Database
init_db()

# ==========================================================
# 3. LOGIN PAGE UI
# ==========================================================
def show_login_screen():
    st.markdown("""<style>.stTextInput > div > div > input {text-align: center;}</style>""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/2933/2933116.png", width=120)
        st.title("🔒 Login System")
        st.markdown("ระบบการพยากรณ์ราคาห้องพัก (Hotel Price Forecasting System)")
        
        tab_log, tab_reg = st.tabs(["เข้าสู่ระบบ (Login)", "ลงทะเบียน (Register)"])
        
        with tab_log:
            u = st.text_input("Username", key="login_u")
            p = st.text_input("Password", type="password", key="login_p")
            if st.button("Login", type="primary", use_container_width=True):
                if login_user(u, p): 
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = u
                    st.rerun()
                else: 
                    st.error("ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง")
        
        with tab_reg:
            nu = st.text_input("New Username", key="reg_u")
            np = st.text_input("New Password", type="password", key="reg_p")
            if st.button("Register", use_container_width=True):
                if register_user(nu, np): 
                    st.success("ลงทะเบียนสำเร็จ! กรุณาเข้าสู่ระบบ")
                else: 
                    st.error("ชื่อผู้ใช้นี้มีอยู่ในระบบแล้ว")

# ==========================================================
# 4. MAIN NAVIGATION & ROUTING
# ==========================================================
if not st.session_state['logged_in']:
    show_login_screen()
else:
    # --- Load Data & Models Once ---
    df_raw = load_data() 
    
    # คำนวณค่าเฉลี่ยประวัติศาสตร์ถ้ายังไม่มี
    if not df_raw.empty and not st.session_state['historical_avg']:
        st.session_state['historical_avg'] = calculate_historical_avg(df_raw)

    # โหลดโมเดลและ Metrics
    xgb_model, lr_model, le_room, le_res, metrics = load_system_models()

    # --- Sidebar Menu ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2933/2933116.png", width=80)
        st.markdown(f"### ผู้ใช้งาน: {st.session_state['username']}")
        
        # เมนูหลัก
        page = st.radio(
            "เมนูใช้งาน:", 
            ["📊 แดชบอร์ด", "📥 จัดการข้อมูล", "🔮 พยากรณ์ราคา", "🧠 วิเคราะห์โมเดล", "ℹ️ เกี่ยวกับระบบ"]
        )
        
        st.divider()
        
        # แสดง Performance ของโมเดลแบบ Real-time ใน Sidebar
        if metrics:
            st.markdown("#### ⚙️ Model Performance")
            st.progress(metrics['xgb']['r2'], text=f"XGBoost: {metrics['xgb']['r2']*100:.1f}%")
            st.progress(metrics['lr']['r2'], text=f"Linear Regression: {metrics['lr']['r2']*100:.1f}%")
        
        st.divider()
        
        # ปุ่ม Logout
        if st.button("ออกจากระบบ (Logout)", use_container_width=True): 
            st.session_state['logged_in'] = False
            st.session_state['username'] = ""
            st.rerun()

    # --- Page Routing Logic ---
    # เรียกใช้ฟังก์ชัน show() จากแต่ละไฟล์ที่เรา Import มา
    try:
        if page == "📊 แดชบอร์ด":
            dashboard_page.show()
            
        elif page == "📥 จัดการข้อมูล":
            manage_page.show(metrics)
            
        elif page == "🔮 พยากรณ์ราคา":
            pricing_page.show(xgb_model, lr_model, le_room, le_res, metrics)
            
        elif page == "🧠 วิเคราะห์โมเดล":
            insight_page.show(metrics)
            
        elif page == "ℹ️ เกี่ยวกับระบบ":
            about_page.show()
            
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดหน้าจอ: {e}")
        st.info("กรุณาตรวจสอบว่าไฟล์หน้าย่อย (_page.py) ทั้งหมดอยู่ในโฟลเดอร์เดียวกัน")