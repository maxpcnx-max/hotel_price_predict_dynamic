import streamlit as st
import pandas as pd
import time
from datetime import datetime
import utils
import pricing_engine
import ui_components

st.set_page_config(page_title="Hotel Price Forecasting", page_icon="🏨", layout="wide")

# Session States
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'historical_avg' not in st.session_state: st.session_state['historical_avg'] = {}

utils.init_db()

def login_page():
    st.title("🔒 Login System")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login", type="primary"):
        if utils.login_user(u, p):
            st.session_state['logged_in'] = True
            st.rerun()
        else: st.error("Invalid Credentials")

if not st.session_state['logged_in']:
    login_page()
else:
    df_raw = utils.load_data()
    if not df_raw.empty and not st.session_state['historical_avg']:
        # คำนวณค่าเฉลี่ยเก็บไว้ใน Session
        df_c = df_raw[df_raw['Night'] > 0].copy()
        df_c['ADR'] = df_c['Price'] / df_c['Night']
        st.session_state['historical_avg'] = df_c.groupby('Target_Room_Type')['ADR'].mean().to_dict()

    xgb, lr, le_room, le_res, metrics = utils.load_system_models()

    with st.sidebar:
        st.header("Menu")
        page = st.radio("เลือกหน้า:", ["📊 แดชบอร์ด", "🔮 พยากรณ์ราคา", "ℹ️ เกี่ยวกับระบบ"])
        if st.button("Logout"):
            st.session_state['logged_in'] = False
            st.rerun()

    if "แดชบอร์ด" in page:
        ui_components.show_dashboard_page(df_raw)
    
    elif "พยากรณ์ราคา" in page:
        st.title("🔮 Forecasting")
        # ยก Logic การทำ UI ของหน้าพยากรณ์มาไว้ที่นี่ หรือแยกไปอีกไฟล์ก็ได้ครับ
        room_list = list(le_room.classes_) if le_room else []
        selected_room = st.selectbox("เลือกห้อง", room_list)
        if st.button("คำนวณ"):
            # ตัวอย่างการเรียกใช้ engine
            # price, _, _ = pricing_engine.calculate_clamped_price(...)
            st.success("กำลังคำนวณ...")

    elif "เกี่ยวกับระบบ" in page:
        st.title("ℹ️ About")
        st.write("ระบบพัฒนาโดย ว่าที่ร้อยตรีพรพินิต วิรัตน์สกุลชัย")