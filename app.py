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
    page_title="Hotel Price Forecasting System (Full CRUD)",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_FILE = "users.db"
DATA_FILE = "check_in_report.csv"
ROOM_FILE = "room_type.csv" # ไฟล์ Master Data
METRICS_FILE = "model_metrics.json"
BASE_PRICE_FILE = "base_prices.json" # ไฟล์เก็บราคาฐาน

MODEL_FILES = {
    'xgb': 'xgb_hotel_model.joblib',
    'lr': 'lr_hotel_model.joblib',
    'le_room': 'le_room.joblib',
    'le_res': 'le_res.joblib'
}

# --- ค่า Default หากไม่มีไฟล์ JSON ---
DEFAULT_BASE_PRICES = {
    'Grand Suite Room': 2700,
    'Villa Suite (Garden)': 2700,
    'Executive Room': 2500,
    'Executive Room with Balcony': 2400,
    'Villa Suite (Bathtub)': 2000,
    'Deluxe Room': 1500,
    'Standard Room': 1000
}

# --- ฟังก์ชันจัดการ Base Prices ---
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
# 2. DATABASE
# ==========================================================
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)')
    # เช็คว่ามี admin หรือยัง ถ้าไม่มีให้สร้าง
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
# 3. BACKEND SYSTEM
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
            # ใช้ errors='coerce' แต่ห้าม dropna ทันที เพื่อรักษาข้อมูล Raw
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            
            # คำนวณเฉพาะแถวที่วันที่ถูกต้อง (เพื่อไม่ให้ Error แต่ข้อมูลไม่หาย)
            mask_valid = df['Date'].notna()
            df.loc[mask_valid, 'is_weekend'] = df.loc[mask_valid, 'Date'].dt.weekday.isin([5, 6]).astype(int)
            df.loc[mask_valid, 'Year'] = df.loc[mask_valid, 'Date'].dt.year.astype(int)
            df.loc[mask_valid, 'month'] = df.loc[mask_valid, 'Date'].dt.month
            df.loc[mask_valid, 'weekday'] = df.loc[mask_valid, 'Date'].dt.weekday
            
        if 'Room' in df.columns:
            df['Room'] = df['Room'].astype(str)

        # 2. Room Type Mapping (ถ้ามี Master File)
        if os.path.exists(ROOM_FILE):
            try:
                room_type = pd.read_csv(ROOM_FILE)
                if 'Room' in room_type.columns: room_type['Room'] = room_type['Room'].astype(str)
                
                if 'Target_Room_Type' in room_type.columns:
                    df = df.merge(room_type[['Room', 'Target_Room_Type']], on='Room', how='left')
                elif 'Room_Type' in room_type.columns:
                    room_type = room_type.rename(columns={'Room_Type': 'Target_Room_Type'})
                    df = df.merge(room_type[['Room', 'Target_Room_Type']], on='Room', how='left')
            except:
                pass
        
        # 3. Handle Unknown Rooms (ใช้ชื่อเดิมถ้าหาไม่เจอ)
        if 'Target_Room_Type' in df.columns:
            df['Target_Room_Type'] = df['Target_Room_Type'].fillna(df['Room'])
        else:
            df['Target_Room_Type'] = df['Room']
        
        df['Reservation'] = df['Reservation'].fillna('Unknown')
        
        # ลบคอลัมน์ที่ซ้ำ
        df = df.loc[:, ~df.columns.duplicated()]
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
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

def save_uploaded_data_with_cleaning(uploaded_file):
    try:
        uploaded_file.seek(0)
        new_data = pd.read_csv(uploaded_file)
        
        if 'Room' in new_data.columns: new_data['Room'] = new_data['Room'].astype(str)
        
        # รับข้อมูลทั้งหมด ไม่ตัด
        data_to_save = new_data

        if not data_to_save.empty:
            if os.path.exists(DATA_FILE):
                current_df = pd.read_csv(DATA_FILE)
                if 'Room' in current_df.columns: current_df['Room'] = current_df['Room'].astype(str)
                
                # ลบคอลัมน์คำนวณก่อน Merge
                cols_to_drop = ['Year', 'month', 'is_weekend', 'weekday', 'Target_Room_Type']
                current_df = current_df.drop(columns=[c for c in cols_to_drop if c in current_df.columns], errors='ignore')
                
                updated_df = pd.concat([current_df, data_to_save], ignore_index=True)
            else:
                updated_df = data_to_save
                
            updated_df.to_csv(DATA_FILE, index=False)
            st.cache_data.clear()
            return True
        else:
            st.error("❌ ไฟล์ไม่มีข้อมูล")
            return False

    except Exception as e:
        st.error(f"Save failed: {e}")
        return False

def retrain_system():
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        status_text.text("⏳ Reading & Cleaning data...")
        df = load_data() 
        
        if df.empty:
            st.error("ไม่พบข้อมูลสำหรับเทรนโมเดล")
            return False, 0
        
        # กรองเฉพาะข้อมูลที่พร้อมสำหรับ Train (ตรงนี้กรองได้ เพราะ Train ต้องใช้ข้อมูลดี)
        df = df.dropna(subset=['Price', 'Night', 'Date'])
        
        df['Night'] = df['Night'].fillna(1)
        df['Adults'] = df['Adults'].fillna(2)
        
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
        
        # Re-initialize Encoders
        le_room_new = LabelEncoder()
        df['RoomType_encoded'] = le_room_new.fit_transform(df['Target_Room_Type'].astype(str))
        le_res_new = LabelEncoder()
        df['Reservation_encoded'] = le_res_new.fit_transform(df['Reservation'].astype(str))
        
        feature_cols = ['Night', 'total_guests', 'is_holiday', 'is_weekend', 'month', 'weekday', 'RoomType_encoded', 'Reservation_encoded']
        X = df[feature_cols]
        X = X.fillna(0)
        y = df['Price']
        
        progress_bar.progress(40)
        status_text.text("🏋️‍♂️ Training new models...")
        
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
            
        st.cache_resource.clear()
        progress_bar.progress(100)
        status_text.success(f"✅ Retraining Complete! New R²: {new_xgb_r2:.4f}")
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
    # โหลด Data มาก่อน (แบบไม่ตัดข้อมูล)
    df_raw = load_data() 
    xgb_model, lr_model, le_room, le_res, metrics = load_system_models()
    
    def show_dashboard_page():
        st.title("📊 Financial Executive Dashboard")
        if df_raw.empty: st.warning("No Data Found"); return

        # 🔥 Fix: กรองเฉพาะข้อมูลที่มีวันที่สมบูรณ์สำหรับแสดงกราฟ (แต่ไม่ลบจากไฟล์จริง)
        df_filtered = df_raw.dropna(subset=['Date']).copy()

        with st.expander("🔎 Filter Data (ตัวกรองข้อมูล)", expanded=True):
            f_col1, f_col2, f_col3 = st.columns(3)
            all_years = sorted(df_filtered['Year'].unique().tolist())
            year_opts = ['All'] + [str(y) for y in all_years]
            with f_col1: sel_year = st.selectbox("📅 Select Year (เลือกปี)", year_opts)
            
            all_months = sorted(df_filtered['month'].unique().tolist())
            month_opts = ['All'] + [datetime(2024, m, 1).strftime('%B') for m in all_months]
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
        st.subheader("📋 Raw Data Explorer")
        with st.expander("คลิกเพื่อดูตารางข้อมูลที่ผ่านการกรองแล้ว"): st.dataframe(df_filtered)

    # ==========================================================
    # 🌟 MANAGE DATA PAGE (FIXED: No Filter & No Duplicate Key)
    # ==========================================================
    def show_manage_data_page():
        st.title("📥 ระบบจัดการฐานข้อมูล (Data Management)")
        
        tab_trans, tab_master, tab_train = st.tabs(["📝 แก้ไขข้อมูลการจอง", "⚙️ ตั้งค่าห้องพัก/ราคาฐาน", "🚀 อัปเดตโมเดล"])

        # ---------------------------------------------------------
        # TAB 1: EDIT TRANSACTIONS
        # ---------------------------------------------------------
        with tab_trans:
            st.subheader("1. นำเข้าข้อมูลใหม่ (Import Data)")
            up_file = st.file_uploader("เลือกไฟล์ CSV", type=['csv'])
            if up_file is not None:
                if st.button("เทข้อมูลเข้าระบบ (Dump Merge)", type="secondary"):
                    if save_uploaded_data_with_cleaning(up_file):
                        st.success("บันทึกเรียบร้อย!"); time.sleep(1); st.rerun()

            st.divider()
            st.subheader("2. ตรวจสอบและแก้ไขข้อมูล")
            st.info("💡 ข้อมูลทุกแถวจะแสดงที่นี่ (รวมถึงแถวที่วันที่ไม่สมบูรณ์) เพื่อให้คุณแก้ไขได้")
            
            # โหลดข้อมูล (load_data ตัวใหม่ที่แก้แล้ว)
            df_current = load_data()
            
            if not df_current.empty:
                edited_df = st.data_editor(
                    df_current,
                    num_rows="dynamic",
                    use_container_width=True,
                    key="booking_editor_unique", # 🔥 Unique Key แก้บั๊ก Duplicate
                    column_config={
                        "Date": st.column_config.DateColumn("Check-in Date", format="DD/MM/YYYY"),
                        "Price": st.column_config.NumberColumn("Price (THB)", format="%d"),
                    }
                )
                
                col_save, col_del = st.columns([1, 4])
                
                # 🔥 BUTTON SAVE - แบบไม่ฆ่าข้อมูล
                with col_save:
                    if st.button("💾 บันทึกการเปลี่ยนแปลง (Save)", type="primary"):
                        try:
                            # 1. รับค่าจากตารางตรงๆ (Raw)
                            df_to_save = edited_df.copy()

                            # 2. Handle Date Format (จัด Format แต่ห้าม Dropna)
                            if 'Date' in df_to_save.columns:
                                df_to_save['Date'] = pd.to_datetime(df_to_save['Date'], dayfirst=True, errors='ignore')

                            # 3. เลือก Column ที่จะเซฟ
                            cols_to_save = ['Date', 'Room', 'Price', 'Reservation', 'Name', 
                                            'Night', 'Adults', 'Children', 'Infants', 'Extra Person']
                            final_cols = [c for c in cols_to_save if c in df_to_save.columns]
                            
                            if final_cols:
                                df_to_save = df_to_save[final_cols]
                                df_to_save.to_csv(DATA_FILE, index=False)
                                st.cache_data.clear()
                                st.success(f"✅ บันทึกสำเร็จ ({len(df_to_save)} รายการ)")
                                time.sleep(1); st.rerun()
                            else:
                                st.error("ไม่พบคอลัมน์ที่ถูกต้อง")
                        except Exception as e:
                            st.error(f"Save Error: {e}")

                with col_del:
                    if st.button("🧨 ล้างข้อมูลทั้งหมด"):
                         if os.path.exists(DATA_FILE):
                            os.remove(DATA_FILE)
                            st.cache_data.clear()
                            st.rerun()

        # ---------------------------------------------------------
        # TAB 2: MASTER DATA
        # ---------------------------------------------------------
        with tab_master:
            st.subheader("⚙️ กำหนดราคาฐาน (Base Prices)")
            current_prices = load_base_prices()
            df_prices = pd.DataFrame(list(current_prices.items()), columns=['Room Type', 'Base Price'])
            
            edited_prices_df = st.data_editor(
                df_prices,
                num_rows="dynamic",
                use_container_width=True,
                key="price_editor_unique", # 🔥 Unique Key
                column_config={
                    "Base Price": st.column_config.NumberColumn("Price", format="%d THB")
                }
            )
            
            if st.button("💾 บันทึกราคาฐาน"):
                new_prices = {}
                for _, row in edited_prices_df.iterrows():
                    if row['Room Type']: new_prices[row['Room Type']] = row['Base Price']
                save_base_prices(new_prices)
                st.success("✅ อัปเดตราคาฐานเรียบร้อย!")

        # ---------------------------------------------------------
        # TAB 3: RETRAIN
        # ---------------------------------------------------------
        with tab_train:
            st.subheader("🧠 Retrain Model")
            r2_val = metrics['xgb']['r2'] if 'xgb' in metrics else 0
            st.write(f"Current R2: {r2_val:.4f}")
            
            if st.button("🚀 Start Retraining", type="primary"):
                success, count = retrain_system()
                if success: st.success(f"Done! Trained on {count} rows."); time.sleep(1); st.rerun()

    def show_pricing_page():
        st.title("🔮 ระบบพยากรณ์ราคา (Price Forecasting)")
        if xgb_model is None: st.error("❌ Model not found"); return

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
                date_range = st.date_input("Select Dates (Check-in - Check-out)", value=[], min_value=None)
            
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
                if selected_room_val != "All":
                    if "Standard" in str(selected_room_val) or "Deluxe" in str(selected_room_val): max_g = 2
                guests = st.number_input(f"Guests (Max {max_g})", min_value=1, max_value=max_g, value=min(2, max_g))

            with c5:
                res_options = ["All (เลือกทั้งหมด)"] + list(le_res.classes_)
                selected_res = st.selectbox("Channel", res_options)
                selected_res_val = "All" if "All" in selected_res else selected_res

            if st.button("🚀 คำนวณราคา (Predict)", type="primary", use_container_width=True):
                if selected_room_val == "All" or selected_res_val == "All":
                    st.info(f"📊 รายงานผลการพยากรณ์รวม (Batch Report)")
                    
                    target_rooms = list(prices.keys()) if selected_room_val == "All" else [selected_room_val]
                    target_res = le_res.classes_ if selected_res_val == "All" else [selected_res_val]
                    
                    results = []
                    for r_type in target_rooms:
                        try: r_code = le_room.transform([r_type])[0]
                        except: continue 
                        
                        base_per_night = get_base_price_safe(r_type)
                        total_base_price = base_per_night * nights
                        
                        for ch_type in target_res:
                            try: res_code = le_res.transform([ch_type])[0]
                            except: continue
                            
                            inp = pd.DataFrame([{
                                'Night': nights, 'total_guests': guests, 
                                'is_holiday': final_is_holiday, 'is_weekend': 1 if checkin_date.weekday() in [5,6] else 0,
                                'month': checkin_date.month, 'weekday': checkin_date.weekday(),
                                'RoomType_encoded': r_code, 'Reservation_encoded': res_code
                            }])
                            
                            raw_xgb = xgb_model.predict(inp)[0]
                            final_xgb = max(raw_xgb, total_base_price)
                            
                            raw_lr = lr_model.predict(inp)[0]
                            final_lr = max(raw_lr, total_base_price)
                            
                            results.append({
                                "Room": r_type, "Channel": ch_type, "Guests": guests,
                                "Floor Price (Base*N)": total_base_price, 
                                "XGB Price": final_xgb, "LR Price": final_lr
                            })
                    
                    if results:
                        st.dataframe(pd.DataFrame(results).style.format("{:,.0f}", subset=["Floor Price (Base*N)", "XGB Price", "LR Price"]), use_container_width=True, height=500)
                    else:
                        st.warning("ไม่พบข้อมูลพยากรณ์ (กรุณา Retrain โมเดลหากเพิ่มห้องใหม่)")

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
                        
                        p_xgb_norm, raw_xgb, floor_p = calculate_clamped_price(xgb_model, inp_norm, selected_room_val, nights)
                        p_lr_norm, raw_lr, _ = calculate_clamped_price(lr_model, inp_norm, selected_room_val, nights)
                        
                        st.divider()
                        st.markdown(f"### 🏨 ผลการวิเคราะห์ราคาห้อง: **{selected_room_val}**")
                        st.caption(f"เงื่อนไข: {nights} คืน | {guests} ท่าน | ช่องทาง {selected_res_val} | Floor Price (Base x Night): {floor_p:,.0f} THB")
                        
                        r1c1, r1c2 = st.columns(2)
                        with r1c1:
                            diff_xgb = p_xgb_norm - floor_p
                            delta_color = "normal"
                            note = ""
                            if raw_xgb < floor_p:
                                note = f"⚠️ Adjusted from {raw_xgb:,.0f}"
                                delta_color = "off"
                                
                            st.container(border=True).metric(
                                label=f"⚡ XGBoost (ปกติ: {guests} ท่าน)",
                                value=f"{p_xgb_norm:,.0f} THB",
                                delta=f"{diff_xgb:+,.0f} THB (vs Floor)",
                                delta_color=delta_color
                            )
                            st.caption(f"MAE: ±{metrics['xgb']['mae']:,.0f} | R²: {metrics['xgb']['r2']*100:.2f}% {note}")
                        
                        with r1c2:
                            diff_lr = p_lr_norm - floor_p
                            note_lr = ""
                            if raw_lr < floor_p: note_lr = f"⚠️ Adjusted from {raw_lr:,.0f}"
                            
                            st.container(border=True).metric(
                                label=f"📉 Linear Regression (ปกติ: {guests} ท่าน)",
                                value=f"{p_lr_norm:,.0f} THB",
                                delta=f"{diff_lr:+,.0f} THB (vs Floor)"
                            )
                            st.caption(f"MAE: ±{metrics['lr']['mae']:,.0f} | R²: {metrics['lr']['r2']*100:.2f}% {note_lr}")

                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาดในการคำนวณ: {e} (กรุณา Retrain โมเดล)")

    def show_model_insight_page():
        st.title("🧠 วิเคราะห์ปัจจัยโมเดล (Dynamic Insight)")
        imp_data = metrics.get('importance', DEFAULT_METRICS['importance'])
        fi_df = pd.DataFrame(list(imp_data.items()), columns=['Feature', 'Importance']).sort_values('Importance', ascending=True)
        st.divider()
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("📊 กราฟความสำคัญ")
            st.plotly_chart(px.bar(fi_df, x='Importance', y='Feature', orientation='h', text_auto='.4f', color='Importance', color_continuous_scale='Blues'), use_container_width=True)
        with c2:
            st.subheader("📋 ตารางข้อมูล")
            st.dataframe(fi_df.sort_values('Importance', ascending=False), use_container_width=True, height=400)

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

    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2933/2933116.png", width=80)
        st.markdown(f"### User: {st.session_state['username']}")
        page = st.radio("เมนูใช้งาน:", ["📊 แดชบอร์ด", "📥 จัดการข้อมูล", "🔮 พยากรณ์ราคา", "🧠 วิเคราะห์โมเดล", "ℹ️ เกี่ยวกับระบบ"])
        st.divider()
        st.markdown("#### ⚙️ Real-time Performance")
        if 'xgb' in metrics:
            st.progress(metrics['xgb']['r2'], text=f"XGBoost: {metrics['xgb']['r2']*100:.1f}%")
        if 'lr' in metrics:
            st.progress(metrics['lr']['r2'], text=f"Linear Regression: {metrics['lr']['r2']*100:.1f}%")
        st.divider()
        if st.button("Logout"): st.session_state['logged_in'] = False; st.rerun()

    if "แดชบอร์ด" in page: show_dashboard_page()
    elif "จัดการข้อมูล" in page: show_manage_data_page()
    elif "พยากรณ์ราคา" in page: show_pricing_page()
    elif "วิเคราะห์โมเดล" in page: show_model_insight_page()
    elif "เกี่ยวกับระบบ" in page: show_about_page()
