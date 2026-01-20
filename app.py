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
from datetime import datetime

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
    page_title="Hotel Price Forecasting DSS",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_FILE = "users.db"
DATA_FILE = "check_in_report.csv"
ROOM_FILE = "room_type.csv" # ไฟล์ Master Data
METRICS_FILE = "model_metrics.json"

MODEL_FILES = {
    'xgb': 'xgb_hotel_model.joblib',
    'lr': 'lr_hotel_model.joblib',
    'le_room': 'le_room.joblib',
    'le_res': 'le_res.joblib'
}

BASE_PRICES = {
    'Grand Suite Room': 2700,
    'Villa Suite (Garden)': 2700,
    'Executive Room': 2500,
    'Executive Room with Balcony': 2400,
    'Villa Suite (Bathtub)': 2000,
    'Deluxe Room': 1500,
    'Standard Room': 1000
}

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
# 3. BACKEND SYSTEM (Data Cleaning Logic)
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
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            df = df.dropna(subset=['Date']) # ลบแถวที่วันที่เป็น Null
            df['is_weekend'] = df['Date'].dt.weekday.isin([5, 6]).astype(int)
            df['Year'] = df['Date'].dt.year.astype(int) # สร้างคอลัมน์ Year แบบ Int
            df['month'] = df['Date'].dt.month
            
        if 'Room' in df.columns:
            df['Room'] = df['Room'].astype(str)

        # 2. Room Type Mapping
        if os.path.exists(ROOM_FILE):
            room_type = pd.read_csv(ROOM_FILE)
            if 'Room' in room_type.columns: room_type['Room'] = room_type['Room'].astype(str)
            
            if 'Room_Type' in room_type.columns:
                df = df.merge(room_type, on='Room', how='left')
                if 'Room_Type' in df.columns: df = df.rename(columns={'Room_Type': 'Target_Room_Type'})
                elif 'Room_Type_y' in df.columns: df = df.rename(columns={'Room_Type_y': 'Target_Room_Type'})
        
        # 3. Filter Outlier
        df = df.dropna(subset=['Target_Room_Type'])
        
        df['Reservation'] = df['Reservation'].fillna('Unknown')
        
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
    
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, 'r') as f: metrics = json.load(f)
    else: metrics = DEFAULT_METRICS
        
    return xgb, lr, le_room, le_res, metrics

def save_uploaded_data_with_cleaning(uploaded_file):
    try:
        uploaded_file.seek(0)
        new_data = pd.read_csv(uploaded_file)
        
        if 'Room' in new_data.columns: new_data['Room'] = new_data['Room'].astype(str)
        
        valid_rooms = set()
        if os.path.exists(ROOM_FILE):
            room_master = pd.read_csv(ROOM_FILE)
            if 'Room' in room_master.columns:
                valid_rooms = set(room_master['Room'].astype(str))
        
        if len(valid_rooms) > 0:
            good_rows = new_data[new_data['Room'].isin(valid_rooms)]
            bad_rows = new_data[~new_data['Room'].isin(valid_rooms)]
            
            if len(bad_rows) > 0:
                st.warning(f"⚠️ ตรวจพบข้อมูลห้องที่ไม่รู้จัก (Outlier) จำนวน {len(bad_rows)} รายการ")
                st.error(f"รายการที่ถูกตัดทิ้ง (Drop): {bad_rows['Room'].unique()}")
                st.info("ระบบจะบันทึกเฉพาะข้อมูลที่ถูกต้องเท่านั้น")
            else:
                st.success("✅ ข้อมูลถูกต้องสมบูรณ์ 100%")
                
            data_to_save = good_rows
        else:
            st.warning("⚠️ ไม่พบไฟล์ room_type.csv ระบบจะบันทึกข้อมูลทั้งหมดโดยไม่กรอง")
            data_to_save = new_data

        if not data_to_save.empty:
            if os.path.exists(DATA_FILE):
                current_df = pd.read_csv(DATA_FILE)
                if 'Room' in current_df.columns: current_df['Room'] = current_df['Room'].astype(str)
                updated_df = pd.concat([current_df, data_to_save], ignore_index=True)
            else:
                updated_df = data_to_save
                
            updated_df.to_csv(DATA_FILE, index=False)
            st.cache_data.clear()
            return True
        else:
            st.error("❌ ไม่มีข้อมูลที่ถูกต้องให้บันทึก (Outlier ทั้งหมด)")
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
            
        df = df.dropna(subset=['Price', 'Night'])
        
        df['Night'] = df['Night'].fillna(1)
        df['Adults'] = df['Adults'].fillna(2)
        df['Children'] = df['Children'].fillna(0)
        df['Infants'] = df['Infants'].fillna(0)
        df['Extra Person'] = df['Extra Person'].fillna(0)
        
        if not os.path.exists("thai_holidays.csv"):
             try: gdown.download("https://drive.google.com/uc?id=1L-pciKEeRce1gzuhdtpIGcLs0fYHnbZw", "thai_holidays.csv", quiet=True)
             except: pass
        if os.path.exists("thai_holidays.csv"):
            holidays_csv = pd.read_csv("thai_holidays.csv")
            holidays_csv['Holiday_Date'] = pd.to_datetime(holidays_csv['Holiday_Date'], dayfirst=True, errors='coerce')
            df['is_holiday'] = df['Date'].isin(holidays_csv['Holiday_Date']).astype(int)
        else: df['is_holiday'] = 0

        # is_weekend มีอยู่แล้วจาก load_data แต่คำนวณซ้ำเพื่อความชัวร์ใน training logic
        df['is_weekend'] = df['Date'].dt.weekday.isin([5, 6]).astype(int)
        
        df['total_guests'] = df[['Adults', 'Children', 'Infants', 'Extra Person']].sum(axis=1)
        df['month'] = df['Date'].dt.month
        df['weekday'] = df['Date'].dt.weekday
        
        le_room_new = LabelEncoder()
        df['RoomType_encoded'] = le_room_new.fit_transform(df['Target_Room_Type'].astype(str))
        le_res_new = LabelEncoder()
        df['Reservation_encoded'] = le_res_new.fit_transform(df['Reservation'].astype(str))
        
        feature_cols = ['Night', 'total_guests', 'is_holiday', 'is_weekend', 'month', 'weekday', 'RoomType_encoded', 'Reservation_encoded']
        X = df[feature_cols]
        y = df['Price']
        
        X = X.fillna(0)
        
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
    df_raw = load_data() # Load ข้อมูลดิบก่อนกรอง
    xgb_model, lr_model, le_room, le_res, metrics = load_system_models()
    if not os.path.exists("thai_holidays.csv"):
        try: gdown.download("https://drive.google.com/uc?id=1L-pciKEeRce1gzuhdtpIGcLs0fYHnbZw", "thai_holidays.csv", quiet=True)
        except: pass

    def show_dashboard_page():
        st.title("📊 Financial Executive Dashboard")
        
        if df_raw.empty: st.warning("No Data Found"); return

        # --- FILTER SECTION ---
        with st.expander("🔎 Filter Data (ตัวกรองข้อมูล)", expanded=True):
            f_col1, f_col2, f_col3 = st.columns(3)
            
            # 1. Year Filter
            all_years = sorted(df_raw['Year'].unique().tolist())
            year_opts = ['All'] + [str(y) for y in all_years]
            with f_col1:
                sel_year = st.selectbox("📅 Select Year (เลือกปี)", year_opts)
            
            # 2. Month Filter
            all_months = sorted(df_raw['month'].unique().tolist())
            month_opts = ['All'] + [datetime(2024, m, 1).strftime('%B') for m in all_months]
            with f_col2:
                sel_month_str = st.selectbox("🗓️ Select Month (เลือกเดือน)", month_opts)

            # Logic กรองข้อมูล
            df_filtered = df_raw.copy()
            
            if sel_year != 'All':
                df_filtered = df_filtered[df_filtered['Year'] == int(sel_year)]
            
            if sel_month_str != 'All':
                # แปลงชื่อเดือนกลับเป็นตัวเลข
                sel_month_num = datetime.strptime(sel_month_str, "%B").month
                df_filtered = df_filtered[df_filtered['month'] == sel_month_num]

        if df_filtered.empty:
            st.warning("⚠️ No data available for the selected filters.")
            return

        st.divider()
        
        # --- TOP METRICS ---
        k1, k2, k3 = st.columns(3)
        with k1: st.metric("💰 Total Revenue", f"{df_filtered['Price'].sum()/1e6:.2f} M THB")
        with k2: st.metric("📦 Total Bookings", f"{len(df_filtered):,} รายการ")
        with k3: st.metric("🏷️ Avg. Booking Value", f"{df_filtered['Price'].mean():,.0f} THB")
        
        st.divider()

        # --- TABS FOR CHARTS ---
        tab1, tab2, tab3 = st.tabs([
            "💰 Financial Overview", 
            "📢 Channel Strategy", 
            "🛌 Product & Behavior"
        ])

        group_col = 'Target_Room_Type' if 'Target_Room_Type' in df_filtered.columns else 'Room'

        # === TAB 1: FINANCIAL ===
        with tab1:
            st.markdown("### 1. Financial Overview (ภาพรวมการเงิน)")
            
            # ROW 1.1
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
                # Group by Month name for display
                monthly = df_filtered.groupby('month').agg({'Price': 'sum', 'Room': 'count'}).reset_index().sort_values('month')
                monthly['M_Name'] = monthly['month'].apply(lambda x: datetime(2024, int(x), 1).strftime('%b'))
                
                fig2 = make_subplots(specs=[[{"secondary_y": True}]])
                fig2.add_trace(go.Scatter(x=monthly['M_Name'], y=monthly['Price'], name="Revenue", line=dict(color='green', width=3)), secondary_y=False)
                fig2.add_trace(go.Scatter(x=monthly['M_Name'], y=monthly['Room'], name="Bookings", line=dict(color='blue', dash='dot')), secondary_y=True)
                fig2.update_layout(legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig2, use_container_width=True)
            
            # ROW 1.2: ADR
            st.subheader("ADR Trend Analysis (Average Daily Rate)")
            monthly_adr = df_filtered.groupby('month').apply(lambda x: x['Price'].sum() / x['Night'].sum()).reset_index(name='ADR')
            monthly_adr['M_Name'] = monthly_adr['month'].apply(lambda x: datetime(2024, int(x), 1).strftime('%b'))
            fig_adr = px.line(monthly_adr, x='M_Name', y='ADR', markers=True, title="ADR per Month")
            st.plotly_chart(fig_adr, use_container_width=True)

        # === TAB 2: CHANNEL ===
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

        # === TAB 3: PRODUCT & BEHAVIOR ===
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
            with c7:
                st.plotly_chart(px.pie(day_rev, values='Price', names='DayType', hole=0.4, title="Revenue Share"), use_container_width=True)
            with c8:
                day_avg = df_filtered.groupby('DayType')['Price'].mean().reset_index()
                st.plotly_chart(px.bar(day_avg, x='DayType', y='Price', title="Avg Booking Value", color='DayType'), use_container_width=True)

        st.divider()
        st.subheader("📋 Raw Data Explorer")
        with st.expander("คลิกเพื่อดูตารางข้อมูลที่ผ่านการกรองแล้ว"):
            st.dataframe(df_filtered)

    def show_manage_data_page():
        st.title("📥 จัดการข้อมูล & อัปเดตโมเดล")
        st.markdown("### 1. นำเข้าข้อมูล & ตรวจสอบความถูกต้อง")
        st.info("ระบบจะตรวจสอบเลขห้องกับไฟล์ Master Data หากพบข้อมูล Outlier จะทำการลบทิ้งอัตโนมัติ")
        
        up_file = st.file_uploader("เลือกไฟล์ Booking CSV (เพื่อเพิ่มข้อมูล)", type=['csv'])
        if up_file is not None:
            if st.button("💾 บันทึกข้อมูลเข้าระบบ", type="primary"):
                if save_uploaded_data_with_cleaning(up_file):
                    st.success("✅ บันทึกข้อมูลเรียบร้อย! ข้อมูลถูกบันทึกลงระบบแล้ว")
                    st.balloons()
                    time.sleep(5) 
                    st.rerun()
        
        st.divider()
        
        st.markdown("### 2. สั่งให้โมเดลเรียนรู้ (Retrain)")
        st.warning("⚠️ กดปุ่มนี้เมื่อมีการเพิ่มข้อมูลใหม่ เพื่อให้ AI ฉลาดขึ้น")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1: st.metric("Current Accuracy (R²)", f"{metrics['xgb']['r2']*100:.2f}%")
        
        if st.button("🚀 เริ่มกระบวนการเรียนรู้ใหม่ (Start Retraining)", type="secondary"):
            success, count = retrain_system()
            if success:
                st.success(f"🎉 โมเดลเรียนรู้ครบ {count:,} รายการ! ระบบพร้อมใช้งานข้อมูลใหม่แล้ว")
                time.sleep(5)
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
                
                base_price = 0
                for key in BASE_PRICES:
                    if key in room: base_price = BASE_PRICES[key]; break
                
                st.divider()
                c_base, c_xgb, c_lr = st.columns(3)
                with c_base:
                    st.info("### 🏷️ Base Price")
                    st.metric("ราคาตั้งต้น", f"{base_price:,.0f} THB")
                with c_xgb:
                    st.success("### ⚡ XGBoost") 
                    st.metric("ราคาแนะนำ", f"{p_xgb:,.0f} THB", delta=f"{p_xgb - base_price:,.0f} THB")
                    st.caption(f"MAE: ±{metrics['xgb']['mae']:,.0f} | R²: {metrics['xgb']['r2']:.4f}")
                with c_lr:
                    st.warning("### 📉 Linear Regression") 
                    st.metric("ราคาประเมิน", f"{p_lr:,.0f} THB", delta=f"{p_lr - base_price:,.0f} THB")
                    st.caption(f"MAE: ±{metrics['lr']['mae']:,.0f} | R²: {metrics['lr']['r2']:.4f}")

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
        
        # ปรับเมนู: ลบ "หน้าหลัก" ออก และให้ "แดชบอร์ด" เป็นตัวเลือกแรก
        page = st.radio("เมนูใช้งาน:", ["📊 แดชบอร์ด", "📥 จัดการข้อมูล", "🔮 พยากรณ์ราคา", "🧠 วิเคราะห์โมเดล", "ℹ️ เกี่ยวกับระบบ"])
        
        st.divider()
        st.markdown("#### ⚙️ Real-time Performance")
        st.progress(metrics['xgb']['r2'], text=f"XGBoost: {metrics['xgb']['r2']*100:.1f}%")
        st.progress(metrics['lr']['r2'], text=f"Linear Regression: {metrics['lr']['r2']*100:.1f}%")
        st.divider()
        if st.button("Logout"): st.session_state['logged_in'] = False; st.rerun()

    # Routing หน้าเว็บ (Default คือ Dashboard)
    if "แดชบอร์ด" in page: show_dashboard_page()
    elif "จัดการข้อมูล" in page: show_manage_data_page()
    elif "พยากรณ์ราคา" in page: show_pricing_page()
    elif "วิเคราะห์โมเดล" in page: show_model_insight_page()
    elif "เกี่ยวกับระบบ" in page: show_about_page()