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
    page_title="Hotel Price Forecasting System",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_FILE = "users.db"
DATA_FILE = "check_in_report.csv"
ROOM_MAPPING_FILE = "room_type.csv" # ไฟล์จับคู่เลขห้อง (Mapping)
ROOM_CONFIG_FILE = "room_config.csv" # ไฟล์ตั้งค่าราคาและกฎของห้อง (Master Data)
METRICS_FILE = "model_metrics.json"

MODEL_FILES = {
    'xgb': 'xgb_hotel_model.joblib',
    'lr': 'lr_hotel_model.joblib',
    'le_room': 'le_room.joblib',
    'le_res': 'le_res.joblib'
}

# ค่า Default กรณีสร้างไฟล์ครั้งแรก
DEFAULT_ROOM_CONFIG = [
    {"Room_Type": "Grand Suite Room", "Base_Price": 2700, "Allow_Extra": True},
    {"Room_Type": "Villa Suite (Garden)", "Base_Price": 2700, "Allow_Extra": True},
    {"Room_Type": "Executive Room", "Base_Price": 2500, "Allow_Extra": True},
    {"Room_Type": "Executive Room with Balcony", "Base_Price": 2400, "Allow_Extra": True},
    {"Room_Type": "Villa Suite (Bathtub)", "Base_Price": 2000, "Allow_Extra": True},
    {"Room_Type": "Deluxe Room", "Base_Price": 1500, "Allow_Extra": True},
    {"Room_Type": "Standard Room", "Base_Price": 1000, "Allow_Extra": False}
]

DEFAULT_METRICS = {
    'xgb': {'mae': 0, 'r2': 0},
    'lr':  {'mae': 0, 'r2': 0},
    'importance': {}
}

if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'username' not in st.session_state: st.session_state['username'] = ""

# ==========================================================
# 2. DATABASE & UTILS
# ==========================================================
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)')
    
    # สร้าง Admin
    c.execute('SELECT * FROM users WHERE username = "admin"')
    if not c.fetchone():
        c.execute('INSERT INTO users VALUES (?,?)', ("admin", "1234"))
        
    # สร้าง User (Role ทั่วไป)
    c.execute('SELECT * FROM users WHERE username = "user"')
    if not c.fetchone():
        c.execute('INSERT INTO users VALUES (?,?)', ("user", "1234"))
        
    conn.commit()
    conn.close()

def login_user(username, password):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
    data = c.fetchone()
    conn.close()
    return data

def load_room_config():
    if not os.path.exists(ROOM_CONFIG_FILE):
        df = pd.DataFrame(DEFAULT_ROOM_CONFIG)
        df.to_csv(ROOM_CONFIG_FILE, index=False)
        return df
    return pd.read_csv(ROOM_CONFIG_FILE)

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
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            df['is_weekend'] = df['Date'].dt.weekday.isin([5, 6]).astype(int)
            df['year'] = df['Date'].dt.year
            df['month'] = df['Date'].dt.month
            
        if 'Room' in df.columns:
            df['Room'] = df['Room'].astype(str)

        if os.path.exists(ROOM_MAPPING_FILE):
            room_type = pd.read_csv(ROOM_MAPPING_FILE)
            if 'Room' in room_type.columns: room_type['Room'] = room_type['Room'].astype(str)
            if 'Room_Type' in room_type.columns:
                df = df.merge(room_type, on='Room', how='left')
                if 'Room_Type' in df.columns: df = df.rename(columns={'Room_Type': 'Target_Room_Type'})
                elif 'Room_Type_y' in df.columns: df = df.rename(columns={'Room_Type_y': 'Target_Room_Type'})
        
        df = df.dropna(subset=['Target_Room_Type'])
        df['Reservation'] = df['Reservation'].fillna('Unknown')
        
        return df
    except: return pd.DataFrame()

@st.cache_resource
def load_system_models():
    for name, file in MODEL_FILES.items():
        if not os.path.exists(file): return None, None, None, None, None
    return joblib.load(MODEL_FILES['xgb']), joblib.load(MODEL_FILES['lr']), joblib.load(MODEL_FILES['le_room']), joblib.load(MODEL_FILES['le_res']), json.load(open(METRICS_FILE)) if os.path.exists(METRICS_FILE) else DEFAULT_METRICS

# ฟังก์ชันคำนวณราคาพร้อม Business Logic
def calculate_price_logic(model_price, base_price, nights, is_holiday, is_weekend):
    multiplier = 1.0
    
    if is_holiday:
        multiplier = max(multiplier, 1.5)
    if is_weekend:
        multiplier = max(multiplier, 1.2)
        
    floor_price_per_night = base_price * multiplier
    total_floor_price = floor_price_per_night * nights
    
    final_price = max(model_price, total_floor_price)
    final_price = max(final_price, base_price * nights * 0.5) 
    
    return final_price, multiplier

def retrain_system():
    try:
        df = load_data() 
        if df.empty: return False, 0
        df = df.dropna(subset=['Price', 'Night'])
        
        # Fill NA
        df['Night'] = df['Night'].fillna(1)
        df['Adults'] = df['Adults'].fillna(2)
        df[['Children', 'Infants', 'Extra Person']] = df[['Children', 'Infants', 'Extra Person']].fillna(0)
        
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
        
        le_room_new = LabelEncoder()
        df['RoomType_encoded'] = le_room_new.fit_transform(df['Target_Room_Type'].astype(str))
        le_res_new = LabelEncoder()
        df['Reservation_encoded'] = le_res_new.fit_transform(df['Reservation'].astype(str))
        
        feature_cols = ['Night', 'total_guests', 'is_holiday', 'is_weekend', 'month', 'weekday', 'RoomType_encoded', 'Reservation_encoded']
        X = df[feature_cols].fillna(0)
        y = df['Price']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        xgb_new = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        xgb_new.fit(X_train, y_train)
        pred_xgb = xgb_new.predict(X_test)
        
        fi_raw = xgb_new.feature_importances_
        col_mapping = {'Night': 'Night', 'total_guests': 'Guests', 'is_holiday': 'Is Holiday', 'is_weekend': 'Is Weekend', 'month': 'Month', 'weekday': 'Weekday', 'RoomType_encoded': 'Room Type', 'Reservation_encoded': 'Reservation'}
        new_importance = {col_mapping.get(col, col): float(val) for col, val in zip(feature_cols, fi_raw)}

        lr_new = LinearRegression()
        lr_new.fit(X_train, y_train)
        pred_lr = lr_new.predict(X_test)
        
        joblib.dump(xgb_new, MODEL_FILES['xgb'])
        joblib.dump(lr_new, MODEL_FILES['lr'])
        joblib.dump(le_room_new, MODEL_FILES['le_room'])
        joblib.dump(le_res_new, MODEL_FILES['le_res'])
        
        new_metrics = {
            'xgb': {'mae': mean_absolute_error(y_test, pred_xgb), 'r2': r2_score(y_test, pred_xgb)},
            'lr':  {'mae': mean_absolute_error(y_test, pred_lr), 'r2': r2_score(y_test, pred_lr)},
            'importance': new_importance
        }
        with open(METRICS_FILE, 'w') as f: json.dump(new_metrics, f)
        st.cache_resource.clear()
        return True, len(df)
    except Exception as e: return False, 0

# ==========================================================
# 4. UI PAGES
# ==========================================================

def login_page():
    st.markdown("""<style>.stTextInput > div > div > input {text-align: center;}</style>""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/2933/2933116.png", width=120)
        st.title("🔒 Login System")
        u = st.text_input("Username"); p = st.text_input("Password", type="password")
        if st.button("Login", type="primary", use_container_width=True):
            if login_user(u, p): st.session_state['logged_in'] = True; st.session_state['username'] = u; st.rerun()
            else: st.error("Username หรือ Password ไม่ถูกต้อง")

if not st.session_state['logged_in']:
    login_page()
else:
    df = load_data() 
    xgb_model, lr_model, le_room, le_res, metrics = load_system_models()
    room_config_df = load_room_config()
    
    # Pre-load holidays
    th_holidays = []
    if os.path.exists("thai_holidays.csv"):
        try:
            h_df = pd.read_csv("thai_holidays.csv")
            th_holidays = pd.to_datetime(h_df['Holiday_Date'], dayfirst=True, errors='coerce').dt.date.tolist()
        except: pass

    # --- PAGE: DASHBOARD ---
    def show_dashboard_page():
        st.title("📊 Financial Executive Dashboard")
        if df.empty: st.warning("No Data Available"); return

        with st.expander("🔎 ตัวกรองข้อมูล (Filter)", expanded=False):
            c_y, c_m = st.columns(2)
            # FIX: Filter NaN & Float
            valid_years = sorted([int(y) for y in df['year'].unique() if pd.notna(y)])
            
            sel_year = c_y.selectbox("เลือกปี (Year)", ["All"] + valid_years)
            sel_month = c_m.selectbox("เลือกเดือน (Month)", ["All"] + list(range(1, 13)))
        
        df_show = df.copy()
        if sel_year != "All": df_show = df_show[df_show['year'] == sel_year]
        if sel_month != "All": df_show = df_show[df_show['month'] == sel_month]
        
        if df_show.empty:
            st.warning("ไม่พบข้อมูลในช่วงเวลาที่เลือก")
            return

        st.divider()
        k1, k2, k3 = st.columns(3)
        with k1: st.metric("💰 Total Revenue", f"{df_show['Price'].sum()/1e6:.2f} M THB")
        with k2: st.metric("📦 Total Bookings", f"{len(df_show):,} รายการ")
        avg_val = df_show['Price'].mean() if len(df_show) > 0 else 0
        with k3: st.metric("🏷️ Avg. Booking Value", f"{avg_val:,.0f} THB")
        
        st.divider()
        tab_fin, tab_chan, tab_cust = st.tabs(["💰 ภาพรวมการเงิน", "🌐 ช่องทางการขาย", "👥 พฤติกรรมลูกค้า"])
        
        group_col = 'Target_Room_Type' if 'Target_Room_Type' in df_show.columns else 'Room'
        
        with tab_fin:
            st.markdown("### Financial Overview")
            c1, c2 = st.columns(2)
            with c1:
                room_perf = df_show.groupby(group_col).agg({'Price': 'sum', 'Night': 'sum'}).reset_index().sort_values('Price', ascending=False)
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Bar(x=room_perf[group_col], y=room_perf['Price'], name="Revenue", marker_color='#1f77b4'), secondary_y=False)
                fig.add_trace(go.Scatter(x=room_perf[group_col], y=room_perf['Night'], name="Nights", mode='lines+markers', marker_color='#ff7f0e'), secondary_y=True)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                if 'month' in df_show.columns:
                    monthly_adr = df_show.groupby('month').apply(lambda x: x['Price'].sum() / x['Night'].sum() if x['Night'].sum() > 0 else 0).reset_index(name='ADR')
                    fig_adr = px.line(monthly_adr, x='month', y='ADR', markers=True, title="ADR Trend Analysis")
                    st.plotly_chart(fig_adr, use_container_width=True)

        with tab_chan:
            st.markdown("### Channel Strategy")
            c3, c4 = st.columns(2)
            with c3:
                res_rev = df_show.groupby('Reservation')['Price'].sum().reset_index()
                st.plotly_chart(px.pie(res_rev, values='Price', names='Reservation', hole=0.4, title="Revenue Share"), use_container_width=True)
            with c4:
                m_res = df.groupby(['month', 'Reservation']).size().reset_index(name='Count')
                m_res['M_Name'] = m_res['month'].apply(lambda x: datetime(2024, int(x), 1).strftime('%b'))
                st.plotly_chart(px.bar(m_res, x='M_Name', y='Count', color='Reservation', title="Monthly Bookings"), use_container_width=True)

        with tab_cust:
            c5, c6 = st.columns(2)
            with c5:
                heatmap_data = df_show.groupby([group_col, 'Reservation']).size().unstack(fill_value=0)
                fig_heat = px.imshow(heatmap_data, text_auto=True, aspect="auto", color_continuous_scale='Blues', title="Room vs Channel Heatmap")
                st.plotly_chart(fig_heat, use_container_width=True)
            with c6:
                if 'is_weekend' in df_show.columns:
                    df_show['DayType'] = df_show['is_weekend'].map({1: 'Weekend', 0: 'Weekday'})
                    day_rev = df_show.groupby('DayType')['Price'].sum().reset_index()
                    st.plotly_chart(px.pie(day_rev, values='Price', names='DayType', hole=0.4, title="Revenue: Weekday vs Weekend"), use_container_width=True)

    # --- PAGE: MANAGE DATA (ADMIN ONLY) ---
    def show_manage_data_page():
        st.title("🛠️ จัดการข้อมูล (Data Management)")
        
        tab_config, tab_edit, tab_import, tab_retrain = st.tabs(["⚙️ ตั้งค่าห้องพัก", "📝 แก้ไขข้อมูลจอง", "📥 นำเข้าไฟล์", "🔄 อัปเดตโมเดล"])
        
        with tab_config:
            st.markdown("### ⚙️ Room Configuration (Master Data)")
            current_config = load_room_config()
            edited_config = st.data_editor(
                current_config, 
                num_rows="dynamic", 
                use_container_width=True, 
                key="room_config_editor",
                column_config={"Allow_Extra": st.column_config.CheckboxColumn("อนุญาตเสริมเตียง?", default=True)}
            )
            if st.button("💾 บันทึกการตั้งค่าห้องพัก"):
                edited_config.to_csv(ROOM_CONFIG_FILE, index=False)
                st.success("✅ บันทึกเรียบร้อย!"); st.rerun()

        with tab_edit:
            st.markdown("### 📝 ข้อมูลการจองในระบบ (Edit/Delete)")
            
            # [FIX] โหลด RAW CSV ตรงๆ เพื่อป้องกัน Column ซ้ำจากการ Merge
            if os.path.exists(DATA_FILE):
                raw_df = pd.read_csv(DATA_FILE)
                # Ensure Room is string
                if 'Room' in raw_df.columns: raw_df['Room'] = raw_df['Room'].astype(str)
                
                edited_df = st.data_editor(raw_df, num_rows="dynamic", use_container_width=True, key="data_editor_raw")
                
                if st.button("💾 บันทึกการแก้ไขข้อมูลจอง"):
                    try:
                        edited_df.to_csv(DATA_FILE, index=False)
                        st.success("✅ บันทึกสำเร็จ!")
                        st.cache_data.clear()
                        time.sleep(1); st.rerun()
                    except Exception as e: st.error(f"Error: {e}")
            else:
                st.warning("ไม่พบไฟล์ข้อมูล (check_in_report.csv)")

        with tab_import:
            st.markdown("### 📥 Import CSV")
            up_file = st.file_uploader("เลือกไฟล์ Booking CSV", type=['csv'])
            if up_file and st.button("Append Data"):
                try:
                    new_data = pd.read_csv(up_file)
                    if 'Room' in new_data.columns: new_data['Room'] = new_data['Room'].astype(str)
                    
                    if os.path.exists(DATA_FILE):
                        current = pd.read_csv(DATA_FILE)
                        if 'Room' in current.columns: current['Room'] = current['Room'].astype(str)
                        updated = pd.concat([current, new_data], ignore_index=True)
                    else: updated = new_data
                        
                    updated.to_csv(DATA_FILE, index=False)
                    st.success("✅ นำเข้าสำเร็จ!"); st.cache_data.clear(); st.rerun()
                except Exception as e: st.error(f"Failed: {e}")
                
        with tab_retrain:
             st.markdown("### 🔄 Update Model Intelligence")
             col_m1, col_m2 = st.columns(2)
             with col_m1: st.metric("Current Accuracy (R²)", f"{metrics['xgb']['r2']*100:.2f}%")
             
             if st.button("🚀 Start Retraining"):
                with st.spinner("Training models... Please wait."):
                    success, count = retrain_system()
                    if success: st.success(f"🎉 Training เสร็จสิ้น! (เรียนรู้จาก {count} รายการ)"); time.sleep(2); st.rerun()

    # --- PAGE: FEATURE IMPORTANCE ---
    def show_importance_page():
        st.title("🧠 ความสำคัญของตัวแปร (Feature Importance)")
        st.markdown("ปัจจัยที่ส่งผลต่อการตั้งราคาห้องพัก (วิเคราะห์จาก XGBoost Model)")
        
        imp_data = metrics.get('importance', DEFAULT_METRICS['importance'])
        if not imp_data: st.warning("ยังไม่มีข้อมูล Model ให้ Retrain ก่อน"); return

        fi_df = pd.DataFrame(list(imp_data.items()), columns=['Feature', 'Importance']).sort_values('Importance', ascending=True)
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.plotly_chart(px.bar(fi_df, x='Importance', y='Feature', orientation='h', text_auto='.4f', color='Importance', color_continuous_scale='Blues'), use_container_width=True)
        with c2:
            st.dataframe(fi_df.sort_values('Importance', ascending=False), use_container_width=True, height=400)

    # --- PAGE: PREDICTION ---
    def show_pricing_page():
        st.title("🔮 ระบบพยากรณ์ราคา (Price Prediction)")
        
        if xgb_model is None: st.error("❌ Model not found. Please Retrain first."); return

        config_df = load_room_config()
        available_channels = sorted(list(set(le_res.classes_) | set(df['Reservation'].unique()))) if not df.empty else le_res.classes_

        with st.container(border=True):
            st.subheader("1. กำหนดข้อมูลการเข้าพัก")
            c_date, c_room, c_chan = st.columns(3)
            with c_date:
                # FIX: Allow past dates (min_value=None)
                dates = st.date_input("เลือกวัน Check-in / Check-out", value=(datetime.now(), datetime.now() + timedelta(days=1)))
                
                nights = 1; is_h = False; checkin_date = datetime.now()
                if isinstance(dates, tuple):
                    if len(dates) == 2:
                        start, end = dates
                        nights = (end - start).days
                        checkin_date = start
                        date_range = pd.date_range(start, end - timedelta(days=1))
                        is_h = any(d.date() in th_holidays for d in date_range)
                    elif len(dates) == 1: checkin_date = dates[0]
                
                st.markdown(f"📅 **{nights} คืน** | 🏖️ เทศกาล: **{'✅ ใช่' if is_h else '❌ ไม่ใช่'}**")

            with c_room:
                room_opts = [f"{row['Room_Type']} (Start {row['Base_Price']:,})" for _, row in config_df.iterrows()]
                selected_opt = st.selectbox("ประเภทห้อง (Room Type)", room_opts)
                selected_room_name = selected_opt.split(" (Start")[0]
                room_info = config_df[config_df['Room_Type'] == selected_room_name].iloc[0]
                base_p = room_info['Base_Price']; allow_extra = room_info['Allow_Extra']
                st.number_input("จำนวนผู้เข้าพัก (Fix)", value=2, disabled=True)
            
            with c_chan: res = st.selectbox("ช่องทาง (Channel)", available_channels)
                
            is_w = checkin_date.weekday() in [5, 6]
            try: r_code = le_room.transform([selected_room_name])[0]
            except: r_code = 0 
            try: res_code = le_res.transform([res])[0]
            except: res_code = 0
            
            inp = pd.DataFrame([{'Night': nights, 'total_guests': 2, 'is_holiday': 1 if is_h else 0, 'is_weekend': 1 if is_w else 0, 'month': checkin_date.month, 'weekday': checkin_date.weekday(), 'RoomType_encoded': r_code, 'Reservation_encoded': res_code}])

            st.divider()
            b1, b2 = st.columns(2)
            calc = b1.button("🚀 คำนวณราคาห้องนี้", type="primary", use_container_width=True)
            calc_all = b2.button("📋 ประเมินราคาทุกห้อง (All Types)", type="secondary", use_container_width=True)
            
            if calc:
                p_xgb = xgb_model.predict(inp)[0]; p_lr = lr_model.predict(inp)[0]
                final_xgb, mul_xgb = calculate_price_logic(p_xgb, base_p, nights, is_h, is_w)
                final_lr, mul_lr = calculate_price_logic(p_lr, base_p, nights, is_h, is_w)
                
                st.markdown("### 🏷️ ผลลัพธ์การพยากรณ์")
                col_a, col_b = st.columns(2)
                
                with col_a:
                    with st.container(border=True):
                        st.markdown("#### 🅰️ ทางเลือก A (Machine Learning)")
                        st.metric("ราคาแนะนำ (2 ท่าน)", f"{final_xgb:,.0f} THB")
                        if allow_extra: 
                            st.metric("➕ รวมเตียงเสริม (+300)", f"{final_xgb + (300*nights):,.0f} THB")
                        st.divider()
                        col_m1, col_m2 = st.columns(2)
                        with col_m1: st.markdown(f"**R² Score:** `{metrics['xgb']['r2']*100:.2f}%`")
                        with col_m2: st.markdown(f"**MAE:** `{metrics['xgb']['mae']:.0f}`")
                        if mul_xgb > 1.0: st.info(f"💡 ปรับราคาขึ้น x{mul_xgb} (Holiday/Weekend Rule)")
                
                with col_b:
                    with st.container(border=True):
                        st.markdown("#### 🅱️ ทางเลือก B (Statistical)")
                        st.metric("ราคาประเมิน (2 ท่าน)", f"{final_lr:,.0f} THB")
                        if allow_extra: 
                            st.metric("➕ รวมเตียงเสริม (+300)", f"{final_lr + (300*nights):,.0f} THB")
                        st.divider()
                        col_m3, col_m4 = st.columns(2)
                        with col_m3: st.markdown(f"**R² Score:** `{metrics['lr']['r2']*100:.2f}%`")
                        with col_m4: st.markdown(f"**MAE:** `{metrics['lr']['mae']:.0f}`")

            if calc_all:
                st.markdown("### 📋 ตารางราคาแนะนำทุกห้อง (สำหรับ 2 ท่าน)")
                results = []
                for _, row in config_df.iterrows():
                    r_name = row['Room_Type']; bp = row['Base_Price']; allow_ex = row['Allow_Extra']
                    try: r_c = le_room.transform([r_name])[0]
                    except: continue 
                    tmp_inp = inp.copy(); tmp_inp['RoomType_encoded'] = r_c; tmp_inp['total_guests'] = 2
                    px = xgb_model.predict(tmp_inp)[0]
                    fx, mx = calculate_price_logic(px, bp, nights, is_h, is_w)
                    note = f"Rule x{mx}" if mx > 1 else "Normal"
                    if allow_ex: note += " | +Extra Option"
                    results.append({"Room Type": r_name, "Base Price": f"{bp:,}", "Recomm. Price": f"{fx:,.0f}", "Note": note})
                st.dataframe(pd.DataFrame(results), use_container_width=True)

    def show_about_page():
        st.title("ℹ️ เกี่ยวกับระบบ")
        c1, c2 = st.columns([1, 2])
        with c1: 
            if os.path.exists("my_profile.jpg"): st.image("my_profile.jpg", width=200)
            else: st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=200)
        with c2:
            st.markdown("### ผู้จัดทำ"); st.markdown("**ว่าที่ร้อยตรีพรพินิต วิรัตน์สกุลชัย**")
            st.caption("คณะนวัตกรรม เทคโนโลยีและการสร้างสรรค์ มหาวิทยาลัยฟาร์อีสเทอร์น")
            st.info("วิทยานิพนธ์: การพัฒนาระบบสนับสนุนการตัดสินใจเพื่อการพยากรณ์ราคาแบบพลวัต")

    # --- SIDEBAR NAV (NEW STRUCTURE) ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2933/2933116.png", width=80)
        st.markdown(f"**User:** {st.session_state['username']}")
        st.divider()
        
        page_selection = ""
        st.markdown("#### 🏠 หน้าแรก")
        if st.button("📊 Dashboard สรุปผล", use_container_width=True): page_selection = "dashboard"
            
        st.markdown("#### 🔮 พยากรณ์")
        if st.button("📈 ระบบพยากรณ์ราคา", use_container_width=True): page_selection = "pricing"
        
        # Admin Only Menu
        if st.session_state['username'] == "admin":
            if st.button("🛠️ จัดการข้อมูล (Admin)", use_container_width=True): page_selection = "manage"
            if st.button("🧠 Feature Importance", use_container_width=True): page_selection = "importance"
            
        st.markdown("#### ℹ️ เกี่ยวกับ")
        if st.button("📝 เกี่ยวกับระบบ", use_container_width=True): page_selection = "about"
        
        st.divider()
        if st.button("Logout", use_container_width=True): st.session_state['logged_in'] = False; st.rerun()

    if 'current_page' not in st.session_state: st.session_state['current_page'] = "dashboard"
    if page_selection: st.session_state['current_page'] = page_selection

    curr = st.session_state['current_page']
    if curr == "dashboard": show_dashboard_page()
    elif curr == "pricing": show_pricing_page()
    elif curr == "manage": show_manage_data_page()
    elif curr == "importance": show_importance_page()
    elif curr == "about": show_about_page()