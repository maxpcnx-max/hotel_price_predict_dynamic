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
ROOM_MAPPING_FILE = "room_type.csv" 
ROOM_CONFIG_FILE = "room_config.csv" 
METRICS_FILE = "model_metrics.json"

MODEL_FILES = {
    'xgb': 'xgb_hotel_model.joblib',
    'lr': 'lr_hotel_model.joblib',
    'le_room': 'le_room.joblib',
    'le_res': 'le_res.joblib'
}

# Default Configuration
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
    'xgb': {'mae': 0.0, 'r2': 0.0},
    'lr':  {'mae': 0.0, 'r2': 0.0},
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
    
    # Admin
    c.execute('SELECT * FROM users WHERE username = "admin"')
    if not c.fetchone(): c.execute('INSERT INTO users VALUES (?,?)', ("admin", "1234"))
        
    # User
    c.execute('SELECT * FROM users WHERE username = "user"')
    if not c.fetchone(): c.execute('INSERT INTO users VALUES (?,?)', ("user", "1234"))
        
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

def get_thai_holidays():
    if not os.path.exists("thai_holidays.csv"):
        try: gdown.download("https://drive.google.com/uc?id=1L-pciKEeRce1gzuhdtpIGcLs0fYHnbZw", "thai_holidays.csv", quiet=True)
        except: return set()
    
    if os.path.exists("thai_holidays.csv"):
        try:
            h_df = pd.read_csv("thai_holidays.csv")
            return set(pd.to_datetime(h_df['Holiday_Date'], dayfirst=True, errors='coerce').dt.date)
        except: return set()
    return set()

init_db()

# ==========================================================
# 3. BACKEND SYSTEM
# ==========================================================

@st.cache_data
def load_data():
    """โหลดข้อมูลและ Merge เพื่อใช้แสดงผล"""
    if not os.path.exists(DATA_FILE):
        try: gdown.download("https://drive.google.com/uc?id=1dxgKIvSTelLaJvAtBSCMCU5K4FuJvfri", DATA_FILE, quiet=True)
        except: return pd.DataFrame()

    try:
        df = pd.read_csv(DATA_FILE)
        
        # Date Handling
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            df['is_weekend'] = df['Date'].dt.weekday.isin([5, 6]).astype(int)
            df['year'] = df['Date'].dt.year
            df['month'] = df['Date'].dt.month
            
        if 'Room' in df.columns:
            df['Room'] = df['Room'].astype(str)

        # Merge Room Type
        if os.path.exists(ROOM_MAPPING_FILE):
            room_type = pd.read_csv(ROOM_MAPPING_FILE)
            if 'Room' in room_type.columns: room_type['Room'] = room_type['Room'].astype(str)
            if 'Room_Type' in room_type.columns:
                df = df.merge(room_type, on='Room', how='left')
                if 'Room_Type' in df.columns: df = df.rename(columns={'Room_Type': 'Target_Room_Type'})
                elif 'Room_Type_y' in df.columns: df = df.rename(columns={'Room_Type_y': 'Target_Room_Type'})
        
        # Filter Invalid
        df = df.dropna(subset=['Target_Room_Type'])
        df['Reservation'] = df['Reservation'].fillna('Unknown')
        
        return df
    except Exception: return pd.DataFrame()

def load_metrics():
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, 'r') as f: return json.load(f)
        except: return DEFAULT_METRICS
    return DEFAULT_METRICS

@st.cache_resource
def load_system_models():
    for name, file in MODEL_FILES.items():
        if not os.path.exists(file): return None, None, None, None, None
    return joblib.load(MODEL_FILES['xgb']), joblib.load(MODEL_FILES['lr']), joblib.load(MODEL_FILES['le_room']), joblib.load(MODEL_FILES['le_res']), load_metrics()

def calculate_price_logic(model_price, base_price, nights, is_holiday, is_weekend):
    multiplier = 1.0
    if is_holiday: multiplier = max(multiplier, 1.5)
    if is_weekend: multiplier = max(multiplier, 1.2)
        
    floor_price = base_price * multiplier * nights
    final_price = max(model_price, floor_price)
    final_price = max(final_price, base_price * nights * 0.5) 
    return final_price, multiplier

def retrain_system():
    try:
        st.cache_data.clear() # Clear Cache First
        df = load_data() 
        if df.empty: return False, 0
        df = df.dropna(subset=['Price', 'Night'])
        
        # Clean Data
        df['Night'] = df['Night'].fillna(1)
        df['Adults'] = df['Adults'].fillna(2)
        df[['Children', 'Infants', 'Extra Person']] = df[['Children', 'Infants', 'Extra Person']].fillna(0)
        
        holidays_set = get_thai_holidays()
        if holidays_set: df['is_holiday'] = df['Date'].dt.date.isin(holidays_set).astype(int)
        else: df['is_holiday'] = 0

        df['is_weekend'] = df['Date'].dt.weekday.isin([5, 6]).astype(int)
        df['total_guests'] = df[['Adults', 'Children', 'Infants', 'Extra Person']].sum(axis=1)
        
        # Train
        le_room_new = LabelEncoder()
        df['RoomType_encoded'] = le_room_new.fit_transform(df['Target_Room_Type'].astype(str))
        le_res_new = LabelEncoder()
        df['Reservation_encoded'] = le_res_new.fit_transform(df['Reservation'].astype(str))
        
        feature_cols = ['Night', 'total_guests', 'is_holiday', 'is_weekend', 'month', 'weekday', 'RoomType_encoded', 'Reservation_encoded']
        X = df[feature_cols].fillna(0); y = df['Price']
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
        
        # Save
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
    except Exception: return False, 0

# ==========================================================
# 4. UI PAGES
# ==========================================================

def login_page():
    st.markdown("""<style>.stTextInput > div > div > input {text-align: center;}</style>""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/2933/2933116.png", width=120)
        st.title("🔒 Login System")
        st.markdown("**ระบบพยากรณ์ราคาห้องพัก (Hotel Price Forecasting)**")
        st.caption("กรุณาเข้าสู่ระบบ (Admin/User)")
        u = st.text_input("Username"); p = st.text_input("Password", type="password")
        if st.button("Login", type="primary", use_container_width=True):
            if login_user(u, p): 
                st.session_state['logged_in'] = True; st.session_state['username'] = u; st.rerun()
            else: st.error("ข้อมูลไม่ถูกต้อง")

if not st.session_state['logged_in']:
    login_page()
else:
    # Load Resources
    df = load_data()
    xgb_model, lr_model, le_room, le_res, metrics = load_system_models()
    if metrics is None: metrics = DEFAULT_METRICS
    room_config_df = load_room_config()
    holidays_set = get_thai_holidays()

    # --- DASHBOARD ---
    def show_dashboard_page():
        st.title("📊 Financial Executive Dashboard")
        if df.empty: st.warning("ไม่พบข้อมูล"); return

        with st.expander("🔎 ตัวกรองข้อมูล (Filter)", expanded=False):
            c_y, c_m = st.columns(2)
            # Filter NaN & Float
            valid_years = sorted([int(y) for y in df['year'].unique() if pd.notna(y)])
            sel_year = c_y.selectbox("เลือกปี (Year)", ["All"] + valid_years)
            sel_month = c_m.selectbox("เลือกเดือน (Month)", ["All"] + list(range(1, 13)))
        
        df_show = df.copy()
        if sel_year != "All": df_show = df_show[df_show['year'] == sel_year]
        if sel_month != "All": df_show = df_show[df_show['month'] == sel_month]
        
        if df_show.empty: st.warning("ไม่พบข้อมูลในช่วงเวลานี้"); return

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
            c1, c2 = st.columns(2)
            with c1:
                rp = df_show.groupby(group_col).agg({'Price': 'sum', 'Night': 'sum'}).reset_index().sort_values('Price', ascending=False)
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Bar(x=rp[group_col], y=rp['Price'], name="Revenue", marker_color='#1f77b4'), secondary_y=False)
                fig.add_trace(go.Scatter(x=rp[group_col], y=rp['Night'], name="Nights", mode='lines+markers', marker_color='#ff7f0e'), secondary_y=True)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                if 'month' in df_show.columns:
                    ma = df_show.groupby('month').apply(lambda x: x['Price'].sum()/x['Night'].sum() if x['Night'].sum()>0 else 0).reset_index(name='ADR')
                    st.plotly_chart(px.line(ma, x='month', y='ADR', markers=True, title="ADR Trend"), use_container_width=True)

        with tab_chan:
            c3, c4 = st.columns(2)
            with c3: st.plotly_chart(px.pie(df_show, values='Price', names='Reservation', hole=0.4, title="Revenue Share"), use_container_width=True)
            with c4:
                m_res = df.groupby(['month', 'Reservation']).size().reset_index(name='Count')
                m_res['M_Name'] = m_res['month'].apply(lambda x: datetime(2024, int(x), 1).strftime('%b'))
                st.plotly_chart(px.bar(m_res, x='M_Name', y='Count', color='Reservation', title="Monthly Bookings"), use_container_width=True)

        with tab_cust:
            c5, c6 = st.columns(2)
            with c5: st.plotly_chart(px.imshow(df_show.groupby([group_col, 'Reservation']).size().unstack(fill_value=0), text_auto=True, aspect="auto", color_continuous_scale='Blues', title="Heatmap"), use_container_width=True)
            with c6:
                if 'is_weekend' in df_show.columns:
                    df_show['DayType'] = df_show['is_weekend'].map({1: 'Weekend', 0: 'Weekday'})
                    st.plotly_chart(px.pie(df_show.groupby('DayType')['Price'].sum().reset_index(), values='Price', names='DayType', hole=0.4, title="Weekday vs Weekend"), use_container_width=True)

    # --- MANAGE DATA (ADMIN) ---
    def show_manage_data_page():
        st.title("🛠️ จัดการข้อมูล (Admin)")
        t1, t2, t3, t4 = st.tabs(["⚙️ ตั้งค่าห้อง", "📝 แก้ไขข้อมูล", "📥 นำเข้าไฟล์", "🔄 อัปเดตโมเดล"])
        
        with t1: # Config
            curr = load_room_config()
            ed = st.data_editor(curr, num_rows="dynamic", use_container_width=True, key="cfg_ed")
            if st.button("💾 บันทึกตั้งค่า"):
                ed.to_csv(ROOM_CONFIG_FILE, index=False)
                with st.spinner("Saving..."): time.sleep(1)
                st.success("✅ บันทึกแล้ว"); st.rerun()

        with t2: # Edit Raw Data
            if os.path.exists(DATA_FILE):
                raw = pd.read_csv(DATA_FILE)
                if 'Room' in raw.columns: raw['Room'] = raw['Room'].astype(str)
                ed_raw = st.data_editor(raw, num_rows="dynamic", use_container_width=True, key="raw_ed")
                if st.button("💾 บันทึกแก้ไข"):
                    ed_raw.to_csv(DATA_FILE, index=False)
                    st.cache_data.clear()
                    with st.spinner("Updating..."): time.sleep(1.5)
                    st.success("✅ บันทึกสำเร็จ"); st.rerun()
            else: st.warning("ไม่พบไฟล์ข้อมูล")

        with t3: # Import
            up = st.file_uploader("CSV Booking", type=['csv'])
            if up and st.button("Append Data"):
                try:
                    new = pd.read_csv(up)
                    if 'Room' in new.columns: new['Room'] = new['Room'].astype(str)
                    curr = pd.read_csv(DATA_FILE) if os.path.exists(DATA_FILE) else new
                    if 'Room' in curr.columns: curr['Room'] = curr['Room'].astype(str)
                    pd.concat([curr, new], ignore_index=True).to_csv(DATA_FILE, index=False)
                    st.cache_data.clear()
                    with st.spinner("Importing..."): time.sleep(1.5)
                    st.success("✅ นำเข้าสำเร็จ"); st.rerun()
                except Exception as e: st.error(f"Error: {e}")

        with t4: # Retrain
            r2 = metrics.get('xgb', {}).get('r2', 0)
            st.metric("Current Accuracy", f"{r2*100:.2f}%")
            if st.button("🚀 Start Retraining"):
                with st.spinner("Training..."):
                    suc, cnt = retrain_system()
                    time.sleep(1)
                    if suc: 
                        st.success(f"🎉 เสร็จสิ้น! ({cnt} records)")
                        time.sleep(2); st.rerun()
                    else: st.error("Failed")

    # --- IMPORTANCE ---
    def show_importance_page():
        st.title("🧠 Feature Importance")
        imp = metrics.get('importance', {})
        if not imp: st.warning("No Data"); return
        df_imp = pd.DataFrame(list(imp.items()), columns=['Feature', 'Importance']).sort_values('Importance', ascending=True)
        c1, c2 = st.columns([2, 1])
        with c1: st.plotly_chart(px.bar(df_imp, x='Importance', y='Feature', orientation='h', text_auto='.4f'), use_container_width=True)
        with c2: st.dataframe(df_imp.sort_values('Importance', ascending=False), use_container_width=True)

    # --- PREDICTION ---
    def show_pricing_page():
        st.title("🔮 ระบบพยากรณ์ราคา")
        if xgb_model is None: st.error("❌ Model not found"); return

        cfg = load_room_config()
        chans = sorted(list(set(le_res.classes_) | set(df['Reservation'].unique()))) if not df.empty else le_res.classes_

        with st.container(border=True):
            st.subheader("1. ข้อมูลการเข้าพัก")
            c1, c2, c3 = st.columns(3)
            with c1:
                dates = st.date_input("Check-in/out", value=(datetime.now(), datetime.now()+timedelta(days=1)))
                nights=1; is_h=False; chk=datetime.now()
                if isinstance(dates, tuple):
                    if len(dates)==2:
                        s, e = dates
                        nights = (e-s).days
                        chk = s
                        # Fix Attribute Error by comparing date objects directly
                        stay_dates = [s.date() + timedelta(days=x) for x in range(nights)]
                        is_h = any(d in holidays_set for d in stay_dates)
                    elif len(dates)==1: chk=dates[0]
                st.markdown(f"📅 **{nights} คืน** | 🏖️ เทศกาล: **{'✅ ใช่' if is_h else '❌ ไม่ใช่'}**")

            with c2:
                opts = [f"{r['Room_Type']} (Start {r['Base_Price']:,})" for _, r in cfg.iterrows()]
                sel = st.selectbox("Room Type", opts)
                r_name = sel.split(" (Start")[0]
                r_info = cfg[cfg['Room_Type']==r_name].iloc[0]
                bp = r_info['Base_Price']; allow_ex = r_info['Allow_Extra']
                st.number_input("Guests (Fix)", value=2, disabled=True)

            with c3: res = st.selectbox("Channel", chans)

            is_w = chk.weekday() in [5, 6]
            try: rc = le_room.transform([r_name])[0]
            except: rc=0
            try: resc = le_res.transform([res])[0]
            except: resc=0
            
            inp = pd.DataFrame([{'Night': nights, 'total_guests': 2, 'is_holiday': 1 if is_h else 0, 'is_weekend': 1 if is_w else 0, 'month': chk.month, 'weekday': chk.weekday(), 'RoomType_encoded': rc, 'Reservation_encoded': resc}])

            st.divider()
            b1, b2 = st.columns(2)
            calc = b1.button("🚀 คำนวณราคา", type="primary", use_container_width=True)
            calc_all = b2.button("📋 ประเมินทุกห้อง", type="secondary", use_container_width=True)

            if calc:
                px = xgb_model.predict(inp)[0]; pl = lr_model.predict(inp)[0]
                fx, mx = calculate_price_logic(px, bp, nights, is_h, is_w)
                fl, ml = calculate_price_logic(pl, bp, nights, is_h, is_w)

                c_a, c_b = st.columns(2)
                with c_a:
                    with st.container(border=True):
                        st.markdown("#### 🅰️ ML (XGBoost)")
                        st.metric("ราคาแนะนำ (2 ท่าน)", f"{fx:,.0f}")
                        if allow_ex: st.metric("➕ เสริมเตียง", f"{fx+(300*nights):,.0f}")
                        st.divider()
                        st.markdown(f"R²: `{metrics['xgb']['r2']*100:.2f}%` | MAE: `{metrics['xgb']['mae']:.0f}`")
                        if mx>1: st.info(f"x{mx} (Rule)")
                with c_b:
                    with st.container(border=True):
                        st.markdown("#### 🅱️ Stat (Linear Reg)")
                        st.metric("ราคาประเมิน (2 ท่าน)", f"{fl:,.0f}")
                        if allow_ex: st.metric("➕ เสริมเตียง", f"{fl+(300*nights):,.0f}")
                        st.divider()
                        st.markdown(f"R²: `{metrics['lr']['r2']*100:.2f}%` | MAE: `{metrics['lr']['mae']:.0f}`")

            if calc_all:
                res_list = []
                for _, r in cfg.iterrows():
                    rn=r['Room_Type']; b=r['Base_Price']; al=r['Allow_Extra']
                    try: rtc = le_room.transform([rn])[0]
                    except: continue
                    tmp = inp.copy(); tmp['RoomType_encoded']=rtc
                    pred = xgb_model.predict(tmp)[0]
                    fp, mp = calculate_price_logic(pred, b, nights, is_h, is_w)
                    note = f"Rule x{mp}" if mp>1 else "-"
                    if al: note += " | +Extra"
                    res_list.append({"Type": rn, "Base": f"{b:,}", "Price": f"{fp:,.0f}", "Note": note})
                st.dataframe(pd.DataFrame(res_list), use_container_width=True)

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

    # --- SIDEBAR (CATEGORY + ROLE) ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2933/2933116.png", width=80)
        st.markdown(f"**User:** {st.session_state['username']}")
        st.divider()
        
        pg = ""
        st.markdown("#### 🏠 หน้าแรก")
        if st.button("📊 Dashboard", use_container_width=True): pg="dash"
            
        st.markdown("#### 🔮 พยากรณ์")
        if st.button("📈 ระบบพยากรณ์", use_container_width=True): pg="price"
        
        if st.session_state['username'] == "admin":
            st.markdown("#### 🔧 Admin")
            if st.button("🛠️ จัดการข้อมูล", use_container_width=True): pg="manage"
            if st.button("🧠 Importance", use_container_width=True): pg="imp"
            
        st.markdown("#### ℹ️ อื่นๆ")
        if st.button("📝 เกี่ยวกับ", use_container_width=True): pg="about"
        
        st.divider()
        if st.button("Logout"): st.session_state['logged_in']=False; st.rerun()

    if 'page' not in st.session_state: st.session_state['page'] = "dash"
    if pg: st.session_state['page'] = pg

    p = st.session_state['page']
    if p=="dash": show_dashboard_page()
    elif p=="price": show_pricing_page()
    elif p=="manage": show_manage_data_page()
    elif p=="imp": show_importance_page()
    elif p=="about": show_about_page()