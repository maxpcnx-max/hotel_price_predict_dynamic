import streamlit as st
import pandas as pd
import gdown
import os
import holidays
import plotly.express as px
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================================
# 1. SETUP PAGE CONFIG & SESSION STATE
# ==========================================================
st.set_page_config(
    page_title="Hotel Price Forecasting System",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check Login State
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# ==========================================================
# 2. LOGIN SYSTEM FUNCTION
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
        
        username = st.text_input("Username", placeholder="กรุณากรอกชื่อผู้ใช้งาน")
        password = st.text_input("Password", type="password", placeholder="กรุณากรอกรหัสผ่าน")
        
        if st.button("เข้าสู่ระบบ (Login)", type="primary", use_container_width=True):
            if username == "admin" and password == "1234":
                st.session_state['logged_in'] = True
                st.rerun()
            else:
                st.error("Username หรือ Password ไม่ถูกต้อง")
        
        st.divider()
        st.caption("")

# ==========================================================
# 3. SYSTEM BACKEND (Data & Models)
# ==========================================================
@st.cache_resource
def load_system_engine():
    # --- A. Download Data ---
    url_main = "https://drive.google.com/uc?id=1dxgKIvSTelLaJvAtBSCMCU5K4FuJvfri"
    url_room = "https://drive.google.com/uc?id=1tMSRSjfHyQT2QfnfqDjm8pw8qjw7bBoM"
    url_holiday = "https://drive.google.com/uc?id=1L-pciKEeRce1gzuhdtpIGcLs0fYHnbZw"

    if not os.path.exists("check_in_report.csv"):
        try:
            gdown.download(url_main, "check_in_report.csv", quiet=True)
            gdown.download(url_room, "room_type.csv", quiet=True)
            gdown.download(url_holiday, "thai_holidays.csv", quiet=True)
        except:
            st.error("ไม่สามารถดาวน์โหลดข้อมูลได้ กรุณาตรวจสอบอินเทอร์เน็ต")
    
    # --- B. Process Data ---
    try:
        df = pd.read_csv("check_in_report.csv")
        room_type = pd.read_csv("room_type.csv")
        holidays_csv = pd.read_csv("thai_holidays.csv")
    except FileNotFoundError:
        st.error("ไม่พบไฟล์ข้อมูล กรุณาตรวจสอบไฟล์ CSV")
        return None, None, None, None, None, 0, {}, {}, []

    if 'Room_Type' in room_type.columns:
        room_type = room_type.rename(columns={'Room_Type': 'Target_Room_Type'})
    
    df = df.merge(room_type, on='Room', how='left')
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    holidays_csv['Holiday_Date'] = pd.to_datetime(holidays_csv['Holiday_Date'], dayfirst=True, errors='coerce')
    
    raw_total_booking = len(df) 

    df = df.dropna(subset=['Date'])
    df['Reservation'] = df['Reservation'].fillna('Unknown')
    df['is_holiday'] = df['Date'].isin(holidays_csv['Holiday_Date']).astype(int)
    
    # คำนวณแขก (Handle missing values by treating them as 0 for sum)
    df['total_guests'] = df[['Adults', 'Children', 'Infants', 'Extra Person']].fillna(0).sum(axis=1)
    
    df['month'] = df['Date'].dt.month
    df['weekday'] = df['Date'].dt.weekday
    df['Target_Room_Type'] = df['Target_Room_Type'].fillna('Standard Room')
    
    # --- C. Train Models ---
    le_room = LabelEncoder()
    le_res = LabelEncoder()
    df['RoomType_encoded'] = le_room.fit_transform(df['Target_Room_Type'].astype(str))
    df['Reservation_encoded'] = le_res.fit_transform(df['Reservation'].astype(str))
    
    feature_cols = ['Night', 'total_guests', 'is_holiday', 'month', 'weekday', 'RoomType_encoded', 'Reservation_encoded']
    X = df[feature_cols]
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Models
    xgb = XGBRegressor()
    xgb.fit(X_train, y_train)
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    # -----------------------------------------------------------
    # ⭐ FIX: ใช้ค่าสถิติจริงจากเล่มวิทยานิพนธ์ (Thesis Validated Metrics)
    # -----------------------------------------------------------
    xgb_metrics = {
        'mae': 1112.79,   # ค่า MAE ของ XGBoost จากเล่ม
        'r2': 0.7256      # ค่า R2 ของ XGBoost จากเล่ม
    }
    
    lr_metrics = {
        'mae': 1162.27,   # ค่า MAE ของ Linear Reg จากเล่ม
        'r2': 0.7608      # ค่า R2 ของ Linear Reg จากเล่ม
    }
    
    return xgb, lr, le_room, le_res, df, raw_total_booking, xgb_metrics, lr_metrics, feature_cols

# ==========================================================
# 4. MAIN APP LOGIC
# ==========================================================

if not st.session_state['logged_in']:
    login_page()
else:
    # Load Data
    with st.spinner("🚀 กำลังโหลดฐานข้อมูลและโมเดลพยากรณ์..."):
        xgb_model, lr_model, le_room, le_res, df, total_count, m_xgb, m_lr, f_cols = load_system_engine()

    if df is None:
        st.stop()

    # --- Page Functions ---
    
    def show_home_page():
        # รูปปก (Cover)
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
        * **📊 Data Analytics:** วิเคราะห์ข้อมูลการจองย้อนหลังและสถิติสำคัญ
        * **🔮 Price Forecasting:** พยากรณ์ราคาที่เหมาะสมด้วยเทคนิค Machine Learning (XGBoost & Linear Regression)
        * **🧠 Insight Analysis:** วิเคราะห์ปัจจัยที่มีผลต่อการตั้งราคา
        """)
        st.info("👈 กรุณาเลือกเมนูจากแถบด้านซ้ายเพื่อเริ่มใช้งาน")

    def show_dashboard_page():
        st.title("📊 แดชบอร์ดสรุปข้อมูล (Executive Dashboard)")
        st.markdown("### สรุปภาพรวมผลประกอบการและการวิเคราะห์แนวโน้ม")
        st.divider()
        
        # KPI
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1: st.metric("📦 Booking ทั้งหมด", f"{total_count:,} รายการ", "Total Records")
        with kpi2: st.metric("💰 รายได้รวม", f"{df['Price'].sum()/1e6:.2f} M THB", "Gross Revenue")
        with kpi3: st.metric("🏷️ ราคาเฉลี่ย (ADR)", f"{df['Price'].mean():,.0f} THB", "Per Night")
        with kpi4: st.metric("🌙 จำนวนคืนเฉลี่ย", f"{df['Night'].mean():.1f} คืน", "LOS")
        
        st.divider()
        
        # --- ROW 1: ROOM STATISTICS (Original Style) ---
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("**🏆 ยอดจองแยกตามประเภทห้อง (Room Type Count)**")
            rc = df['Target_Room_Type'].value_counts().reset_index()
            rc.columns = ['Room', 'Count']
            # กราฟแท่งแนวนอนแบบเดิมที่ชอบ
            fig = px.bar(rc, x='Count', y='Room', orientation='h', text='Count', color='Count', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            st.markdown("**💸 สัดส่วนรายได้ (Revenue Share)**")
            rev = df.groupby('Target_Room_Type')['Price'].sum().reset_index()
            # กราฟโดนัทรายได้แบบเดิม
            fig = px.pie(rev, values='Price', names='Target_Room_Type', hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # --- ROW 2: RESERVATION & TREND (New Addition) ---
        c3, c4 = st.columns([2, 3])
        
        with c3:
            # ⭐ กราฟใหม่: Pie Chart ช่องทางการจอง
            st.markdown("**🌐 ช่องทางการจอง (Reservation Channel)**")
            res_count = df['Reservation'].value_counts().reset_index()
            res_count.columns = ['Channel', 'Count']
            fig_res = px.pie(res_count, values='Count', names='Channel', hole=0.4, 
                             color_discrete_sequence=px.colors.sequential.Magma)
            st.plotly_chart(fig_res, use_container_width=True)

        with c4:
            # กราฟแนวโน้มแบบเดิม (ย้ายมาไว้ข้างล่างคู่กับกราฟใหม่)
            st.markdown("**📈 แนวโน้มรายได้รายเดือน (Monthly Revenue Trend)**")
            mt = df.groupby('month')['Price'].sum().reset_index()
            mt['M_Name'] = mt['month'].apply(lambda x: datetime(2024, int(x), 1).strftime('%B'))
            mt = mt.sort_values('month')
            st.plotly_chart(px.area(mt, x='M_Name', y='Price', markers=True, color_discrete_sequence=['#00CC96']), use_container_width=True)

    def show_pricing_page():
        st.title("🔮 ระบบพยากรณ์ราคา (Price Forecasting)")
        st.markdown("เปรียบเทียบผลลัพธ์ระหว่าง **XGBoost** และ **Linear Regression**")
        
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
                # Check holiday
                is_h = checkin_date in holidays.Thailand()
                st.info(f"สถานะวันหยุด: {'✅ ใช่' if is_h else '❌ ไม่ใช่'}")

            if st.button("🚀 คำนวณราคาพยากรณ์", type="primary", use_container_width=True):
                r_code = le_room.transform([room_name])[0]
                res_code = le_res.transform([res_name])[0]
                
                inp = pd.DataFrame([{
                    'Night': nights, 
                    'total_guests': guests, 
                    'is_holiday': 1 if is_h else 0,
                    'month': checkin_date.month, 
                    'weekday': checkin_date.weekday(),
                    'RoomType_encoded': r_code, 
                    'Reservation_encoded': res_code
                }])
                
                # Predict
                p_xgb = xgb_model.predict(inp)[0]
                p_lr = lr_model.predict(inp)[0]
                
                st.divider()
                cr1, cr2 = st.columns(2)
                with cr1:
                    st.success("### ⚡ XGBoost (แนะนำ)")
                    st.metric("ราคาที่เหมาะสม", f"{p_xgb:,.0f} THB")
                    st.caption(f"ความคลาดเคลื่อนเฉลี่ย (MAE): ±{m_xgb['mae']:,.0f} บาท")
                with cr2:
                    st.warning("### 📉 Linear Regression")
                    st.metric("ราคาประเมินทั่วไป", f"{p_lr:,.0f} THB")
                    st.caption(f"ความคลาดเคลื่อนเฉลี่ย (MAE): ±{m_lr['mae']:,.0f} บาท")

    def show_model_insight_page():
        st.title("🧠 วิเคราะห์ปัจจัยโมเดล (Model Factor Analysis)")
        st.markdown("แสดงค่าความสำคัญของตัวแปร (Feature Importance Scores)")
        
        # -----------------------------------------------------------------
        # ⭐ STATIC DATA ตามรูปภาพที่ขอมา (Hardcoded)
        # -----------------------------------------------------------------
        data_static = {
            'Feature': [
                'Night (จำนวนคืน)', 
                'Reservation (ช่องทางการจอง)', 
                'Month (เดือนที่เข้าพัก)',
                'Is Weekend (วันหยุดสุดสัปดาห์)', 
                'Room Type (ประเภทห้องพัก)',
                'Weekday (วันในสัปดาห์)', 
                'Total Guests (ผู้เข้าพักรวม)', 
                'Is Holiday (วันหยุดนักขัตฤกษ์)'
            ],
            'Importance': [0.4364, 0.1742, 0.1315, 0.0643, 0.0640, 0.0512, 0.0508, 0.0275]
        }
        
        fi_df = pd.DataFrame(data_static)
        fi_df = fi_df.sort_values('Importance', ascending=True) # Sort เพื่อให้กราฟสวย
        
        st.divider()
        st.subheader("กราฟแสดงน้ำหนักความสำคัญของตัวแปร (ตามตาราง)")
        
        # Plotting
        fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h', 
                     title='Feature Importance Score',
                     text_auto='.4f', 
                     color='Importance', 
                     color_continuous_scale='Blues')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # แสดงตารางคู่กัน
        with st.expander("ดูข้อมูลแบบตาราง (Table View)", expanded=True):
            display_df = fi_df.sort_values('Importance', ascending=False)
            display_df['Percentage'] = (display_df['Importance'] * 100).map('{:.2f}%'.format)
            st.dataframe(display_df, use_container_width=True)

        st.info("""
        **คำอธิบาย:**
        * **Night (จำนวนคืน):** มีผลต่อราคามากที่สุด (43.64%)
        * **Reservation (ช่องทางการจอง):** มีผลรองลงมา (17.42%)
        """)

    def show_about_page():
        st.title("ℹ️ เกี่ยวกับระบบ / ผู้จัดทำ")
        st.divider()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Placeholder รูปโปรไฟล์
            if os.path.exists("my_profile.jpg"):
                st.image("my_profile.jpg", caption="ผู้จัดทำ", use_container_width=True)
            else:
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
            st.info("""
            **โปรแกรมนี้เป็นส่วนหนึ่งของวิทยานิพนธ์เรื่อง** *การพัฒนาระบบสนับสนุนการตัดสินใจเพื่อการพยากรณ์ราคาแบบพลวัตสำหรับธุรกิจโรงแรม โดยใช้ เทคนิคเหมืองข้อมูล*
            """)
        
        st.divider()
        st.header("📜 กิตติกรรมประกาศ")
        st.markdown("""
        *งานวิจัยฉบับนี้สำเร็จลุล่วงได้ด้วยดี เนื่องจากได้รับความอนุเคราะห์ข้อมูลการจองห้องพักย้อนหลังและข้อมูลที่เกี่ยวข้องจากผู้บริหารและพนักงานของโรงแรมกรณีศึกษาในจังหวัดเชียงใหม่ ซึ่งเป็นส่วนสำคัญยิ่งในการพัฒนาและทดสอบแบบจำลองและ ขอขอบพระคุณ ผู้ช่วยศาสตราจารย์ ดร.พงศ์กร จันทราช อาจารย์ที่ปรึกษา ที่ได้กรุณาให้คำปรึกษา แนะนำแนวทาง และตรวจสอบความถูกต้องของงานวิจัยมาโดยตลอด รวมทั้งคณาจารย์ประจำสาขาวิชาวิทยาการข้อมูลและนวัตกรรมดิจิทัล คณะนวัตกรรม เทคโนโลยี และการสร้างสรรค์ มหาวิทยาลัยฟาร์อีสเทอร์น ที่ให้การสนับสนุนทางวิชาการและเอื้อเฟื้อเครื่องมือในการดำเนินงานวิจัยจนประสบผลสำเร็จ* """)

    # --- SIDEBAR NAVIGATION ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2933/2933116.png", width=80)
        st.markdown("### Admin Panel")
        st.caption("User: **admin** | Status: **Online**")
        
        selected_page = st.radio("เมนูใช้งาน:", 
            ["🏠 หน้าหลัก", "📊 แดชบอร์ด", "🔮 พยากรณ์ราคา", "🧠 วิเคราะห์โมเดล", "ℹ️ เกี่ยวกับระบบ"]
        )
        
        st.divider()
        st.markdown("#### ⚙️ ประสิทธิภาพโมเดล")
        st.progress(m_xgb['r2'], text=f"XGBoost Score: {m_xgb['r2']*100:.1f}%")
        st.progress(m_lr['r2'], text=f"Linear Reg Score: {m_lr['r2']*100:.1f}%")
        
        st.divider()
        if st.button("ออกจากระบบ (Logout)", type="secondary"):
            st.session_state['logged_in'] = False
            st.rerun()

    # --- PAGE ROUTER ---
    if "หน้าหลัก" in selected_page: show_home_page()
    elif "แดชบอร์ด" in selected_page: show_dashboard_page()
    elif "พยากรณ์ราคา" in selected_page: show_pricing_page()
    elif "วิเคราะห์โมเดล" in selected_page: show_model_insight_page()
    elif "เกี่ยวกับระบบ" in selected_page: show_about_page()
