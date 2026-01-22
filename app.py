import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import plotly.express as px

# --- Configuration ---
st.set_page_config(page_title="Hotel Price Prediction App", layout="wide")

FILES = {
    "report": "check_in_report.csv",
    "room_type": "room_type.csv",
    "holidays": "thai_holidays.csv",
    "model_lr": "lr_hotel_model.joblib",
    "model_xgb": "xgb_hotel_model.joblib",
    "le_res": "le_res.joblib",
    "le_room": "le_room.joblib"
}

# --- 1. Load Data Section ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(FILES["report"])
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        return df
    except FileNotFoundError:
        # Default Empty Structure
        return pd.DataFrame(columns=["Reservation", "Name", "Date", "Night", "Room", 
                                     "Adults", "Children", "Infants", "Extra Person", "Price"])

@st.cache_data
def load_aux_data():
    df_room = pd.DataFrame(columns=['Room', 'Room_Type'])
    df_holidays = pd.DataFrame(columns=['Holiday_Date', 'Holiday_Name'])

    try:
        df_room = pd.read_csv(FILES["room_type"])
    except FileNotFoundError: pass

    try:
        df_holidays = pd.read_csv(FILES["holidays"])
        df_holidays['Holiday_Date'] = pd.to_datetime(df_holidays['Holiday_Date'], dayfirst=True, errors='coerce')
    except FileNotFoundError: pass
        
    return df_room, df_holidays

def load_models():
    try:
        lr = joblib.load(FILES["model_lr"])
        xg = joblib.load(FILES["model_xgb"])
        le_res = joblib.load(FILES["le_res"])
        le_room = joblib.load(FILES["le_room"])
        return lr, xg, le_res, le_room
    except Exception:
        return None, None, None, None

# --- Main Execution ---
df_checkin = load_data()
df_room_map, df_holidays = load_aux_data()
lr_model, xgb_model, le_res, le_room = load_models()

# --- Helper Functions ---
def is_holiday(date_obj, holiday_df):
    if pd.isna(date_obj) or holiday_df.empty: return 0
    return 1 if date_obj in holiday_df['Holiday_Date'].values else 0

def prepare_features(date, night, adults, children, extra, room_type_str, reservation_str):
    date = pd.to_datetime(date)
    total_guests = adults + children + extra
    is_hol = is_holiday(date, df_holidays)
    is_weekend = 1 if date.weekday() >= 5 else 0
    month = date.month
    weekday = date.weekday()
    
    try: res_encoded = le_res.transform([reservation_str])[0]
    except: res_encoded = -1
    try: room_encoded = le_room.transform([room_type_str])[0]
    except: room_encoded = -1

    return np.array([[night, total_guests, is_hol, is_weekend, month, weekday, room_encoded, res_encoded]])

# --- UI Layout ---
st.title("🏨 Hotel Management Dashboard")

tab_dash, tab_data, tab_pred = st.tabs(["📊 Dashboard", "📂 Data & Retrain", "🔮 Predict Price"])

# --- Tab 1: Dashboard ---
with tab_dash:
    st.header("Business Overview")
    
    total_records = len(df_checkin)
    total_rev = df_checkin['Price'].sum() if not df_checkin.empty else 0
    
    m1, m2 = st.columns(2)
    m1.metric("Total Bookings", f"{total_records}")
    m2.metric("Total Revenue", f"{total_rev:,.0f} THB")
    
    st.divider()
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Bookings by Channel")
        if not df_checkin.empty:
            count_res = df_checkin['Reservation'].value_counts().reset_index()
            count_res.columns = ['Channel', 'Count']
            fig1 = px.bar(count_res, x='Channel', y='Count', color='Channel', text_auto=True)
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("No data available.")

    with c2:
        st.subheader("Price Trend")
        if not df_checkin.empty:
            plot_df = df_checkin.dropna(subset=['Date']).sort_values('Date')
            fig2 = px.line(plot_df, x='Date', y='Price', markers=True)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No data available.")

# --- Tab 2: Data Management ---
with tab_data:
    st.header("Data Management")
    
    # --- Toolbar ---
    with st.expander("🛠️ Tools (Import / Reset / Hard Reset)", expanded=False):
        col_imp, col_res, col_hard = st.columns(3)
        
        # 1. Import Append
        with col_imp:
            st.markdown("##### Import (Append)")
            uploaded_file = st.file_uploader("Add CSV data", type="csv")
            if uploaded_file and st.button("➕ Confirm Append"):
                try:
                    new_df = pd.read_csv(uploaded_file)
                    current_df = pd.read_csv(FILES["report"]) if pd.io.common.file_exists(FILES["report"]) else pd.DataFrame()
                    combined_df = pd.concat([current_df, new_df], ignore_index=True)
                    combined_df.to_csv(FILES["report"], index=False)
                    
                    st.success(f"Added {len(new_df)} rows!")
                    st.cache_data.clear()
                    st.rerun() # <--- FORCE RELOAD
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # 2. Reload
        with col_res:
            st.markdown("##### Reload")
            if st.button("🔄 Refresh Page"):
                st.cache_data.clear()
                st.rerun()

        # 3. Hard Reset
        with col_hard:
            st.markdown("##### Factory Reset")
            if st.button("⚠️ Hard Reset Data", type="primary", help="Clear all data and reset to empty table"):
                # Write empty CSV with headers only
                empty_df = pd.DataFrame(columns=["Reservation", "Name", "Date", "Night", "Room", 
                                                 "Adults", "Children", "Infants", "Extra Person", "Price"])
                empty_df.to_csv(FILES["report"], index=False)
                
                st.warning("All data has been cleared!")
                st.cache_data.clear()
                st.rerun() # <--- FORCE RELOAD

    st.divider()

    # Editor
    edited_df = st.data_editor(df_checkin, num_rows="dynamic", use_container_width=True)

    col_save, col_train = st.columns([1, 4])
    
    with col_save:
        if st.button("💾 Save Changes"):
            edited_df.to_csv(FILES["report"], index=False)
            st.toast("Saved & Reloading...", icon="✅")
            st.cache_data.clear()
            st.rerun() # <--- FORCE RELOAD TO UPDATE DASHBOARD
            
    with col_train:
        if st.button("🚀 Retrain Models"):
            with st.spinner("Training..."):
                try:
                    # Prep Data
                    train_df = edited_df.copy()
                    train_df['Date'] = pd.to_datetime(train_df['Date'], dayfirst=True, errors='coerce')
                    train_df = train_df.dropna(subset=['Date'])
                    
                    if train_df.empty:
                        st.error("No valid data.")
                        st.stop()

                    # Fix Merge
                    train_df['Room'] = pd.to_numeric(train_df['Room'], errors='coerce')
                    if not df_room_map.empty:
                        df_room_map['Room'] = pd.to_numeric(df_room_map['Room'], errors='coerce')
                        train_df = train_df.merge(df_room_map, on='Room', how='left')
                    else:
                        train_df['Room_Type'] = 'Unknown'

                    # Features
                    train_df['total_guests'] = train_df['Adults'] + train_df['Children'] + train_df['Extra Person']
                    train_df['is_holiday'] = train_df['Date'].apply(lambda x: is_holiday(x, df_holidays))
                    train_df['is_weekend'] = train_df['Date'].apply(lambda x: 1 if x.weekday() >= 5 else 0)
                    train_df['month'] = train_df['Date'].dt.month
                    train_df['weekday'] = train_df['Date'].dt.weekday

                    # Encode
                    le_res_new = LabelEncoder()
                    train_df['Reservation_encoded'] = le_res_new.fit_transform(train_df['Reservation'].astype(str))
                    
                    le_room_new = LabelEncoder()
                    train_df['Room_Type'] = train_df['Room_Type'].fillna('Unknown')
                    train_df['RoomType_encoded'] = le_room_new.fit_transform(train_df['Room_Type'].astype(str))
                    
                    # Fit
                    features_cols = ['Night', 'total_guests', 'is_holiday', 'is_weekend', 'month', 'weekday', 'RoomType_encoded', 'Reservation_encoded']
                    train_df = train_df.dropna(subset=features_cols + ['Price'])
                    
                    X = train_df[features_cols]
                    y = train_df['Price']

                    new_lr = LinearRegression()
                    new_lr.fit(X, y)
                    new_xgb = xgb.XGBRegressor(objective='reg:squarederror')
                    new_xgb.fit(X, y)

                    joblib.dump(new_lr, FILES["model_lr"])
                    joblib.dump(new_xgb, FILES["model_xgb"])
                    joblib.dump(le_res_new, FILES["le_res"])
                    joblib.dump(le_room_new, FILES["le_room"])
                    
                    st.success(f"Retrained! ({len(train_df)} rows)")
                    # Reload models in session
                    lr_model, xgb_model = new_lr, new_xgb
                    
                except Exception as e:
                    st.error(f"Failed: {e}")

# --- Tab 3: Prediction ---
with tab_pred:
    st.header("Predict Price")
    
    if lr_model is None:
        st.warning("Please Retrain Models first.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            p_date = st.date_input("Check-in", value=datetime.today())
            p_res = st.selectbox("Channel", le_res.classes_)
            p_room = st.selectbox("Room Type", le_room.classes_)
        with c2:
            p_night = st.number_input("Nights", 1, 30, 1)
            p_adult = st.number_input("Adults", 1, 10, 2)
            p_child = st.number_input("Children", 0, 10, 0)
            p_extra = st.number_input("Extra", 0, 5, 0)
            
        if st.button("Predict"):
            X_pred = prepare_features(p_date, p_night, p_adult, p_child, p_extra, p_room, p_res)
            try:
                st.success(f"LR: {lr_model.predict(X_pred)[0]:,.0f} | AI: {xgb_model.predict(X_pred)[0]:,.0f} THB")
            except: st.error("Prediction failed")
