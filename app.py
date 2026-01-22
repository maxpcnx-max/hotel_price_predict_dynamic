import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import plotly.express as px

# --- Configuration & Load Data ---
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

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(FILES["report"])
        # Ensure consistent date parsing
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        return df
    except FileNotFoundError:
        return pd.DataFrame(columns=["Reservation", "Name", "Date", "Night", "Room", 
                                     "Adults", "Children", "Infants", "Extra Person", "Price"])

@st.cache_data
def load_aux_data():
    df_room = pd.DataFrame(columns=['Room', 'Room_Type'])
    df_holidays = pd.DataFrame(columns=['Holiday_Date', 'Holiday_Name'])

    try:
        df_room = pd.read_csv(FILES["room_type"])
    except FileNotFoundError:
        st.warning(f"⚠️ ไม่พบไฟล์ {FILES['room_type']}")

    try:
        df_holidays = pd.read_csv(FILES["holidays"])
        df_holidays['Holiday_Date'] = pd.to_datetime(df_holidays['Holiday_Date'], dayfirst=True, errors='coerce')
    except FileNotFoundError:
        st.warning(f"⚠️ ไม่พบไฟล์ {FILES['holidays']}")
        
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

df_checkin = load_data()
df_room_map, df_holidays = load_aux_data()
lr_model, xgb_model, le_res, le_room = load_models()

# --- Helper Functions ---
def is_holiday(date_obj, holiday_df):
    if pd.isna(date_obj): return 0
    return 1 if date_obj in holiday_df['Holiday_Date'].values else 0

def prepare_features(date, night, adults, children, extra, room_type_str, reservation_str):
    date = pd.to_datetime(date)
    total_guests = adults + children + extra
    is_hol = is_holiday(date, df_holidays)
    is_weekend = 1 if date.weekday() >= 5 else 0
    month = date.month
    weekday = date.weekday()
    
    try:
        res_encoded = le_res.transform([reservation_str])[0]
    except:
        res_encoded = -1
        
    try:
        room_encoded = le_room.transform([room_type_str])[0]
    except:
        room_encoded = -1

    features = np.array([[night, total_guests, is_hol, is_weekend, month, weekday, room_encoded, res_encoded]])
    return features

# --- UI Layout ---
st.title("🏨 Hotel Price Prediction & Management System")

tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📂 Data Management", "📊 Visualization"])

# --- Tab 1: Prediction ---
with tab1:
    st.header("Price Prediction")
    
    if lr_model is None or xgb_model is None:
        st.warning("⚠️ Models not loaded. Please go to 'Data Management' and click 'Retrain Models' first.")
    
    col1, col2 = st.columns(2)
    with col1:
        req_date = st.date_input("Check-in Date", value=datetime.today())
        req_time = st.time_input("Check-in Time", value=datetime.now().time())
        req_res = st.selectbox("Reservation Channel", le_res.classes_ if le_res else [])
        req_room = st.selectbox("Room Type", le_room.classes_ if le_room else [])

    with col2:
        req_night = st.number_input("Nights", min_value=1, value=1)
        req_adults = st.number_input("Adults", min_value=1, value=2)
        req_children = st.number_input("Children", min_value=0, value=0)
        req_extra = st.number_input("Extra Person", min_value=0, value=0)

    if st.button("Predict Price", type="primary"):
        if lr_model and xgb_model:
            X = prepare_features(req_date, req_night, req_adults, req_children, req_extra, req_room, req_res)
            try:
                pred_lr = lr_model.predict(X)[0]
                pred_xgb = xgb_model.predict(X)[0]
                
                st.subheader("Prediction Results")
                c1, c2 = st.columns(2)
                c1.metric(label="Linear Regression", value=f"{pred_lr:,.2f} THB")
                c2.metric(label="XGBoost Model", value=f"{pred_xgb:,.2f} THB")
            except Exception as e:
                st.error(f"Prediction Error: {e}")
        else:
            st.error("Models not loaded.")

# --- Tab 2: Data Management ---
with tab2:
    st.header("Manage Check-in Records")
    
    # --- New Feature: Import / Export / Reset ---
    with st.expander("🛠️ Tools: Import, Export, Reset", expanded=False):
        c_imp, c_exp, c_reset = st.columns(3)
        
        with c_imp:
            st.markdown("#### Import Data")
            uploaded_file = st.file_uploader("Upload CSV (Overwrites current data)", type="csv")
            if uploaded_file is not None:
                if st.button("⚠️ Confirm Import", help="This will replace all current data!"):
                    try:
                        new_df = pd.read_csv(uploaded_file)
                        # Basic validation
                        required_cols = ["Reservation", "Date", "Night", "Price"]
                        if all(col in new_df.columns for col in required_cols):
                            new_df.to_csv(FILES["report"], index=False)
                            st.success("Data imported successfully!")
                            st.cache_data.clear()
                            st.rerun()
                        else:
                            st.error(f"CSV must contain columns: {required_cols}")
                    except Exception as e:
                        st.error(f"Import failed: {e}")

        with c_exp:
            st.markdown("#### Export Data")
            # Convert current DF to CSV string for download
            csv_data = df_checkin.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name="check_in_report_backup.csv",
                mime="text/csv",
            )
            
        with c_reset:
            st.markdown("#### Reset View")
            if st.button("🔄 Reload / Reset View"):
                st.cache_data.clear()
                st.rerun()

    st.divider()

    # --- Editable Dataframe ---
    edited_df = st.data_editor(df_checkin, num_rows="dynamic", use_container_width=True)

    c_save, c_retrain = st.columns([1, 4])
    
    with c_save:
        if st.button("💾 Save Changes"):
            edited_df.to_csv(FILES["report"], index=False)
            st.success("Data saved to file!")
            st.cache_data.clear()
            
    with c_retrain:
        if st.button("🔄 Retrain Models"):
            with st.spinner("Retraining models..."):
                try:
                    # Prepare Data
                    train_df = edited_df.copy()
                    train_df['Date'] = pd.to_datetime(train_df['Date'], dayfirst=True, errors='coerce')
                    train_df = train_df.dropna(subset=['Date'])
                    
                    if train_df.empty:
                        st.error("No valid data to train on.")
                        st.stop()

                    train_df['Room'] = pd.to_numeric(train_df['Room'], errors='coerce')
                    train_df = train_df.merge(df_room_map, on='Room', how='left')
                    
                    train_df['total_guests'] = train_df['Adults'] + train_df['Children'] + train_df['Extra Person']
                    train_df['is_holiday'] = train_df['Date'].apply(lambda x: is_holiday(x, df_holidays))
                    train_df['is_weekend'] = train_df['Date'].apply(lambda x: 1 if x.weekday() >= 5 else 0)
                    train_df['month'] = train_df['Date'].dt.month
                    train_df['weekday'] = train_df['Date'].dt.weekday
                    
                    # Re-fit Encoders
                    le_res_new = LabelEncoder()
                    train_df['Reservation_encoded'] = le_res_new.fit_transform(train_df['Reservation'].astype(str))
                    
                    le_room_new = LabelEncoder()
                    train_df['RoomType_encoded'] = le_room_new.fit_transform(train_df['Room_Type'].astype(str))
                    
                    features_cols = ['Night', 'total_guests', 'is_holiday', 'is_weekend', 'month', 'weekday', 'RoomType_encoded', 'Reservation_encoded']
                    train_df = train_df.dropna(subset=features_cols + ['Price'])
                    
                    if len(train_df) < 5:
                         st.warning("Training data is too small.")

                    X = train_df[features_cols]
                    y = train_df['Price']
                    
                    # Train
                    new_lr = LinearRegression()
                    new_lr.fit(X, y)
                    new_xgb = xgb.XGBRegressor(objective='reg:squarederror')
                    new_xgb.fit(X, y)
                    
                    # Save
                    joblib.dump(new_lr, FILES["model_lr"])
                    joblib.dump(new_xgb, FILES["model_xgb"])
                    joblib.dump(le_res_new, FILES["le_res"])
                    joblib.dump(le_room_new, FILES["le_room"])
                    
                    st.success(f"Retrained successfully! ({len(train_df)} records)")
                    st.balloons()
                    
                    # Update session state models
                    lr_model, xgb_model, le_res, le_room = new_lr, new_xgb, le_res_new, le_room_new
                    
                except Exception as e:
                    st.error(f"Error: {e}")

# --- Tab 3: Visualization ---
with tab3:
    st.header("Dashboard")
    st.metric("Total Records", len(df_checkin))
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        if not df_checkin.empty:
            count_res = df_checkin['Reservation'].value_counts().reset_index()
            count_res.columns = ['Channel', 'Count']
            fig1 = px.bar(count_res, x='Channel', y='Count', color='Channel', title="Reservations by Channel")
            st.plotly_chart(fig1, use_container_width=True)
            
    with col_chart2:
        if not df_checkin.empty:
            df_chart = df_checkin.copy()
            df_chart['Date'] = pd.to_datetime(df_chart['Date'], dayfirst=True, errors='coerce')
            df_chart = df_chart.dropna(subset=['Date']).sort_values('Date')
            fig2 = px.line(df_chart, x='Date', y='Price', title="Price Trend Over Time")
            st.plotly_chart(fig2, use_container_width=True)
