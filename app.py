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
    df_room = pd.read_csv(FILES["room_type"])
    df_holidays = pd.read_csv(FILES["holidays"])
    # Parse holiday dates
    df_holidays['Holiday_Date'] = pd.to_datetime(df_holidays['Holiday_Date'], dayfirst=True, errors='coerce')
    return df_room, df_holidays

def load_models():
    try:
        lr = joblib.load(FILES["model_lr"])
        xg = joblib.load(FILES["model_xgb"])
        le_res = joblib.load(FILES["le_res"])
        le_room = joblib.load(FILES["le_room"])
        return lr, xg, le_res, le_room
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

df_checkin = load_data()
df_room_map, df_holidays = load_aux_data()
lr_model, xgb_model, le_res, le_room = load_models()

# --- Helper Functions ---
def is_holiday(date_obj, holiday_df):
    return 1 if date_obj in holiday_df['Holiday_Date'].values else 0

def prepare_features(date, night, adults, children, extra, room_type_str, reservation_str):
    # derived features
    date = pd.to_datetime(date)
    total_guests = adults + children + extra
    is_hol = is_holiday(date, df_holidays)
    is_weekend = 1 if date.weekday() >= 5 else 0
    month = date.month
    weekday = date.weekday()
    
    # Encoding
    try:
        res_encoded = le_res.transform([reservation_str])[0]
    except:
        res_encoded = -1 # Handle unknown
        
    try:
        # Note: model expects room type string encoded, not the ID
        room_encoded = le_room.transform([room_type_str])[0]
    except:
        room_encoded = -1

    # Feature order must match model training:
    # ['Night', 'total_guests', 'is_holiday', 'is_weekend', 'month', 'weekday', 'RoomType_encoded', 'Reservation_encoded']
    features = np.array([[night, total_guests, is_hol, is_weekend, month, weekday, room_encoded, res_encoded]])
    return features

# --- UI Layout ---
st.title("🏨 Hotel Price Prediction & Management System")

tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📂 Data Management", "📊 Visualization"])

# --- Tab 1: Prediction ---
with tab1:
    st.header("Price Prediction")
    col1, col2 = st.columns(2)
    
    with col1:
        req_date = st.date_input("Check-in Date", value=datetime.today())
        req_time = st.time_input("Check-in Time", value=datetime.now().time()) # User requested Time
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
            
            # Predict
            pred_lr = lr_model.predict(X)[0]
            pred_xgb = xgb_model.predict(X)[0]
            
            st.subheader("Prediction Results")
            c1, c2 = st.columns(2)
            c1.metric(label="Linear Regression", value=f"{pred_lr:,.2f} THB")
            c2.metric(label="XGBoost Model", value=f"{pred_xgb:,.2f} THB")
        else:
            st.error("Models not loaded.")

# --- Tab 2: Data Management ---
with tab2:
    st.header("Manage Check-in Records")
    
    # Editable Dataframe
    edited_df = st.data_editor(df_checkin, num_rows="dynamic", use_container_width=True)

    c_save, c_retrain = st.columns([1, 4])
    
    with c_save:
        if st.button("💾 Save Changes"):
            # Save back to CSV
            # Convert Date back to string format if needed or keep as standard
            edited_df.to_csv(FILES["report"], index=False)
            st.success("Data saved successfully!")
            st.cache_data.clear() # Clear cache to reload new data
            
    with c_retrain:
        if st.button("🔄 Retrain Models"):
            with st.spinner("Retraining models..."):
                try:
                    # 1. Prepare Training Data
                    train_df = edited_df.copy()
                    train_df['Date'] = pd.to_datetime(train_df['Date'], dayfirst=True)
                    
                    # Merge with room type to get string labels
                    # Ensure Room column is float/int matching
                    train_df['Room'] = pd.to_numeric(train_df['Room'], errors='coerce')
                    train_df = train_df.merge(df_room_map, on='Room', how='left')
                    
                    # Calculate Features
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
                    
                    # Define X and y
                    features = ['Night', 'total_guests', 'is_holiday', 'is_weekend', 'month', 'weekday', 'RoomType_encoded', 'Reservation_encoded']
                    X = train_df[features]
                    y = train_df['Price']
                    
                    # Train Models
                    new_lr = LinearRegression()
                    new_lr.fit(X, y)
                    
                    new_xgb = xgb.XGBRegressor(objective='reg:squarederror')
                    new_xgb.fit(X, y)
                    
                    # Save Artifacts
                    joblib.dump(new_lr, FILES["model_lr"])
                    joblib.dump(new_xgb, FILES["model_xgb"])
                    joblib.dump(le_res_new, FILES["le_res"])
                    joblib.dump(le_room_new, FILES["le_room"])
                    
                    st.success("Models retrained and saved successfully!")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"Error during retraining: {e}")

# --- Tab 3: Visualization ---
with tab3:
    st.header("Dashboard")
    
    # Metric
    total_records = len(df_checkin)
    st.metric("Total Records", total_records)
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Reservations by Channel")
        if not df_checkin.empty:
            count_res = df_checkin['Reservation'].value_counts().reset_index()
            count_res.columns = ['Channel', 'Count']
            fig1 = px.bar(count_res, x='Channel', y='Count', color='Channel')
            st.plotly_chart(fig1, use_container_width=True)
            
    with col_chart2:
        st.subheader("Price Trend")
        if not df_checkin.empty:
            # Sort by date
            df_chart = df_checkin.sort_values('Date')
            fig2 = px.line(df_chart, x='Date', y='Price', title="Price Over Time")
            st.plotly_chart(fig2, use_container_width=True)
