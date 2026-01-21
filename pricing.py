import streamlit as st
import pandas as pd
import holidays
from datetime import datetime, timedelta
from utils import get_base_price, calculate_clamped_price

def show(xgb_model, lr_model, le_room, le_res, metrics):
    st.title("🔮 ระบบพยากรณ์ราคา (Price Forecasting)")
    if xgb_model is None: st.error("❌ Model not found"); return

    with st.container(border=True):
        st.subheader("🛠️ กำหนดเงื่อนไขการจอง")
        c1, c2 = st.columns(2)
        date_range = c1.date_input("Select Dates", value=[], min_value=None)
        
        nights, checkin_date, auto_h, auto_w = 1, datetime.now(), False, False
        if len(date_range) == 2:
            checkin_date, nights = date_range[0], (date_range[1] - date_range[0]).days
            curr = checkin_date
            while curr < date_range[1]:
                if curr in holidays.Thailand(): auto_h = True
                if curr.weekday() in [5, 6]: auto_w = True
                curr += timedelta(days=1)
        
        c2.number_input("Nights", value=max(1, nights), disabled=True)
        use_holiday = c2.checkbox("รวมวันหยุดนักขัตฤกษ์", value=auto_h)
        use_weekend = c2.checkbox("รวมวันหยุดเสาร์-อาทิตย์", value=auto_w)

        c3, c4, c5 = st.columns(3)
        room_map = {"All (เลือกทั้งหมด)": "All"}
        for r in le_room.classes_:
            if pd.isna(r): continue
            room_map[f"{r} (Base: {get_base_price(r):,.0f})"] = r
        sel_room = room_map[c3.selectbox("Room Type", list(room_map.keys()))]
        
        max_g = 2 if sel_room != "All" and ("Standard" in str(sel_room) or "Deluxe" in str(sel_room)) else 4
        guests = c4.number_input(f"Guests (Max {max_g})", 1, max_g, 2)
        
        res_opts = ["All"] + list(le_res.classes_)
        sel_res = c5.selectbox("Channel", res_opts)

        if st.button("🚀 คำนวณราคา", type="primary", use_container_width=True):
            if sel_room == "All" or sel_res == "All":
                target_rooms = le_room.classes_ if sel_room == "All" else [sel_room]
                target_res = le_res.classes_ if sel_res == "All" else [sel_res]
                results = []
                for r_type in target_rooms:
                    if pd.isna(r_type): continue
                    for ch_type in target_res:
                        f_xgb, _, _ = calculate_clamped_price(xgb_model, checkin_date, max(1, nights), guests, le_room.transform([r_type])[0], le_res.transform([ch_type])[0], r_type, use_holiday, use_weekend)
                        results.append({"Room": r_type, "Channel": ch_type, "XGB Price": f_xgb})
                st.dataframe(pd.DataFrame(results), use_container_width=True)
            else:
                f_xgb, _, _ = calculate_clamped_price(xgb_model, checkin_date, max(1, nights), guests, le_room.transform([sel_room])[0], le_res.transform([sel_res])[0], sel_room, use_holiday, use_weekend)
                st.metric("⚡ XGBoost Price", f"{f_xgb:,.0f} THB")