import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

def show_dashboard_page(df_raw):
    st.title("📊 Financial Executive Dashboard")
    if df_raw.empty: st.warning("No Data Found"); return

    # Filters
    all_years = sorted(df_raw['Year'].unique().tolist())
    year_opts = ['All'] + [str(y) for y in all_years]
    sel_year = st.selectbox("📅 Select Year", year_opts)
    
    df_filtered = df_raw.copy()
    if sel_year != 'All': df_filtered = df_filtered[df_filtered['Year'] == int(sel_year)]

    # Metrics
    k1, k2, k3 = st.columns(3)
    k1.metric("💰 Total Revenue", f"{df_filtered['Price'].sum()/1e6:.2f} M THB")
    k2.metric("📦 Total Bookings", f"{len(df_filtered):,} รายการ")
    k3.metric("🏷️ Avg. Booking Value", f"{df_filtered['Price'].mean():,.0f} THB")

    # Tabs
    tab1, tab2 = st.tabs(["💰 Financial", "📢 Channel"])
    with tab1:
        room_perf = df_filtered.groupby('Target_Room_Type')['Price'].sum().reset_index()
        fig = px.bar(room_perf, x='Target_Room_Type', y='Price', title="Revenue by Room Type")
        st.plotly_chart(fig, use_container_width=True)