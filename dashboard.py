import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from utils import load_data

def show():
    st.title("📊 Financial Executive Dashboard")
    df_raw = load_data()
    if df_raw.empty: st.warning("No Data Found"); return

    with st.expander("🔎 Filter Data", expanded=True):
        f1, f2, f3 = st.columns(3)
        year_opts = ['All'] + [str(y) for y in sorted(df_raw['Year'].unique())]
        sel_year = f1.selectbox("📅 Select Year", year_opts)
        month_opts = ['All'] + [datetime(2024, m, 1).strftime('%B') for m in sorted(df_raw['month'].unique())]
        sel_month_str = f2.selectbox("🗓️ Select Month", month_opts)
        
        df_filtered = df_raw.copy()
        if sel_year != 'All': df_filtered = df_filtered[df_filtered['Year'] == int(sel_year)]
        if sel_month_str != 'All':
            df_filtered = df_filtered[df_filtered['month'] == datetime.strptime(sel_month_str, "%B").month]

    if df_filtered.empty: st.warning("⚠️ No data available for the selected filters."); return

    st.divider()
    k1, k2, k3 = st.columns(3)
    k1.metric("💰 Total Revenue", f"{df_filtered['Price'].sum()/1e6:.2f} M THB")
    k2.metric("📦 Total Bookings", f"{len(df_filtered):,} รายการ")
    k3.metric("🏷️ Avg. Booking Value", f"{df_filtered['Price'].mean():,.0f} THB")
    
    tab1, tab2, tab3 = st.tabs(["💰 Financial", "📢 Channel", "🛌 Product"])
    g_col = 'Target_Room_Type' if 'Target_Room_Type' in df_filtered.columns else 'Room'

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            perf = df_filtered.groupby(g_col).agg({'Price': 'sum', 'Night': 'sum'}).reset_index().sort_values('Price', ascending=False)
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=perf[g_col], y=perf['Price'], name="Revenue"), secondary_y=False)
            fig.add_trace(go.Scatter(x=perf[g_col], y=perf['Night'], name="Nights", mode='lines+markers'), secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            monthly = df_filtered.groupby('month').agg({'Price': 'sum', 'Room': 'count'}).reset_index()
            monthly['M_Name'] = monthly['month'].apply(lambda x: datetime(2024, int(x), 1).strftime('%b'))
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            fig2.add_trace(go.Scatter(x=monthly['M_Name'], y=monthly['Price'], name="Revenue"), secondary_y=False)
            fig2.add_trace(go.Scatter(x=monthly['M_Name'], y=monthly['Room'], name="Bookings", line=dict(dash='dot')), secondary_y=True)
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        c3, c4 = st.columns(2)
        res_rev = df_filtered.groupby('Reservation')['Price'].sum().reset_index()
        c3.plotly_chart(px.pie(res_rev, values='Price', names='Reservation', hole=0.4), use_container_width=True)
        m_res = df_filtered.groupby(['month', 'Reservation']).size().reset_index(name='Count')
        m_res['M_Name'] = m_res['month'].apply(lambda x: datetime(2024, int(x), 1).strftime('%b'))
        c4.plotly_chart(px.bar(m_res, x='M_Name', y='Count', color='Reservation'), use_container_width=True)

    with tab3:
        heatmap_data = df_filtered.groupby([g_col, 'Reservation']).size().unstack(fill_value=0)
        st.plotly_chart(px.imshow(heatmap_data, text_auto=True, color_continuous_scale='Blues'), use_container_width=True)