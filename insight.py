import streamlit as st
import pandas as pd
import plotly.express as px

def show(metrics):
    st.title("🧠 วิเคราะห์ปัจจัยโมเดล")
    imp_data = metrics.get('importance', {})
    fi_df = pd.DataFrame(list(imp_data.items()), columns=['Feature', 'Importance']).sort_values('Importance', ascending=True)
    st.plotly_chart(px.bar(fi_df, x='Importance', y='Feature', orientation='h', color='Importance'), use_container_width=True)