import streamlit as st
import time
from utils import save_uploaded_data_with_cleaning, retrain_system

def show(metrics):
    st.title("📥 จัดการข้อมูล & อัปเดตโมเดล")
    st.info("ระบบจะตรวจสอบเลขห้องกับไฟล์ Master Data หากพบข้อมูล Outlier จะทำการลบทิ้งอัตโนมัติ")
    up_file = st.file_uploader("เลือกไฟล์ Booking CSV", type=['csv'])
    if up_file and st.button("💾 บันทึกข้อมูลเข้าระบบ", type="primary"):
        if save_uploaded_data_with_cleaning(up_file):
            st.success("✅ บันทึกข้อมูลเรียบร้อย!"); st.balloons(); time.sleep(2); st.rerun()
    st.divider()
    st.markdown("### 2. สั่งให้โมเดลเรียนรู้ (Retrain)")
    st.metric("Current Accuracy (R²)", f"{metrics['xgb']['r2']*100:.2f}%")
    if st.button("🚀 เริ่มกระบวนการเรียนรู้ใหม่", type="secondary"):
        success, count = retrain_system()
        if success: st.success(f"🎉 เรียนรู้ครบ {count:,} รายการ!"); time.sleep(2); st.rerun()