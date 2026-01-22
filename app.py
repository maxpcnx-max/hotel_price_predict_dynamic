import streamlit as st
import pandas as pd
import os

# ======================
# CONFIG
# ======================
st.set_page_config(page_title="Data Manager", layout="wide")

DATA_FILE = "data.csv"

# ======================
# INIT
# ======================
if not os.path.exists(DATA_FILE):
    pd.DataFrame().to_csv(DATA_FILE, index=False)

# ======================
# DATA IO
# ======================
def load_data():
    return pd.read_csv(DATA_FILE)

def save_data(df):
    df.to_csv(DATA_FILE, index=False)

# ======================
# MERGE FUNCTION (SAFE)
# ======================
def merge_uploaded_file(uploaded_file):
    df_existing = load_data()
    df_new = pd.read_csv(uploaded_file)

    if df_existing.empty:
        merged = df_new.copy()
    else:
        # โครงสร้างต้องตรงกัน
        if set(df_existing.columns) != set(df_new.columns):
            st.error("❌ คอลัมน์ของไฟล์ใหม่ไม่ตรงกับไฟล์เดิม")
            st.write("เดิม:", list(df_existing.columns))
            st.write("ใหม่:", list(df_new.columns))
            return

        df_new = df_new[df_existing.columns]
        merged = pd.concat([df_existing, df_new], ignore_index=True)

    save_data(merged)
    st.success(f"✅ Merge สำเร็จ (+{len(df_new)} แถว)")
    st.rerun()

# ======================
# UI
# ======================
st.title("📊 Data Manager (Safe Version)")

df = load_data()

# ---------- INFO ----------
col1, col2 = st.columns(2)
col1.metric("จำนวนข้อมูลทั้งหมด", len(df))
col2.metric("จำนวนคอลัมน์", len(df.columns))

# ======================
# DATA EDITOR
# ======================
st.subheader("✏️ แก้ไขข้อมูล (Add / Edit / Delete)")

if df.empty:
    st.info("ยังไม่มีข้อมูลในระบบ")
    edited_df = st.data_editor(
        pd.DataFrame(),
        num_rows="dynamic",
        use_container_width=True
    )
else:
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True
    )

if st.button("💾 บันทึกการแก้ไข"):
    save_data(edited_df)
    st.success("บันทึกข้อมูลเรียบร้อย")
    st.rerun()

# ======================
# MERGE SECTION
# ======================
st.divider()
st.subheader("📥 Merge ไฟล์ CSV ใหม่")

uploaded_file = st.file_uploader(
    "เลือกไฟล์ CSV (โครงสร้างต้องตรงกับไฟล์เดิม)",
    type=["csv"]
)

if uploaded_file is not None:
    preview = pd.read_csv(uploaded_file)
    st.markdown("### 🔍 ตัวอย่างข้อมูลจากไฟล์ใหม่")
    st.dataframe(preview.head(10), use_container_width=True)

    if st.button("⚠️ ยืนยัน Merge (ต่อข้อมูลเข้าไป)"):
        merge_uploaded_file(uploaded_file)

# ======================
# DOWNLOAD
# ======================
st.divider()
st.subheader("⬇️ ดาวน์โหลดข้อมูล")

st.download_button(
    label="Download CSV",
    data=load_data().to_csv(index=False).encode("utf-8"),
    file_name="data.csv",
    mime="text/csv"
)
