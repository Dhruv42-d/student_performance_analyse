import os
import subprocess
import streamlit as st

st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="🎓",
    layout="wide",
)

if not os.path.exists("best_model.pkl"):
    with st.spinner("Training models for the first time, please wait..."):
        subprocess.run(["python", "train_model.py"], check=True)

st.title("🎓 Student Performance ML Dashboard")
st.markdown("---")

st.markdown("""
Welcome to the **Student Performance Analysis Dashboard**.

Use the **sidebar** to navigate between the four pages:

| Page | What it shows |
|---|---|
| 📊 Data Overview | Raw dataset — dimensions, sample rows, stats, value counts, missing values |
| 🧹 Data Cleaning | Cleaning pipeline, shape comparison, engineered columns, score distribution |
| 📈 Visualizations | Univariate, bivariate and multivariate charts |
| 🤖 Prediction | Predict a student's percentage and grade (single or bulk CSV upload) |
""")
