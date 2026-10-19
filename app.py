import streamlit as st

st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="🎓",
    layout="wide",
)

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

> **Before using the Prediction page**, run `python train_model.py` from the project folder  
> to train the models and generate the required `.pkl` files.
""")
