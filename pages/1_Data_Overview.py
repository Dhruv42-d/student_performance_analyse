import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Data Overview", layout="wide")

RAW_CSV = "StudentsPerformance_10000.csv"

if not os.path.exists(RAW_CSV):
    st.error(f"File not found: **{RAW_CSV}**. Please make sure the file is in the project root directory.")
    st.stop()

df = pd.read_csv(RAW_CSV)

for col in ["math score", "reading score", "writing score"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

st.header("Data Overview")
st.write("This page shows the original, unmodified student performance dataset before any cleaning or feature engineering.")

st.subheader("Dataset Dimensions")
rows, cols = df.shape
st.write(f"**Rows:** {rows:,}   |   **Columns:** {cols}")

st.subheader("First 10 Rows")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("Descriptive Statistics")
numeric_cols = ["math score", "reading score", "writing score"]
desc_stats = df[numeric_cols].agg(["mean", "min", "max", "std"]).round(2)
desc_stats.index = ["Mean", "Min", "Max", "Std"]
st.dataframe(desc_stats, use_container_width=True)

st.subheader("Categorical Column Distributions")
categorical_cols = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
]
total = len(df)
for col in categorical_cols:
    st.write(f"**{col}**")
    value_counts = df[col].value_counts().reset_index()
    value_counts.columns = ["Value", "Count"]
    value_counts["Percentage (%)"] = round((value_counts["Count"] / total) * 100, 2)
    st.dataframe(value_counts, use_container_width=True, hide_index=True)

st.subheader("Missing Values")
missing = df.isnull().sum()
if missing.sum() == 0:
    st.success("No missing values found.")
else:
    missing_df = missing.reset_index()
    missing_df.columns = ["Column", "Missing Count"]
    missing_df = missing_df[missing_df["Missing Count"] > 0]
    st.dataframe(missing_df, use_container_width=True, hide_index=True)
