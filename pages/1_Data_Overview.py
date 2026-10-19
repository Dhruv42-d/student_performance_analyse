"""
Page 1 — Data Overview
Loads StudentsPerformance_10000.csv and displays:
  - Dataset dimensions (rows × columns)
  - First 10 rows
  - (Extended in task 5.2: descriptive stats, value counts, missing value summary)
"""

import os

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Data Overview", layout="wide")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAW_CSV = "StudentsPerformance_10000.csv"

# ---------------------------------------------------------------------------
# Load raw data
# ---------------------------------------------------------------------------
if not os.path.exists(RAW_CSV):
    st.error(
        f"File not found: **{RAW_CSV}**. "
        "Please make sure the file is in the project root directory."
    )
    st.stop()

df = pd.read_csv(RAW_CSV)

# Ensure score columns are numeric
for _col in ["math score", "reading score", "writing score"]:
    df[_col] = pd.to_numeric(df[_col], errors="coerce")

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.header("Data Overview")
st.write(
    "This page shows the original, unmodified student performance dataset "
    "before any cleaning or feature engineering."
)

# ---------------------------------------------------------------------------
# Section 1 — Dataset dimensions
# ---------------------------------------------------------------------------
st.subheader("Dataset Dimensions")

rows, cols = df.shape
st.write(f"**Rows:** {rows:,}   |   **Columns:** {cols}")

# ---------------------------------------------------------------------------
# Section 2 — First 10 rows
# ---------------------------------------------------------------------------
st.subheader("First 10 Rows")
st.dataframe(df.head(10), use_container_width=True)

# ---------------------------------------------------------------------------
# Section 3 — Descriptive Statistics (Requirements 2.5)
# ---------------------------------------------------------------------------
st.subheader("Descriptive Statistics")

numeric_cols = ["math score", "reading score", "writing score"]
desc_stats = df[numeric_cols].agg(["mean", "min", "max", "std"]).round(2)
desc_stats.index = ["Mean", "Min", "Max", "Std"]

st.dataframe(desc_stats, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 4 — Categorical Column Distributions (Requirement 2.6)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Section 5 — Missing Values (Requirements 2.7, 2.8)
# ---------------------------------------------------------------------------
st.subheader("Missing Values")

missing = df.isnull().sum()

if missing.sum() == 0:
    st.success("No missing values found.")
else:
    missing_df = missing.reset_index()
    missing_df.columns = ["Column", "Missing Count"]
    missing_df = missing_df[missing_df["Missing Count"] > 0]
    st.dataframe(missing_df, use_container_width=True, hide_index=True)
