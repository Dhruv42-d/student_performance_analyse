"""
Page 2 — Data Cleaning & Processing
Shows the transformation pipeline from raw → clean → training-ready data.

Task 6 implements the full page.
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Data Cleaning", layout="wide")
st.title("Data Cleaning & Processing")

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
RAW_CSV = "StudentsPerformance_10000.csv"
CLEAN_CSV = "clean_data.csv"
TRAIN_CSV = "training_data.csv"

# ---------------------------------------------------------------------------
# Load all three CSVs (Requirements 3.1, 3.2)
# ---------------------------------------------------------------------------
for path in [RAW_CSV, CLEAN_CSV, TRAIN_CSV]:
    try:
        if path == RAW_CSV:
            raw_df = pd.read_csv(path)
            for _col in ["math score", "reading score", "writing score"]:
                raw_df[_col] = pd.to_numeric(raw_df[_col], errors="coerce")
        elif path == CLEAN_CSV:
            clean_df = pd.read_csv(path)
        else:
            train_df = pd.read_csv(path)
            train_df["percentage"] = pd.to_numeric(train_df["percentage"], errors="coerce")
    except FileNotFoundError:
        st.error(
            f"File not found: **{path}**. "
            "Please make sure the file is in the project root directory."
        )
        st.stop()

# ---------------------------------------------------------------------------
# Section 1 — Processing steps (Requirement 3.2)
# ---------------------------------------------------------------------------
st.subheader("Processing Steps Applied")

steps = [
    "Standardized column names (lowercased, stripped whitespace).",
    "Removed duplicate rows.",
    "Handled missing values (none found in this dataset).",
    "Computed engineered numeric columns: `total` (sum of three scores), "
    "`average_score` (mean of three scores), `percentage` (scaled 0–100), "
    "`verbal_score` (average of reading and writing), `score_range` (max − min score).",
    "One-hot encoded the 5 categorical columns: `gender`, `race/ethnicity`, "
    "`parental level of education`, `lunch`, and `test preparation course`.",
    "Dropped the original 3 score columns and categorical columns from the training set.",
]

for i, step in enumerate(steps, start=1):
    st.markdown(f"**{i}.** {step}")

# ---------------------------------------------------------------------------
# Section 2 — Shape comparison (Requirement 3.3)
# ---------------------------------------------------------------------------
st.subheader("Dataset Shape Comparison")

shape_data = {
    "Dataset": ["Raw (StudentsPerformance_10000.csv)", "Clean (clean_data.csv)", "Training (training_data.csv)"],
    "Rows": [raw_df.shape[0], clean_df.shape[0], train_df.shape[0]],
    "Columns": [raw_df.shape[1], clean_df.shape[1], train_df.shape[1]],
}
st.dataframe(pd.DataFrame(shape_data), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Section 3 — First 5 rows of clean_data.csv (Requirement 3.4)
# ---------------------------------------------------------------------------
st.subheader("Clean Data — First 5 Rows")
st.dataframe(clean_df.head(5), use_container_width=True)

# ---------------------------------------------------------------------------
# Section 4 — First 5 rows of training_data.csv (Requirement 3.5)
# ---------------------------------------------------------------------------
st.subheader("Training Data — First 5 Rows")
st.dataframe(train_df.head(5), use_container_width=True)

# ---------------------------------------------------------------------------
# Section 5 — Engineered columns (Requirement 3.6)
# ---------------------------------------------------------------------------
st.subheader("Engineered & Encoded Columns")

one_hot_cols = [c for c in train_df.columns if "_" in c and c not in [
    "math score", "reading score", "writing score",
    "total", "average_score", "percentage", "verbal_score", "score_range"
]]
numeric_engineered = ["total", "average_score", "percentage", "verbal_score", "score_range"]

st.markdown("**One-hot encoded columns:**")
st.write(one_hot_cols)

st.markdown("**Numeric engineered columns:**")
st.write(numeric_engineered)

# ---------------------------------------------------------------------------
# Section 6 — Histogram of percentage (Requirement 3.7)
# ---------------------------------------------------------------------------
st.subheader("Distribution of Percentage Score")

fig, ax = plt.subplots()
ax.hist(train_df["percentage"], bins=20, color="steelblue", edgecolor="white")
ax.set_xlabel("Percentage Score")
ax.set_ylabel("Number of Students")
ax.set_title("Distribution of Student Percentage Scores")
st.pyplot(fig)
plt.close(fig)
