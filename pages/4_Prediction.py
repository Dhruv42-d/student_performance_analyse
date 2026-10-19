"""
Page 4 — Prediction
Provides single-student and bulk-upload prediction using the best trained model.

Sub-components (added across tasks 9.1 → 9.8):
  9.1  Helper functions: get_grade, encode_single_input
  9.4  Model loading + model results table
  9.6  Single prediction form
  9.8  Bulk prediction + sample CSV download
"""

import os
import pickle

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_grade(percentage: float) -> str:
    """Map a predicted percentage to a letter grade.

    Grade boundaries:
        A  — percentage >= 90
        B  — 75 <= percentage < 90
        C  — 60 <= percentage < 75
        D  — 50 <= percentage < 60
        F  — percentage < 50
    """
    if percentage >= 90:
        return "A"
    if percentage >= 75:
        return "B"
    if percentage >= 60:
        return "C"
    if percentage >= 50:
        return "D"
    return "F"


def encode_single_input(
    selections: dict,
    feature_names: list,
) -> pd.DataFrame:
    """Convert a dict of raw categorical selections to a one-hot encoded
    single-row DataFrame aligned to feature_names column order.

    Algorithm:
        1. Build a zero-filled dict keyed by every name in feature_names.
        2. For each (column, value) pair in selections, set the one-hot key
           ``{column}_{value}`` to 1.
        3. Return pd.DataFrame([row_dict])[feature_names] so that the column
           order matches feature_names exactly.

    Args:
        selections:    Mapping of raw categorical column name → selected value.
                       Example: {"gender": "male", "lunch": "standard"}
        feature_names: Ordered list of one-hot feature column names as stored
                       in feature_names.pkl.

    Returns:
        A single-row DataFrame with shape (1, len(feature_names)), values in
        {0, 1}, and columns in the same order as feature_names.
    """
    # Start with all zeros
    row_dict = {name: 0 for name in feature_names}

    # Flip the appropriate one-hot key to 1 for each selection
    for column, value in selections.items():
        one_hot_key = f"{column}_{value}"
        if one_hot_key in row_dict:
            row_dict[one_hot_key] = 1

    # Build DataFrame and enforce column order
    return pd.DataFrame([row_dict])[feature_names]

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Prediction", layout="wide")
st.title("Student Performance Prediction")

# ---------------------------------------------------------------------------
# Model loading (Requirements 5.1, 5.2)
# ---------------------------------------------------------------------------
try:
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error(
        "Required file 'best_model.pkl' not found. "
        "Please run `python train_model.py` first to generate the model files."
    )
    st.stop()

try:
    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
except FileNotFoundError:
    st.error(
        "Required file 'feature_names.pkl' not found. "
        "Please run `python train_model.py` first to generate the model files."
    )
    st.stop()

try:
    with open("model_results.pkl", "rb") as f:
        model_results_df = pickle.load(f)
except FileNotFoundError:
    st.warning("Model comparison unavailable: 'model_results.pkl' not found.")
    model_results_df = None

# ---------------------------------------------------------------------------
# Determine best model name
# ---------------------------------------------------------------------------
if model_results_df is not None:
    best_model_name = model_results_df.loc[model_results_df["r2"].idxmax(), "model"]
else:
    best_model_name = "Unknown"

# ---------------------------------------------------------------------------
# Model results table (Requirements 6.1, 6.2)
# ---------------------------------------------------------------------------
if model_results_df is not None:
    st.subheader("Model Comparison")
    display_df = model_results_df.copy()
    display_df["model"] = display_df["model"].apply(
        lambda name: f"{name} (Best)" if name == best_model_name else name
    )
    display_df["r2"] = display_df["r2"].round(4)
    display_df["rmse"] = display_df["rmse"].round(4)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Best model label (Requirement 6.3)
# ---------------------------------------------------------------------------
st.write(f"Best Model: {best_model_name}")

# ---------------------------------------------------------------------------
# Single Prediction Form (Requirements 5.3 – 5.8)
# ---------------------------------------------------------------------------
st.subheader("Single Student Prediction")

GENDER_OPTIONS = ["female", "male"]
RACE_OPTIONS = ["group A", "group B", "group C", "group D", "group E", "group F"]
EDUCATION_OPTIONS = [
    "associate's degree", "bachelor's degree", "high school",
    "master's degree", "some college", "some high school",
]
LUNCH_OPTIONS = ["free/reduced", "standard"]
PREP_OPTIONS = ["completed", "none"]

col1, col2 = st.columns(2)

with col1:
    sel_gender = st.selectbox("Gender", GENDER_OPTIONS)
    sel_race = st.selectbox("Race/Ethnicity", RACE_OPTIONS)
    sel_education = st.selectbox("Parental Level of Education", EDUCATION_OPTIONS)

with col2:
    sel_lunch = st.selectbox("Lunch", LUNCH_OPTIONS)
    sel_prep = st.selectbox("Test Preparation Course", PREP_OPTIONS)

if st.button("Predict"):
    selections = {
        "gender": sel_gender,
        "race/ethnicity": sel_race,
        "parental level of education": sel_education,
        "lunch": sel_lunch,
        "test preparation course": sel_prep,
    }
    input_df = encode_single_input(selections, feature_names)
    predicted_pct = model.predict(input_df)[0]
    grade = get_grade(predicted_pct)

    st.markdown("### Prediction Result")
    st.metric("Predicted Percentage", f"{predicted_pct:.2f}%")
    st.metric("Grade", grade)

    st.markdown("#### Input Summary")
    summary_df = pd.DataFrame(
        list(selections.items()),
        columns=["Feature", "Selected Value"],
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Bulk Prediction (Requirements 5.9 – 5.15)
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Bulk Prediction")

REQUIRED_COLS = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
]

# --- Sample CSV download (always available) ---
sample_data = pd.DataFrame([{
    "gender": "female",
    "race/ethnicity": "group B",
    "parental level of education": "bachelor's degree",
    "lunch": "standard",
    "test preparation course": "completed",
}])
st.download_button(
    label="Download Sample CSV",
    data=sample_data.to_csv(index=False),
    file_name="sample_input.csv",
    mime="text/csv",
)

# --- File uploader ---
uploaded_file = st.file_uploader("Upload a CSV file for bulk prediction", type=["csv"])

if uploaded_file is not None:
    bulk_df = pd.read_csv(uploaded_file)

    # Validate required columns
    missing_cols = [col for col in REQUIRED_COLS if col not in bulk_df.columns]
    if missing_cols:
        st.error(f"Uploaded CSV is missing required columns: {missing_cols}")
    else:
        # One-hot encode and align to feature_names
        encoded = pd.get_dummies(bulk_df[REQUIRED_COLS])
        encoded = encoded.reindex(columns=feature_names, fill_value=0)

        # Predict
        predictions = model.predict(encoded)
        results_df = bulk_df.copy()
        results_df["predicted_percentage"] = [round(float(p), 2) for p in predictions]
        results_df["grade"] = results_df["predicted_percentage"].apply(get_grade)

        # Display results
        st.subheader("Prediction Results")
        st.dataframe(results_df, use_container_width=True)

        # Download button for results
        st.download_button(
            label="Download Results CSV",
            data=results_df.to_csv(index=False),
            file_name="bulk_predictions.csv",
            mime="text/csv",
        )
