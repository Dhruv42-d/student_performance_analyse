"""
Page 3 – Visualizations
Univariate, Bivariate, and Multivariate analysis of student performance data.
Task 7.1 implements: file header, imports, CSV loading, and Univariate Analysis section.
Task 7.2 will extend this file with Bivariate and Multivariate sections.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Visualizations", layout="wide")
st.title("Visualizations")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
CSV_PATH = "StudentsPerformance_10000.csv"

try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    st.error(
        f"Required file '{CSV_PATH}' was not found. "
        "Please make sure the file is present in the project root directory."
    )
    st.stop()

# Ensure score columns are numeric (CSV may read them as strings)
for _col in ["math score", "reading score", "writing score"]:
    df[_col] = pd.to_numeric(df[_col], errors="coerce")

# ---------------------------------------------------------------------------
# Section 1 — Univariate Analysis
# ---------------------------------------------------------------------------
st.header("Univariate Analysis")

# --- Histogram: math score ---
fig, ax = plt.subplots()
ax.hist(df["math score"], bins=20, color="steelblue", edgecolor="white")
ax.set_title("Distribution of Math Score")
ax.set_xlabel("Math Score")
ax.set_ylabel("Number of Students")
st.pyplot(fig)
plt.close(fig)

# --- Histogram: reading score ---
fig, ax = plt.subplots()
ax.hist(df["reading score"], bins=20, color="darkorange", edgecolor="white")
ax.set_title("Distribution of Reading Score")
ax.set_xlabel("Reading Score")
ax.set_ylabel("Number of Students")
st.pyplot(fig)
plt.close(fig)

# --- Histogram: writing score ---
fig, ax = plt.subplots()
ax.hist(df["writing score"], bins=20, color="mediumseagreen", edgecolor="white")
ax.set_title("Distribution of Writing Score")
ax.set_xlabel("Writing Score")
ax.set_ylabel("Number of Students")
st.pyplot(fig)
plt.close(fig)

# --- Bar chart: student count by gender ---
gender_counts = df["gender"].value_counts()
fig, ax = plt.subplots()
ax.bar(gender_counts.index, gender_counts.values, color="cornflowerblue", edgecolor="white")
ax.set_title("Student Count by Gender")
ax.set_xlabel("Gender")
ax.set_ylabel("Number of Students")
st.pyplot(fig)
plt.close(fig)

# --- Bar chart: student count by race/ethnicity ---
race_counts = df["race/ethnicity"].value_counts().sort_index()
fig, ax = plt.subplots()
ax.bar(race_counts.index, race_counts.values, color="mediumpurple", edgecolor="white")
ax.set_title("Student Count by Race/Ethnicity")
ax.set_xlabel("Race/Ethnicity")
ax.set_ylabel("Number of Students")
plt.xticks(rotation=15, ha="right")
st.pyplot(fig)
plt.close(fig)

# ---------------------------------------------------------------------------
# Section 2 — Bivariate Analysis
# ---------------------------------------------------------------------------
st.header("Bivariate Analysis")

# --- Box plots: scores grouped by test preparation course ---
st.subheader("Scores by Test Preparation Course")
fig, ax = plt.subplots()
df.boxplot(
    column=["math score", "reading score", "writing score"],
    by="test preparation course",
    ax=ax,
)
ax.set_title("Score Distribution by Test Preparation Course")
plt.suptitle("")  # suppress the automatic pandas "Boxplot grouped by" title
ax.set_xlabel("Test Preparation Course")
ax.set_ylabel("Score")
st.pyplot(fig)
plt.close(fig)

# --- Box plots: scores grouped by lunch ---
st.subheader("Scores by Lunch Type")
fig, ax = plt.subplots()
df.boxplot(
    column=["math score", "reading score", "writing score"],
    by="lunch",
    ax=ax,
)
ax.set_title("Score Distribution by Lunch Type")
plt.suptitle("")  # suppress the automatic pandas "Boxplot grouped by" title
ax.set_xlabel("Lunch Type")
ax.set_ylabel("Score")
st.pyplot(fig)
plt.close(fig)

# ---------------------------------------------------------------------------
# Section 3 — Multivariate Analysis
# ---------------------------------------------------------------------------
st.header("Multivariate Analysis")

# --- Correlation heatmap ---
st.subheader("Correlation Heatmap (Math, Reading, Writing Scores)")
fig, ax = plt.subplots()
corr_matrix = df[["math score", "reading score", "writing score"]].corr()
sns.heatmap(corr_matrix, annot=True, ax=ax, cmap="coolwarm", fmt=".2f")
ax.set_title("Correlation Heatmap of Scores")
st.pyplot(fig)
plt.close(fig)

# --- Scatter plot: math score vs reading score, colored by gender ---
st.subheader("Math Score vs Reading Score by Gender")
fig, ax = plt.subplots()
gender_colors = {"female": "hotpink", "male": "steelblue"}
for gender_val, group in df.groupby("gender"):
    color = gender_colors.get(gender_val, "gray")
    ax.scatter(
        group["math score"],
        group["reading score"],
        label=gender_val,
        color=color,
        alpha=0.5,
        s=15,
    )
ax.set_title("Math Score vs Reading Score by Gender")
ax.set_xlabel("Math Score")
ax.set_ylabel("Reading Score")
ax.legend(title="Gender")
st.pyplot(fig)
plt.close(fig)

# --- Grouped bar chart: average scores by parental level of education ---
st.subheader("Average Scores by Parental Level of Education")
avg_by_edu = (
    df.groupby("parental level of education")[
        ["math score", "reading score", "writing score"]
    ]
    .mean()
    .sort_index()
)
fig, ax = plt.subplots()
avg_by_edu.plot(kind="bar", ax=ax, edgecolor="white")
ax.set_title("Average Scores by Parental Level of Education")
ax.set_xlabel("Parental Level of Education")
ax.set_ylabel("Average Score")
ax.legend(title="Score Type")
plt.xticks(rotation=30, ha="right")
st.pyplot(fig)
plt.close(fig)

# ---------------------------------------------------------------------------
# Section 4 — Multivariate Analysis (Advanced — from notebook)
# ---------------------------------------------------------------------------
st.header("Multivariate Analysis — Advanced")

# Compute total score column used across all charts
df["total"] = df[["math score", "reading score", "writing score"]].sum(axis=1)

# --- Chart 1: Lunch Type vs Average Total Score by Gender ---
st.subheader("Lunch Type vs Average Total Score by Gender")
lunch_gender = df.groupby(["lunch", "gender"], as_index=False)["total"].mean()
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=lunch_gender, x="lunch", y="total", hue="gender", ax=ax)
ax.set_title("Lunch Type vs Average Total Score by Gender")
ax.set_xlabel("Lunch Type")
ax.set_ylabel("Average Total Score")
ax.legend(title="Gender")
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.dataframe(lunch_gender.round(2), use_container_width=True, hide_index=True)

# --- Chart 2: Parental Education vs Average Total Score by Gender ---
st.subheader("Parental Education vs Average Total Score by Gender")
education_gender = df.groupby(
    ["parental level of education", "gender"], as_index=False
)["total"].mean()
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=education_gender, x="parental level of education", y="total", hue="gender", ax=ax)
ax.set_title("Parental Education vs Average Total Score by Gender")
ax.set_xlabel("Parental Level of Education")
ax.set_ylabel("Average Total Score")
plt.xticks(rotation=35, ha="right")
ax.legend(title="Gender")
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.dataframe(education_gender.round(2), use_container_width=True, hide_index=True)

# --- Chart 3: Race/Ethnicity by Gender vs Individual Scores ---
st.subheader("Race/Ethnicity vs Individual Scores by Gender")
race_gender_scores = df.groupby(
    ["race/ethnicity", "gender"], as_index=False
)[["math score", "reading score", "writing score"]].mean()
race_gender_melted = race_gender_scores.melt(
    id_vars=["race/ethnicity", "gender"],
    var_name="Subject",
    value_name="Average Score",
)
fig, ax = plt.subplots(figsize=(14, 6))
sns.barplot(data=race_gender_melted, x="race/ethnicity", y="Average Score", hue="Subject", ax=ax)
ax.set_title("Race/Ethnicity vs Individual Scores by Gender")
ax.set_xlabel("Race/Ethnicity")
ax.set_ylabel("Average Score")
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.dataframe(race_gender_scores.round(2), use_container_width=True, hide_index=True)

# --- Chart 4: Test Preparation by Parental Education vs Average Total Score ---
st.subheader("Test Preparation by Parental Education vs Average Total Score")
prep_education = df.groupby(
    ["parental level of education", "test preparation course"], as_index=False
)["total"].mean()
fig, ax = plt.subplots(figsize=(13, 6))
sns.barplot(data=prep_education, x="parental level of education", y="total", hue="test preparation course", ax=ax)
ax.set_title("Test Preparation by Parental Education vs Average Total Score")
ax.set_xlabel("Parental Level of Education")
ax.set_ylabel("Average Total Score")
plt.xticks(rotation=35, ha="right")
ax.legend(title="Test Preparation")
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.dataframe(prep_education.round(2), use_container_width=True, hide_index=True)
