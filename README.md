# Student Performance ML Dashboard — Complete Code Reference

This document explains **every single line of code** in every file of this project.
No prior knowledge is assumed. Every new word, function, and concept is explained where it first appears.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [How to Run](#2-how-to-run)
3. [Project File Structure](#3-project-file-structure)
4. [app.py — Home Page](#4-apppy--home-page)
5. [train_model.py — ML Training Script](#5-train_modelpy--ml-training-script)
6. [pages/1_Data_Overview.py — Page 1](#6-pages1_data_overviewpy--page-1)
7. [pages/2_Data_Cleaning.py — Page 2](#7-pages2_data_cleaningpy--page-2)
8. [pages/3_Visualizations.py — Page 3](#8-pages3_visualizationspy--page-3)
9. [pages/4_Prediction.py — Page 4](#9-pages4_predictionpy--page-4)
10. [Key Terms Glossary](#10-key-terms-glossary)

---

## 1. Project Overview

This project does two things:

**Part 1 — Machine Learning Training (`train_model.py`)**
Reads student exam data, trains 6 different regression models, evaluates which one is best, and saves that model to disk.

**Part 2 — Interactive Web Dashboard (Streamlit app)**
A 4-page web application that lets users:
- View and explore the raw student dataset
- See how the data was cleaned and transformed
- View charts and visual analysis
- Predict a student's performance percentage by entering their details

**What is Streamlit?**
Streamlit is a Python library that converts a Python script into an interactive website. You do not need to know HTML, CSS, or JavaScript. Every `st.something()` call renders something on the web page.

**What is a regression model?**
A regression model is a mathematical function that predicts a continuous number (like a percentage score) from input features (like gender, lunch type, etc.).

---

## 2. How to Run

```bash
# Step 1: Install all required packages
pip install -r requirements.txt

# Step 2: Train the ML models (creates .pkl files)
python train_model.py

# Step 3: Launch the web dashboard
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 3. Project File Structure

```
p_1/
├── app.py                         ← Streamlit home page (entry point)
├── train_model.py                 ← Offline training script (run once)
├── pages/
│   ├── 1_Data_Overview.py         ← Page 1: raw data explorer
│   ├── 2_Data_Cleaning.py         ← Page 2: data pipeline walkthrough
│   ├── 3_Visualizations.py        ← Page 3: all charts
│   └── 4_Prediction.py            ← Page 4: predict student scores
├── StudentsPerformance_10000.csv  ← Original raw dataset (10,000 students)
├── clean_data.csv                 ← Cleaned version (duplicates removed, names fixed)
├── training_data.csv              ← Final training data (one-hot encoded, 26 columns)
├── best_model.pkl                 ← Saved best model (created by train_model.py)
├── feature_names.pkl              ← Saved column order (created by train_model.py)
├── model_results.pkl              ← Saved model scores table (created by train_model.py)
├── requirements.txt               ← List of packages with pinned versions
└── tests/
    ├── test_train_model.py        ← Tests for training pipeline
    └── test_prediction.py         ← Tests for prediction logic
```

**What is a .pkl file?**
`.pkl` stands for "pickle" — Python's way of saving any Python object (a trained model, a list, a table) to a file on disk. You can later load it back and use it as if you never closed Python.

**What is requirements.txt?**
A text file listing all Python packages this project needs, with exact version numbers pinned. `pip install -r requirements.txt` reads this file and installs everything at once.

---

## 4. `app.py` — Home Page

This is the **entry point** — the file Streamlit runs first when you type `streamlit run app.py`.

```python
import streamlit as st
```
`import` — loads a Python library so we can use its functions.
`streamlit` — the web framework library. We give it the short alias `st` so we can type `st.title(...)` instead of `streamlit.title(...)`.

```python
st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="🎓",
    layout="wide",
)
```
`st.set_page_config()` — configures the browser tab and page layout. **Must be the very first Streamlit call in the file.** If you put it after another `st.` call, you get an error.
- `page_title` — text shown in the browser tab
- `page_icon` — emoji or image shown in the browser tab
- `layout="wide"` — uses the full browser width instead of a narrow centered column

```python
st.title("🎓 Student Performance ML Dashboard")
```
`st.title()` — renders a large heading (H1 level) on the page.

```python
st.markdown("---")
```
`st.markdown()` — renders text using Markdown formatting. `"---"` in Markdown draws a horizontal dividing line.

```python
st.markdown("""
Welcome to the **Student Performance Analysis Dashboard**.
...
""")
```
Triple quotes `"""..."""` allow a multi-line string in Python.
`**bold text**` — Markdown syntax for bold.
`| Page | What it shows |` — Markdown syntax for a table.
`> text` — Markdown syntax for a blockquote (indented note box).

**Why does `app.py` exist if it just shows a welcome message?**
Streamlit requires an entry point file. It also automatically discovers every `.py` file in the `pages/` folder and adds them as sidebar navigation links. The `app.py` file becomes the "Home" page in that sidebar.

---

## 5. `train_model.py` — ML Training Script

This is **not a Streamlit page** — it is a plain Python script. Run it once from the terminal before opening the dashboard. It trains 6 models, picks the best one, and saves it to disk.

### Imports (Lines 1–9)

```python
import sys
```
`sys` — Python's built-in module for system operations. We use `sys.exit(1)` to stop the program with an error code.

```python
import pickle
```
`pickle` — Python's built-in module for serialising (saving) and deserialising (loading) Python objects to/from files.

```python
import pandas as pd
```
`pandas` — the most popular Python library for working with tabular data (like Excel spreadsheets). `pd` is the standard alias.
A **DataFrame** is pandas' main data structure — like a table with labelled rows and columns.

```python
import numpy as np
```
`numpy` — library for fast numerical computations. `np` is the standard alias. We use it for `np.sqrt()` (square root).

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
```
`sklearn` (scikit-learn) — the main Python machine learning library. We import 6 specific model classes from it.

```python
from sklearn.model_selection import train_test_split
```
`train_test_split` — a function that randomly splits a dataset into two parts: one for training the model and one for testing how good it is.

```python
from sklearn.metrics import r2_score, mean_squared_error
```
`r2_score` — computes R² (how well the model fits). Range: 0 to 1. 1.0 = perfect.
`mean_squared_error` — computes the average of squared prediction errors. We take its square root to get RMSE.

---

### Step 1 — Load Training Data

```python
try:
    df = pd.read_csv("training_data.csv")
except FileNotFoundError:
    print("Error: 'training_data.csv' not found.")
    sys.exit(1)
```

`try: ... except:` — Python's error handling. The code in the `try` block runs first. If it throws the named error (`FileNotFoundError`), execution jumps to the `except` block instead of crashing.

`pd.read_csv("training_data.csv")` — reads a CSV (Comma-Separated Values) file from disk and returns it as a pandas DataFrame. The filename is relative to wherever you run the script from.

`FileNotFoundError` — a built-in Python error that is raised when you try to open a file that does not exist.

`sys.exit(1)` — immediately stops the Python script. The number `1` is an **exit code** — `0` means success, any other number means failure. This lets other tools (like CI/CD pipelines) know the script failed.

**Why `training_data.csv` and not `StudentsPerformance_10000.csv`?**
The raw CSV has only 8 columns. We need 26 columns for the model (original scores dropped, one-hot encoded categories added). `training_data.csv` is the pre-processed version.

---

### Step 2 — Extract Target Variable

```python
y = df["percentage"]
```

`df["percentage"]` — selects a single column from the DataFrame by its name. Returns a **Series** (a single column, like a list with an index).

`y` — by convention, `y` is used for the **target variable** — the value we want the model to learn to predict.

**Why extract BEFORE dropping columns?**
In Step 3 we will drop the `percentage` column from the DataFrame. If we extracted `y` after dropping, we would get an error because the column would no longer exist. This ordering is critical.

---

### Step 3 — Build Feature Matrix

```python
COLUMNS_TO_DROP = [
    "math score", "reading score", "writing score",
    "total", "average_score", "verbal_score", "score_range",
    "percentage",
]
X = df.drop(columns=COLUMNS_TO_DROP)
```

`COLUMNS_TO_DROP` — a Python list of column names to remove. Using ALL_CAPS for the variable name is a Python convention for constants (values that never change).

`df.drop(columns=...)` — returns a new DataFrame with the listed columns removed. The original `df` is unchanged.

`X` — by convention, `X` is used for the **feature matrix** — the input columns the model learns from.

**Why drop these columns?**
- `math score`, `reading score`, `writing score` — these ARE the scores. Using them to predict `percentage` would be cheating (data leakage) because in real life you don't know them when you want to predict performance.
- `total`, `average_score`, `verbal_score`, `score_range`, `percentage` — these are all computed directly from the three scores, so same problem.

After dropping, `X` contains only the 18 one-hot encoded columns like `gender_male`, `lunch_standard`, etc.

---

### Step 4 — Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

`train_test_split()` — shuffles the data and splits it into two groups.
- `test_size=0.2` — 20% of the data goes to the test set. 80% goes to training.
- `random_state=42` — fixes the random seed so you get the same split every time you run the script. Without this, the split would be different each run.

**Why split?**
If you train and test on the same data, the model will memorise the answers (overfitting) and appear to perform perfectly. The test set contains rows the model has never seen — this gives an honest measure of real-world performance.

Result: 4 variables.
- `X_train` — 80% of input rows (for training)
- `X_test` — 20% of input rows (for testing)
- `y_train` — 80% of target values (matching `X_train`)
- `y_test` — 20% of target values (matching `X_test`)

---

### Step 5 — `select_best_model()` Function

```python
def select_best_model(results: list[dict]) -> dict:
    return max(results, key=lambda rec: (rec["r2"], -rec["rmse"]))
```

`def` — keyword to define a function.
`results: list[dict]` — a **type hint** saying the parameter should be a list of dictionaries. Python does not enforce this, but it documents what the function expects.
`-> dict` — type hint saying this function returns a dictionary.

`max(results, key=...)` — finds the item in `results` with the largest value according to the `key` function.

`lambda rec: (rec["r2"], -rec["rmse"])` — a **lambda** is an anonymous (nameless) one-line function.
- `rec` — the current dictionary being evaluated
- `(rec["r2"], -rec["rmse"])` — a **tuple** used as the comparison key
- Python compares tuples left to right: first compare R², then (if tied) compare -RMSE
- `-rec["rmse"]` — the negative sign converts "lowest RMSE wins" into "highest -RMSE wins", so `max()` does the right thing

**What does this function do in plain English?**
Given a list of model results, return the result with the highest R² score. If two models have the same R², return the one with the lower RMSE.

---

### Step 6 — Model Definitions

```python
models = [
    LinearRegression(),
    Ridge(alpha=1.0),
    Lasso(alpha=1.0),
    RandomForestRegressor(n_estimators=100, random_state=42),
    GradientBoostingRegressor(random_state=42),
    ExtraTreesRegressor(n_estimators=100, random_state=42),
]
```

Each line creates an **instance** of a model class. Parentheses `()` call the class constructor. The arguments inside are **hyperparameters** — settings that control how the model learns.

| Model | Plain English Description |
|---|---|
| `LinearRegression()` | Finds the best straight line through the data. Simple, fast, interpretable. |
| `Ridge(alpha=1.0)` | Like linear regression but adds a penalty for large weights (prevents overfitting). `alpha` controls how strong the penalty is. |
| `Lasso(alpha=1.0)` | Like Ridge but its penalty can shrink some weights to exactly zero — effectively ignoring those features. |
| `RandomForestRegressor(n_estimators=100)` | Builds 100 decision trees, each trained on a random subset of rows and features. Predicts by averaging all 100 trees. |
| `GradientBoostingRegressor()` | Builds trees one by one, each new tree focused on correcting the errors made by all previous trees. |
| `ExtraTreesRegressor(n_estimators=100)` | Like Random Forest but each split point is chosen randomly (not optimally). Faster and often comparable in accuracy. |

**What is overfitting?**
When a model memorises the training data so well that it fails on new unseen data. Regularisation (Ridge, Lasso) adds a penalty to prevent this.

**What is a decision tree?**
A tree-like flowchart of yes/no questions about the features. At the end of each path is a predicted value.

---

### Step 7 — Training Loop

```python
results = []

for model, name in zip(models, model_names):
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2   = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results.append({"name": name, "model": model, "r2": r2, "rmse": rmse})
    except Exception as exc:
        print(f"[WARNING] Model '{name}' failed: {exc}. Skipping.")
```

`results = []` — start with an empty list to collect results.

`zip(models, model_names)` — `zip()` pairs up two lists element by element. Each iteration gives one `(model_object, model_name_string)` pair.

`model.fit(X_train, y_train)` — **trains** the model. The model looks at all the training rows and adjusts its internal parameters to minimise prediction error. After this call, the model has "learned" from the data.

`model.predict(X_test)` — generates predictions for the test rows. The model has never seen these rows before. Returns an array of predicted percentages.

`r2_score(y_test, y_pred)` — computes R². Compares actual values (`y_test`) against predictions (`y_pred`).

`mean_squared_error(y_test, y_pred)` — computes MSE (mean squared error). `np.sqrt()` converts it to RMSE.

**Why sqrt?**
MSE is in "squared percentage points" which is hard to interpret. RMSE brings it back to "percentage points" — the same unit as the original target, making it human-readable.

`results.append({...})` — adds a dictionary with the model's name, object, R², and RMSE to the results list.

`except Exception as exc:` — catches ANY error. If a model fails for any reason, we print a warning and skip it rather than crashing the whole script.

---

### Step 8 — Print Comparison Table

```python
print(f"{'Model':<25} {'R²':>10} {'RMSE':>10}")
```

`f"..."` — an **f-string** (formatted string literal). Variables and expressions inside `{}` are replaced with their values.

`{'Model':<25}` — left-align (`<`) the text "Model" in a field 25 characters wide.
`{'R²':>10}` — right-align (`>`) "R²" in a field 10 characters wide.

This creates neat columns when printed to the terminal.

---

### Steps 9–12 — Save Everything

```python
best_result = select_best_model(results)
best_model = best_result["model"]
```
Calls our function to pick the winner. Extracts the actual model object.

```python
feature_names = X.columns.tolist()
```
`X.columns` — the column names of the feature DataFrame (returns an Index object).
`.tolist()` — converts it to a plain Python list.

We save this because the prediction page needs to know the exact column order to send input to the model correctly.

```python
model_results = pd.DataFrame(
    [{"model": rec["name"], "r2": rec["r2"], "rmse": rec["rmse"]} for rec in results]
)
```
**List comprehension** — `[expression for item in list]` — creates a new list by applying an expression to every item. Here it creates a list of simplified dictionaries (dropping the model object, keeping only name/R²/RMSE). `pd.DataFrame()` converts that list of dicts into a table.

```python
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
```
`open("filename", "wb")` — opens a file for writing in binary mode (`"wb"` = write binary).
`with ... as f:` — a **context manager**. Automatically closes the file when the block ends, even if an error occurs.
`pickle.dump(object, file)` — serialises `object` and writes the binary data to `file`.

The same pattern saves `feature_names` and `model_results`.

---

## 6. `pages/1_Data_Overview.py` — Page 1

Streamlit automatically turns every file in the `pages/` folder into a sidebar link. The number prefix `1_` controls the order.

### Imports

```python
import os
import pandas as pd
import streamlit as st
```
`os` — Python's operating system module. We use `os.path.exists()` to check if a file exists before trying to open it.

### File Existence Check

```python
RAW_CSV = "StudentsPerformance_10000.csv"

if not os.path.exists(RAW_CSV):
    st.error(f"File not found: **{RAW_CSV}**.")
    st.stop()
```

`RAW_CSV = "..."` — a constant storing the filename. Using a constant means if the filename ever changes, you only need to update it in one place.

`os.path.exists(path)` — returns `True` if the file exists, `False` if not.

`not os.path.exists(...)` — `not` flips True/False. So this reads: "if the file does NOT exist".

`st.error(message)` — renders a red error box on the page.

`st.stop()` — **immediately halts** all further rendering of this page. Nothing below this line runs. This is used instead of `return` because Streamlit pages are scripts, not functions — they do not have a `return` statement.

### Load and Fix Data Types

```python
df = pd.read_csv(RAW_CSV)

for _col in ["math score", "reading score", "writing score"]:
    df[_col] = pd.to_numeric(df[_col], errors="coerce")
```

`for _col in [...]` — loops over the list. The underscore prefix `_col` is a convention meaning "this is a loop variable I don't want to confuse with other variables".

`pd.to_numeric(series, errors="coerce")` — converts a column to numeric (integer or float).
- `errors="coerce"` — if a value cannot be converted (e.g. a stray text cell), replace it with `NaN` (Not a Number) instead of crashing.

**Why is this needed?**
CSV files store everything as text. Sometimes pandas reads numeric columns as strings. Matplotlib's histogram function requires actual numbers — passing strings causes a `TypeError`. This line guarantees the columns are numeric.

### Dataset Dimensions

```python
rows, cols = df.shape
st.write(f"**Rows:** {rows:,}   |   **Columns:** {cols}")
```

`df.shape` — returns a tuple `(number_of_rows, number_of_columns)`.
`rows, cols = df.shape` — **tuple unpacking** — assigns both values in one line.
`{rows:,}` — the `:,` format specifier adds thousands separators (e.g. 10000 → 10,000).

`st.write()` — a general-purpose Streamlit function. It renders text (with Markdown), numbers, DataFrames, etc. automatically depending on what you pass it.

### First 10 Rows

```python
st.dataframe(df.head(10), use_container_width=True)
```

`df.head(10)` — returns the first 10 rows of the DataFrame.
`st.dataframe()` — renders an interactive scrollable table in the browser.
`use_container_width=True` — stretches the table to fill the full width of the page.

### Descriptive Statistics

```python
numeric_cols = ["math score", "reading score", "writing score"]
desc_stats = df[numeric_cols].agg(["mean", "min", "max", "std"]).round(2)
desc_stats.index = ["Mean", "Min", "Max", "Std"]
```

`df[list_of_columns]` — selects multiple columns at once. Returns a DataFrame.

`.agg(["mean", "min", "max", "std"])` — **aggregate** function. Applies multiple statistical functions at once to each column.
- `mean` — average value
- `min` — smallest value
- `max` — largest value
- `std` — **standard deviation** — how spread out the values are. A low std means most values are close to the mean; a high std means they are spread widely.

`.round(2)` — rounds all values to 2 decimal places.

`desc_stats.index = ["Mean", "Min", "Max", "Std"]` — replaces the default row labels (`mean`, `min`, etc.) with nicer capitalised labels.

### Categorical Value Counts

```python
total = len(df)

for col in categorical_cols:
    value_counts = df[col].value_counts().reset_index()
    value_counts.columns = ["Value", "Count"]
    value_counts["Percentage (%)"] = round((value_counts["Count"] / total) * 100, 2)
    st.dataframe(value_counts, use_container_width=True, hide_index=True)
```

`len(df)` — number of rows in the DataFrame.

`df[col].value_counts()` — counts how many times each unique value appears in the column. Returns a Series with values as the index and counts as the data.

`.reset_index()` — converts the index (the unique values) into a regular column, giving a two-column DataFrame.

`.columns = ["Value", "Count"]` — renames the two columns.

`(value_counts["Count"] / total) * 100` — divides each count by total rows and multiplies by 100 to get a percentage.

`round(..., 2)` — rounds to 2 decimal places.

`hide_index=True` — hides the row numbers (0, 1, 2...) from the displayed table.

### Missing Values

```python
missing = df.isnull().sum()

if missing.sum() == 0:
    st.success("No missing values found.")
else:
    missing_df = missing.reset_index()
    missing_df.columns = ["Column", "Missing Count"]
    missing_df = missing_df[missing_df["Missing Count"] > 0]
    st.dataframe(missing_df, ...)
```

`df.isnull()` — returns a DataFrame of the same shape filled with `True` (where a cell is NaN/empty) or `False`.

`.sum()` — when called on a boolean DataFrame, counts `True` values per column. Returns a Series: column name → count of missing values.

`missing.sum()` — sums all column counts. If this is 0, there are no missing values anywhere.

`st.success(message)` — renders a green success box.

`missing_df[missing_df["Missing Count"] > 0]` — **boolean indexing** — filters the DataFrame to keep only rows where `Missing Count` is greater than 0 (i.e., only columns that actually have missing values).

---

## 7. `pages/2_Data_Cleaning.py` — Page 2

### Loading Three Files in a Loop

```python
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
        st.error(f"File not found: **{path}**.")
        st.stop()
```

`if ... elif ... else` — checks conditions in order. Only the first matching block runs.

All three files are needed for this page. If any one is missing, `st.stop()` halts the page.

### Numbered Steps List

```python
steps = ["Standardized column names...", "Removed duplicate rows...", ...]

for i, step in enumerate(steps, start=1):
    st.markdown(f"**{i}.** {step}")
```

`enumerate(iterable, start=1)` — loops over a list and also gives a counter. `start=1` makes the counter start at 1 instead of 0.

`st.markdown(f"**{i}.** {step}")` — renders each step as a numbered bold markdown item.

### Shape Comparison

```python
shape_data = {
    "Dataset": ["Raw ...", "Clean ...", "Training ..."],
    "Rows": [raw_df.shape[0], clean_df.shape[0], train_df.shape[0]],
    "Columns": [raw_df.shape[1], clean_df.shape[1], train_df.shape[1]],
}
st.dataframe(pd.DataFrame(shape_data), use_container_width=True, hide_index=True)
```

`df.shape[0]` — number of rows (index 0 of the tuple).
`df.shape[1]` — number of columns (index 1 of the tuple).

`pd.DataFrame(dict)` — creates a DataFrame from a dictionary. Keys become column names; values (lists) become column data.

### Detecting One-Hot Columns

```python
one_hot_cols = [c for c in train_df.columns if "_" in c and c not in [...]]
```

**List comprehension** — creates a list by filtering.
`"_" in c` — checks if the string `"_"` appears anywhere in the column name.
`c not in [...]` — excludes numeric engineered columns that also happen to have underscores.

One-hot columns follow the naming pattern `{original_column}_{value}` — e.g. `gender_male`, `lunch_standard`. They all contain `_`.

### Histogram with Matplotlib

```python
fig, ax = plt.subplots()
ax.hist(train_df["percentage"], bins=20, color="steelblue", edgecolor="white")
ax.set_xlabel("Percentage Score")
ax.set_ylabel("Number of Students")
ax.set_title("Distribution of Student Percentage Scores")
st.pyplot(fig)
plt.close(fig)
```

`plt.subplots()` — creates a new **figure** (`fig`) and **axes** (`ax`). A figure is the whole image; axes is the plotting area inside it. Always create a new `fig, ax` for each chart to prevent one chart's settings leaking into the next.

`ax.hist(data, bins=20)` — draws a histogram. Divides the range of values into 20 equal-width buckets (bins) and counts how many values fall in each bucket.

`color="steelblue"` — fills the bars with the named colour.
`edgecolor="white"` — draws white borders between bars, making them easier to distinguish.

`ax.set_xlabel()`, `ax.set_ylabel()`, `ax.set_title()` — label the x-axis, y-axis, and chart title.

`st.pyplot(fig)` — renders the matplotlib figure inside Streamlit. **Never use `plt.show()`** in Streamlit — it either does nothing or opens a popup window.

`plt.close(fig)` — frees the memory used by the figure. Without this, figures accumulate and cause memory warnings.

---

## 8. `pages/3_Visualizations.py` — Page 3

### Univariate Analysis

**Univariate** means analysing one variable at a time.

```python
fig, ax = plt.subplots()
ax.hist(df["math score"], bins=20, color="steelblue", edgecolor="white")
ax.set_title("Distribution of Math Score")
st.pyplot(fig)
plt.close(fig)
```
Same pattern for reading score and writing score with different colours.

```python
gender_counts = df["gender"].value_counts()
fig, ax = plt.subplots()
ax.bar(gender_counts.index, gender_counts.values, color="cornflowerblue", edgecolor="white")
```

`df["gender"].value_counts()` — counts occurrences of "male" and "female". Returns a Series where the index is `["female", "male"]` and values are counts.

`ax.bar(x, heights)` — draws a bar chart.
- `gender_counts.index` — the category labels (x positions)
- `gender_counts.values` — the bar heights

```python
race_counts = df["race/ethnicity"].value_counts().sort_index()
```
`.sort_index()` — sorts the Series alphabetically by its index (group A, B, C...) rather than by count.

```python
plt.xticks(rotation=15, ha="right")
```
`plt.xticks(rotation=15)` — rotates the x-axis tick labels by 15 degrees so they don't overlap.
`ha="right"` — horizontal alignment. Aligns the right edge of each label to its tick mark.

### Bivariate Analysis

**Bivariate** means analysing the relationship between two variables.

```python
df.boxplot(
    column=["math score", "reading score", "writing score"],
    by="test preparation course",
    ax=ax,
)
plt.suptitle("")
```

`df.boxplot(column=..., by=...)` — pandas built-in box plot. Groups the data by the `by` column and draws side-by-side box plots for each group.

**Box plot explained:**
- The box shows the middle 50% of values (25th to 75th percentile — the IQR, interquartile range)
- The line inside the box is the median (middle value)
- The whiskers extend to 1.5× the IQR
- Dots beyond the whiskers are outliers

`plt.suptitle("")` — pandas automatically adds a title "Boxplot grouped by ..." above the chart. This line removes it by setting it to empty string, keeping only the title we set manually.

### Multivariate Analysis

**Multivariate** means analysing relationships between 3 or more variables at once.

#### Correlation Heatmap

```python
corr_matrix = df[["math score", "reading score", "writing score"]].corr()
sns.heatmap(corr_matrix, annot=True, ax=ax, cmap="coolwarm", fmt=".2f")
```

`.corr()` — computes the **Pearson correlation coefficient** between every pair of columns. Returns a 3×3 matrix of values from -1 to +1.
- `1.0` = perfect positive relationship (when one goes up, the other always goes up)
- `0.0` = no relationship
- `-1.0` = perfect negative relationship

`sns.heatmap()` — draws the correlation matrix as a colour-coded grid. `seaborn` (imported as `sns`) is a statistical visualisation library built on top of matplotlib.

`annot=True` — prints the correlation number inside each cell.
`cmap="coolwarm"` — colour map: blue for low values (negative), red for high values (positive).
`fmt=".2f"` — format string for the annotations: 2 decimal places, floating point.

#### Scatter Plot

```python
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
ax.legend(title="Gender")
```

`df.groupby("gender")` — splits the DataFrame into sub-DataFrames, one per unique gender value. Each iteration gives `(gender_value, sub_dataframe)`.

`gender_colors.get(gender_val, "gray")` — looks up the colour for this gender. If not found, defaults to "gray".

`ax.scatter(x, y)` — draws individual dots at each (x, y) coordinate.
`alpha=0.5` — makes dots 50% transparent so overlapping points are visible as a darker colour.
`s=15` — dot size (in points²).
`label=gender_val` — assigns a label for the legend.

`ax.legend(title="Gender")` — renders the legend box showing which colour is which gender.

#### Grouped Bar Chart

```python
avg_by_edu = (
    df.groupby("parental level of education")[
        ["math score", "reading score", "writing score"]
    ]
    .mean()
    .sort_index()
)
fig, ax = plt.subplots()
avg_by_edu.plot(kind="bar", ax=ax, edgecolor="white")
```

`df.groupby("parental level of education")[...]` — groups by education level and selects only the three score columns.

`.mean()` — calculates the average of each column per group.

`.sort_index()` — sorts alphabetically by education level.

`avg_by_edu.plot(kind="bar", ax=ax)` — pandas DataFrames have a built-in `.plot()` method. `kind="bar"` creates a grouped bar chart with one bar per score per education level.

#### Advanced Charts (from Notebook)

```python
df["total"] = df[["math score", "reading score", "writing score"]].sum(axis=1)
```
`.sum(axis=1)` — sums across columns (axis=1 means "sum horizontally per row"). This adds all three scores for each student.

```python
lunch_gender = df.groupby(["lunch", "gender"], as_index=False)["total"].mean()
sns.barplot(data=lunch_gender, x="lunch", y="total", hue="gender", ax=ax)
```
`groupby([col1, col2])` — groups by two columns simultaneously.
`as_index=False` — keeps the group columns as regular columns instead of making them the index.
`sns.barplot(hue="gender")` — draws grouped bars, one colour per gender value.

```python
race_gender_melted = race_gender_scores.melt(
    id_vars=["race/ethnicity", "gender"],
    var_name="Subject",
    value_name="Average Score",
)
```

`.melt()` — converts a **wide format** table into a **long format** table.

**Wide format (before):**

| race/ethnicity | gender | math score | reading score | writing score |
|---|---|---|---|---|
| group A | female | 65.2 | 70.1 | 68.3 |

**Long format (after melt):**

| race/ethnicity | gender | Subject | Average Score |
|---|---|---|---|
| group A | female | math score | 65.2 |
| group A | female | reading score | 70.1 |
| group A | female | writing score | 68.3 |

`id_vars` — columns to keep as-is (not melted).
`var_name` — name for the new column that holds the old column names.
`value_name` — name for the new column that holds the values.

Seaborn's `barplot` with `hue` works best with long-format data.

---

## 9. `pages/4_Prediction.py` — Page 4

### `get_grade(percentage)` Function

```python
def get_grade(percentage: float) -> str:
    if percentage >= 90:
        return "A"
    if percentage >= 75:
        return "B"
    if percentage >= 60:
        return "C"
    if percentage >= 50:
        return "D"
    return "F"
```

`percentage: float` — type hint: expects a decimal number.
`-> str` — type hint: returns a string.

The function uses a cascade of `if` statements. Python evaluates them top to bottom and returns as soon as one matches. The final `return "F"` has no `if` — it is the default case, reached only when none of the conditions above matched.

**Grade boundaries:**
| Grade | Condition |
|---|---|
| A | 90 or above |
| B | 75 to 89.99 |
| C | 60 to 74.99 |
| D | 50 to 59.99 |
| F | below 50 |

---

### `encode_single_input(selections, feature_names)` Function

```python
def encode_single_input(selections: dict, feature_names: list) -> pd.DataFrame:
    row_dict = {name: 0 for name in feature_names}

    for column, value in selections.items():
        one_hot_key = f"{column}_{value}"
        if one_hot_key in row_dict:
            row_dict[one_hot_key] = 1

    return pd.DataFrame([row_dict])[feature_names]
```

**Why is this function needed?**
The ML model was trained on one-hot encoded data — 18 binary (0/1) columns like `gender_male`, `lunch_standard`, etc. When a user picks "male" from the dropdown, we need to convert that into a row with `gender_male=1` and `gender_female=0`.

`row_dict = {name: 0 for name in feature_names}` — **dictionary comprehension**. Creates a dictionary with all 18 feature names as keys and 0 as every value. This is our starting "blank row".

`selections.items()` — returns pairs of `(key, value)` from the dictionary. E.g. `("gender", "male")`.

`f"{column}_{value}"` — builds the one-hot column name. E.g. `column="gender"`, `value="male"` → `"gender_male"`.

`if one_hot_key in row_dict:` — checks if this key exists in our feature names. If the user somehow provides an unexpected value, we just ignore it (no crash).

`row_dict[one_hot_key] = 1` — sets that specific column to 1.

`pd.DataFrame([row_dict])` — wraps the dictionary in a list `[...]` and converts to a single-row DataFrame. The `[...]` is required because DataFrame expects a list of rows.

`[feature_names]` — reorders the columns to exactly match the order the model was trained on. This is critical — models are very sensitive to column order.

---

### Loading Pickle Files

```python
try:
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Required file 'best_model.pkl' not found. Please run `python train_model.py` first.")
    st.stop()
```

`open("filename", "rb")` — opens for reading in binary mode (`rb` = read binary).
`pickle.load(f)` — reads the binary data and reconstructs the Python object exactly as it was when saved.

The same pattern loads `feature_names.pkl` and `model_results.pkl`. The difference:
- `best_model.pkl` and `feature_names.pkl` are **required** — without them we cannot predict. Missing them calls `st.stop()`.
- `model_results.pkl` is **optional** — without it we just hide the comparison table. Missing it calls `st.warning()` but does NOT stop the page.

```python
if model_results_df is not None:
    best_model_name = model_results_df.loc[model_results_df["r2"].idxmax(), "model"]
else:
    best_model_name = "Unknown"
```

`model_results_df.loc[row_selector, column_name]` — selects a specific cell. `.loc` uses labels (not positions).

`model_results_df["r2"].idxmax()` — finds the **index** (row number) of the maximum R² value.

So `model_results_df.loc[model_results_df["r2"].idxmax(), "model"]` reads: "from the row with the highest R², give me the value in the 'model' column".

---

### Model Results Table with "(Best)" Label

```python
display_df = model_results_df.copy()
display_df["model"] = display_df["model"].apply(
    lambda name: f"{name} (Best)" if name == best_model_name else name
)
display_df["r2"] = display_df["r2"].round(4)
display_df["rmse"] = display_df["rmse"].round(4)
st.dataframe(display_df, use_container_width=True, hide_index=True)
```

`.copy()` — creates an independent copy. If we modified `model_results_df` directly, the change would persist across reruns in Streamlit's session. Using a copy prevents this.

`.apply(lambda name: ...)` — applies the lambda function to every value in the column.
`f"{name} (Best)" if name == best_model_name else name` — a **conditional expression** (ternary). Reads: "if name equals best_model_name, add ' (Best)'; otherwise keep it unchanged".

`.round(4)` — round to 4 decimal places for clean display.

---

### Single Prediction Form

```python
GENDER_OPTIONS = ["female", "male"]
RACE_OPTIONS = ["group A", "group B", "group C", "group D", "group E", "group F"]
EDUCATION_OPTIONS = ["associate's degree", "bachelor's degree", ...]
LUNCH_OPTIONS = ["free/reduced", "standard"]
PREP_OPTIONS = ["completed", "none"]
```
These lists define the valid values for each dropdown. They match exactly the values present in the original dataset.

```python
col1, col2 = st.columns(2)

with col1:
    sel_gender = st.selectbox("Gender", GENDER_OPTIONS)
    sel_race = st.selectbox("Race/Ethnicity", RACE_OPTIONS)
    sel_education = st.selectbox("Parental Level of Education", EDUCATION_OPTIONS)

with col2:
    sel_lunch = st.selectbox("Lunch", LUNCH_OPTIONS)
    sel_prep = st.selectbox("Test Preparation Course", PREP_OPTIONS)
```

`st.columns(2)` — splits the page into 2 side-by-side columns. Returns a list of column objects.
`with col1:` — anything rendered inside this block appears in column 1.

`st.selectbox("Label", options)` — renders a dropdown menu. Returns the currently selected value as a string.

```python
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
```

`st.button("Predict")` — renders a clickable button. Returns `True` on the run when clicked, `False` otherwise.

`if st.button(...)` — the whole block inside only executes when the button is clicked.

`encode_single_input(selections, feature_names)` — calls our helper function to convert the dropdown selections into the 18-column one-hot DataFrame the model expects.

`model.predict(input_df)` — runs the model and returns an array (even for a single row).
`[0]` — gets the first (and only) element from the array — a single float number.

```python
st.metric("Predicted Percentage", f"{predicted_pct:.2f}%")
st.metric("Grade", grade)
```
`st.metric(label, value)` — renders a large bold number with a label. Used for prominent key figures.
`f"{predicted_pct:.2f}%"` — formats to 2 decimal places and adds a `%` sign.

```python
summary_df = pd.DataFrame(
    list(selections.items()),
    columns=["Feature", "Selected Value"],
)
st.dataframe(summary_df, use_container_width=True, hide_index=True)
```

`selections.items()` — returns a view of `(key, value)` pairs from the dict.
`list(...)` — converts the view to a list of tuples: `[("gender", "male"), ("lunch", "standard"), ...]`
`pd.DataFrame(list_of_tuples, columns=[...])` — converts the list of pairs into a 2-column table.

---

### Bulk Prediction

```python
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
```

`st.download_button()` — renders a button that triggers a file download in the browser.
`sample_data.to_csv(index=False)` — converts the DataFrame to CSV text. `index=False` omits the row numbers column.
`mime="text/csv"` — MIME type tells the browser what kind of file this is so it saves with the right application.

```python
uploaded_file = st.file_uploader("Upload a CSV file for bulk prediction", type=["csv"])

if uploaded_file is not None:
    bulk_df = pd.read_csv(uploaded_file)
```

`st.file_uploader()` — renders a file upload widget. Returns `None` if no file is uploaded, or a file-like object when a file is chosen.
`type=["csv"]` — restricts accepted file types to CSV only.
`if uploaded_file is not None:` — only process after a file is actually uploaded.

```python
missing_cols = [col for col in REQUIRED_COLS if col not in bulk_df.columns]
if missing_cols:
    st.error(f"Uploaded CSV is missing required columns: {missing_cols}")
```

`[col for col in REQUIRED_COLS if col not in bulk_df.columns]` — list comprehension that finds which required columns are absent from the uploaded file.
`if missing_cols:` — evaluates to `True` if the list is non-empty (any columns are missing).

```python
encoded = pd.get_dummies(bulk_df[REQUIRED_COLS])
encoded = encoded.reindex(columns=feature_names, fill_value=0)
```

`pd.get_dummies(df)` — automatically one-hot encodes all columns that contain strings. For example, the `gender` column becomes `gender_female` and `gender_male`.

**Problem:** `pd.get_dummies` only creates columns for values that actually appear in the uploaded file. If the file has no "group F" students, the `race/ethnicity_group F` column will be missing.

`.reindex(columns=feature_names, fill_value=0)` — fixes this by reordering and adding any missing columns.
- `columns=feature_names` — sets the exact columns we need, in the exact order.
- `fill_value=0` — any column that was missing gets filled with 0.

```python
predictions = model.predict(encoded)
results_df = bulk_df.copy()
results_df["predicted_percentage"] = [round(float(p), 2) for p in predictions]
results_df["grade"] = results_df["predicted_percentage"].apply(get_grade)
```

`model.predict(encoded)` — predicts for all rows at once. Returns an array.
`[round(float(p), 2) for p in predictions]` — list comprehension converting each prediction to a float rounded to 2dp.
`results_df["predicted_percentage"].apply(get_grade)` — applies our `get_grade` function to every row's percentage.

```python
st.download_button(
    label="Download Results CSV",
    data=results_df.to_csv(index=False),
    file_name="bulk_predictions.csv",
    mime="text/csv",
)
```
Provides a download button for the complete results table with predictions appended.

---

## 10. Key Terms Glossary

| Term | Explanation |
|---|---|
| **DataFrame** | A table with labelled rows and columns. The main data structure in pandas. Like an Excel sheet in Python. |
| **Series** | A single column from a DataFrame. Like a list with an index. |
| **CSV** | Comma-Separated Values. A plain text file format where each row is a line and columns are separated by commas. |
| **ML / Machine Learning** | Teaching a computer to find patterns in data and make predictions without being explicitly programmed with rules. |
| **Regression** | An ML task where the goal is to predict a continuous number (like a percentage score), as opposed to classification which predicts a category. |
| **Feature** | A single measurable input variable used by the model (e.g. `gender_male`, `lunch_standard`). |
| **Target variable** | The value the model tries to predict. In this project: `percentage`. |
| **Feature matrix (X)** | The table of all input features. One row per student, one column per feature. |
| **Train/Test split** | Dividing the dataset: ~80% for training (model learns from it), ~20% for testing (model is evaluated on unseen data). |
| **Overfitting** | When a model memorises training data too well and fails on new data. Like a student who memorises answers instead of understanding concepts. |
| **R² (R-squared)** | A score from 0 to 1 measuring how well the model's predictions match reality. 1.0 = perfect. |
| **RMSE** | Root Mean Squared Error — average prediction error in the same units as the target. Lower is better. |
| **One-hot encoding** | Converting a categorical column (e.g. `gender`) into multiple binary columns (`gender_male=1, gender_female=0`). Required because ML models work with numbers, not text. |
| **Data leakage** | Accidentally using information in training that would not be available at prediction time, causing falsely good results. |
| **Pickle** | Python's built-in format for saving any Python object to a file and loading it back later. |
| **Hyperparameter** | A setting you choose before training (like `n_estimators=100`), as opposed to parameters the model learns during training. |
| **Regularisation** | A technique (used in Ridge and Lasso) that penalises complex models to prevent overfitting. |
| **Lambda function** | A short anonymous function written on one line. `lambda x: x * 2` means "a function that takes x and returns x times 2". |
| **List comprehension** | A compact way to create a list: `[x*2 for x in items if x > 0]`. |
| **Dictionary comprehension** | Like list comprehension but creates a dict: `{k: 0 for k in keys}`. |
| **Tuple unpacking** | Assigning multiple variables at once: `a, b = (1, 2)` sets `a=1` and `b=2`. |
| **Context manager (with)** | `with open(...) as f:` — automatically handles setup/teardown (like closing a file) even if an error occurs. |
| **f-string** | `f"Hello {name}"` — a Python string that embeds variable values directly. The `f` prefix enables the `{}` syntax. |
| **st.stop()** | Halts Streamlit page rendering immediately. Nothing below this line is shown. Used for fatal errors. |
| **st.pyplot(fig)** | Renders a matplotlib figure in Streamlit. Always use this instead of `plt.show()`. |
| **plt.close(fig)** | Frees memory after rendering. Prevents memory warnings on pages with many charts. |
| **Correlation** | A number from -1 to +1 measuring how closely two variables move together. +1 = always move together, 0 = no relationship, -1 = always move opposite. |
| **Histogram** | A chart showing the distribution of a single variable — how many values fall into each range (bin). |
| **Box plot** | A chart showing median, quartiles, and outliers for a variable. Good for comparing distributions across groups. |
| **Scatter plot** | A chart plotting two variables as x and y coordinates. Each dot is one data point. |
| **Heatmap** | A grid chart where each cell's colour represents a value. Used here for the correlation matrix. |
| **Univariate** | Analysis of one variable at a time. |
| **Bivariate** | Analysis of the relationship between exactly two variables. |
| **Multivariate** | Analysis involving three or more variables simultaneously. |
| **pd.get_dummies()** | Automatically one-hot encodes all string columns in a DataFrame. |
| **.reindex()** | Reorders DataFrame columns and adds missing ones (filled with a default value). |
| **.apply()** | Applies a function to every element in a Series or column. |
| **.groupby()** | Groups rows by unique values in one or more columns, then allows aggregate operations on each group. |
| **.melt()** | Converts a wide-format table (many columns) to a long-format table (many rows). Used for seaborn charts. |
| **seaborn (sns)** | A statistical visualisation library built on matplotlib. Better defaults and support for grouped data. |
| **idxmax()** | Returns the index label of the maximum value in a Series. |
| **.loc[row, col]** | Selects a DataFrame cell by row label and column label. |
| **NaN** | Not a Number — Python/pandas representation of a missing or undefined numeric value. |
| **Boolean indexing** | Filtering a DataFrame by a True/False condition: `df[df["col"] > 0]` keeps only rows where "col" is positive. |
