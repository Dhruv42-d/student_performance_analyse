import sys
import pickle

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ---------------------------------------------------------------------------
# Step 1: Load training data
# ---------------------------------------------------------------------------
try:
    df = pd.read_csv("training_data.csv")
except FileNotFoundError:
    print("Error: 'training_data.csv' not found. Please ensure the file exists in the project directory.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 2: Extract target variable BEFORE any column dropping
# ---------------------------------------------------------------------------
y = df["percentage"]

# ---------------------------------------------------------------------------
# Step 3: Build feature matrix by dropping non-feature columns
# ---------------------------------------------------------------------------
COLUMNS_TO_DROP = [
    "math score",
    "reading score",
    "writing score",
    "total",
    "average_score",
    "verbal_score",
    "score_range",
    "percentage",   # target column — already extracted as y
]
X = df.drop(columns=COLUMNS_TO_DROP)


# ---------------------------------------------------------------------------
# Step 4: Train/test split (80/20)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------------------------
# Step 5: Helper — select the best model by R², tie-break by lowest RMSE
# ---------------------------------------------------------------------------
def select_best_model(results: list[dict]) -> dict:
    """
    Given a list of dicts with keys 'name', 'r2', 'rmse',
    returns the dict with the highest r2, breaking ties by lowest rmse.
    """
    return max(results, key=lambda rec: (rec["r2"], -rec["rmse"]))

# ---------------------------------------------------------------------------
# Step 6: Model definitions
# ---------------------------------------------------------------------------
models = [
    LinearRegression(),
    Ridge(alpha=1.0),
    Lasso(alpha=1.0),
    RandomForestRegressor(n_estimators=100, random_state=42),
    GradientBoostingRegressor(random_state=42),
    ExtraTreesRegressor(n_estimators=100, random_state=42),
]

model_names = [
    "Linear Regression",
    "Ridge Regression",
    "Lasso Regression",
    "Random Forest",
    "Gradient Boosting",
    "Extra Trees",
]

# ---------------------------------------------------------------------------
# Step 7: Training loop — fit, evaluate, catch exceptions, skip failures
# ---------------------------------------------------------------------------
results = []

for model, name in zip(models, model_names):
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2   = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results.append({"name": name, "model": model, "r2": r2, "rmse": rmse})
    except Exception as exc:
        print(f"[WARNING] Model '{name}' failed to fit: {exc}. Skipping.")

# ---------------------------------------------------------------------------
# Step 8: Print comparison table
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"{'Model':<25} {'R²':>10} {'RMSE':>10}")
print("-" * 60)
for rec in results:
    print(f"{rec['name']:<25} {rec['r2']:>10.4f} {rec['rmse']:>10.4f}")
print("=" * 60 + "\n")

# ---------------------------------------------------------------------------
# Step 9: Select best model
# ---------------------------------------------------------------------------
best_result = select_best_model(results)
best_model = best_result["model"]

print(f"Best model: {best_result['name']}  (R²={best_result['r2']:.4f}, RMSE={best_result['rmse']:.4f})")

# ---------------------------------------------------------------------------
# Step 10: Capture feature names
# ---------------------------------------------------------------------------
feature_names = X.columns.tolist()

# ---------------------------------------------------------------------------
# Step 11: Build model_results DataFrame
# ---------------------------------------------------------------------------
model_results = pd.DataFrame(
    [{"model": rec["name"], "r2": rec["r2"], "rmse": rec["rmse"]} for rec in results]
)

# ---------------------------------------------------------------------------
# Step 12: Serialize artifacts to pickle files
# ---------------------------------------------------------------------------
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)

with open("model_results.pkl", "wb") as f:
    pickle.dump(model_results, f)

print("Training complete. Saved best_model.pkl, feature_names.pkl, model_results.pkl.")
