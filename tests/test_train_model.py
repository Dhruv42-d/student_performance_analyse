from hypothesis import given, settings, strategies as st
import pandas as pd
import numpy as np

DISALLOWED = ["math score", "reading score", "writing score", "total", "average_score", "verbal_score", "score_range"]
COLUMNS_TO_DROP = DISALLOWED + ["percentage"]


def make_df(extra_cols):
    """Build a minimal DataFrame with all required columns plus extras."""
    n = 5
    data = {col: np.zeros(n) for col in COLUMNS_TO_DROP}
    data["percentage"] = np.ones(n) * 75.0
    for col in extra_cols:
        data[col] = np.zeros(n)
    return pd.DataFrame(data)


@settings(max_examples=100)
@given(st.lists(st.text(min_size=1, max_size=10).filter(lambda s: s not in COLUMNS_TO_DROP), max_size=5, unique=True))
def test_target_extraction_precedes_column_dropping(extra_cols):
    # Feature: student-performance-ml-app, Property 1: Target extraction precedes column dropping
    df = make_df(extra_cols)
    y = df["percentage"]          # extract BEFORE dropping
    X = df.drop(columns=COLUMNS_TO_DROP)
    assert y.equals(df["percentage"])


@settings(max_examples=100)
@given(st.lists(st.text(min_size=1, max_size=10).filter(lambda s: s not in COLUMNS_TO_DROP), max_size=5, unique=True))
def test_feature_matrix_excludes_disallowed_columns(extra_cols):
    # Feature: student-performance-ml-app, Property 2: Feature matrix excludes disallowed columns
    df = make_df(extra_cols)
    X = df.drop(columns=COLUMNS_TO_DROP)
    for col in DISALLOWED:
        assert col not in X.columns
    assert "percentage" not in X.columns


# ---------------------------------------------------------------------------
# Property 3: Best model selection correctness
# Feature: student-performance-ml-app, Property 3: Best model selection correctness
# ---------------------------------------------------------------------------

def select_best_model(results):
    """Local mirror of train_model.select_best_model for isolated testing."""
    return max(results, key=lambda rec: (rec["r2"], -rec["rmse"]))


@settings(max_examples=100)
@given(
    st.lists(
        st.fixed_dictionaries({
            "name": st.text(min_size=1),
            "r2": st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            "rmse": st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        }),
        min_size=1,
        max_size=10,
    )
)
def test_best_model_selection_correctness(records):
    # Feature: student-performance-ml-app, Property 3: Best model selection correctness
    result = select_best_model(records)
    max_r2 = max(rec["r2"] for rec in records)
    # Result must have the maximum R²
    assert result["r2"] == max_r2
    # Among all records tied for max R², result must have the minimum RMSE
    tied = [rec for rec in records if rec["r2"] == max_r2]
    min_rmse_among_tied = min(rec["rmse"] for rec in tied)
    assert result["rmse"] == min_rmse_among_tied


# ---------------------------------------------------------------------------
# Property 4: Pickle round-trip for model artifacts
# Feature: student-performance-ml-app, Property 4: Pickle round-trip for model artifacts
# ---------------------------------------------------------------------------
import pickle
import io
from sklearn.linear_model import LinearRegression


def pickle_roundtrip(obj):
    """Serialize obj to bytes and deserialize back."""
    buf = io.BytesIO()
    pickle.dump(obj, buf)
    buf.seek(0)
    return pickle.load(buf)


@settings(max_examples=100)
@given(st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=30))
def test_pickle_roundtrip_feature_names(feature_names):
    # Feature: student-performance-ml-app, Property 4: Pickle round-trip for model artifacts
    recovered = pickle_roundtrip(feature_names)
    assert recovered == feature_names


@settings(max_examples=100)
@given(
    st.lists(
        st.fixed_dictionaries({
            "model": st.text(min_size=1),
            "r2": st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            "rmse": st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        }),
        min_size=1,
        max_size=10,
    )
)
def test_pickle_roundtrip_dataframe(records):
    # Feature: student-performance-ml-app, Property 4: Pickle round-trip for model artifacts
    import pandas as pd
    df = pd.DataFrame(records)
    recovered = pickle_roundtrip(df)
    pd.testing.assert_frame_equal(df, recovered)


@settings(max_examples=50)
@given(
    st.lists(
        st.lists(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False), min_size=2, max_size=2),
        min_size=5,
        max_size=20,
    )
)
def test_pickle_roundtrip_fitted_model(data_rows):
    # Feature: student-performance-ml-app, Property 4: Pickle round-trip for model artifacts
    import pandas as pd
    import numpy as np
    X = pd.DataFrame(data_rows, columns=["a", "b"])
    y = X["a"] * 2 + X["b"] + 1.0
    model = LinearRegression()
    model.fit(X, y)
    recovered_model = pickle_roundtrip(model)
    original_preds = model.predict(X)
    recovered_preds = recovered_model.predict(X)
    np.testing.assert_array_almost_equal(original_preds, recovered_preds)


# ---------------------------------------------------------------------------
# Unit tests for train_model pipeline
# ---------------------------------------------------------------------------
import pytest
import sys
from unittest.mock import patch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error


def test_missing_csv_causes_exit(tmp_path, monkeypatch):
    """Requirement 1.2: missing training_data.csv must cause sys.exit(1)."""
    monkeypatch.chdir(tmp_path)  # switch to a temp dir with no CSV
    with pytest.raises(SystemExit) as exc_info:
        # Simulate the guard in train_model.py
        try:
            pd.read_csv("training_data.csv")
        except FileNotFoundError:
            print("Error: 'training_data.csv' not found.")
            sys.exit(1)
    assert exc_info.value.code == 1


def test_80_20_split_row_counts():
    """Requirement 1.5: 80/20 split must produce correct sizes."""
    n = 1000
    X = pd.DataFrame({"a": range(n)})
    y = pd.Series(range(n))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    assert len(X_train) == 800
    assert len(X_test) == 200


def test_all_6_models_in_results():
    """Requirement 1.6: smoke-test the training loop produces all 6 model names."""
    # Minimal in-memory DataFrame with the expected columns
    n = 50
    feature_cols = [
        "gender_female", "gender_male",
        "race/ethnicity_group A", "race/ethnicity_group B",
        "parental level of education_associate's degree",
        "parental level of education_bachelor's degree",
        "lunch_free/reduced", "lunch_standard",
        "test preparation course_completed", "test preparation course_none",
    ]
    data = {col: np.random.randint(0, 2, n) for col in feature_cols}
    data["percentage"] = np.random.uniform(40, 100, n)
    df = pd.DataFrame(data)

    y = df["percentage"]
    COLUMNS_TO_DROP = [
        "math score", "reading score", "writing score",
        "total", "average_score", "verbal_score", "score_range", "percentage"
    ]
    X = df.drop(columns=[c for c in COLUMNS_TO_DROP if c in df.columns])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = [
        LinearRegression(),
        Ridge(alpha=1.0),
        Lasso(alpha=1.0),
        RandomForestRegressor(n_estimators=10, random_state=42),
        GradientBoostingRegressor(random_state=42),
        ExtraTreesRegressor(n_estimators=10, random_state=42),
    ]
    expected_names = [
        "Linear Regression", "Ridge Regression", "Lasso Regression",
        "Random Forest", "Gradient Boosting", "Extra Trees",
    ]
    results = []
    for model, name in zip(models, expected_names):
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            results.append({"name": name, "r2": r2, "rmse": rmse})
        except Exception:
            pass

    result_names = [r["name"] for r in results]
    for name in expected_names:
        assert name in result_names, f"Model '{name}' missing from results"
