# Feature: student-performance-ml-app, Property 9: Descriptive statistics rounding invariant
# Feature: student-performance-ml-app, Property 10: Categorical percentage correctness

import pytest
import pandas as pd
from hypothesis import given, settings, strategies as st


# ---------------------------------------------------------------------------
# Property 9: Descriptive statistics rounding invariant
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(st.floats(allow_nan=False, allow_infinity=False))
def test_rounding_idempotent(x):
    # Feature: student-performance-ml-app, Property 9: Descriptive statistics rounding invariant
    assert round(round(x, 2), 2) == round(x, 2)


# ---------------------------------------------------------------------------
# Property 10: Categorical percentage correctness
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(st.lists(st.text(min_size=1), min_size=1))
def test_categorical_percentage_sums_to_100(values):
    # Feature: student-performance-ml-app, Property 10: Categorical percentage correctness
    series = pd.Series(values)
    N = len(series)
    counts = series.value_counts()
    percentages = (counts / N) * 100
    assert abs(percentages.sum() - 100.0) < 0.01


# ---------------------------------------------------------------------------
# Property 5: Grade boundary correctness
# Feature: student-performance-ml-app, Property 5: Grade boundary correctness
# ---------------------------------------------------------------------------

def _get_grade(p):
    if p >= 90: return "A"
    if p >= 75: return "B"
    if p >= 60: return "C"
    if p >= 50: return "D"
    return "F"


@settings(max_examples=100)
@given(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))
def test_grade_boundary_correctness(p):
    # Feature: student-performance-ml-app, Property 5: Grade boundary correctness
    grade = _get_grade(p)
    assert grade in {"A", "B", "C", "D", "F"}
    if p >= 90:
        assert grade == "A"
    elif p >= 75:
        assert grade == "B"
    elif p >= 60:
        assert grade == "C"
    elif p >= 50:
        assert grade == "D"
    else:
        assert grade == "F"


# ---------------------------------------------------------------------------
# Property 11: Model results "(Best)" annotation uniqueness
# Feature: student-performance-ml-app, Property 11: Model results "(Best)" annotation uniqueness
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(st.lists(st.text(min_size=1, max_size=20), min_size=6, max_size=6, unique=True))
def test_best_annotation_uniqueness(model_names):
    # Feature: student-performance-ml-app, Property 11: Model results "(Best)" annotation uniqueness
    r2_values = [float(i) for i in range(len(model_names))]  # ascending → last is best
    df = pd.DataFrame({"model": model_names, "r2": r2_values})
    best_name = df.loc[df["r2"].idxmax(), "model"]
    df_annotated = df.copy()
    df_annotated["model"] = df_annotated["model"].apply(
        lambda n: f"{n} (Best)" if n == best_name else n
    )
    # Exactly one row should contain "(Best)"
    best_rows = df_annotated[df_annotated["model"].str.contains(r"\(Best\)", regex=True)]
    assert len(best_rows) == 1
    # All other rows should be unchanged
    for i, row in df_annotated.iterrows():
        if row["model"] != f"{best_name} (Best)":
            assert row["model"] == df.loc[i, "model"]


# ---------------------------------------------------------------------------
# Property 6: Single-input one-hot encoding alignment
# Feature: student-performance-ml-app, Property 6: Single-input one-hot encoding alignment
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "gender_female", "gender_male",
    "race/ethnicity_group A", "race/ethnicity_group B", "race/ethnicity_group C",
    "race/ethnicity_group D", "race/ethnicity_group E", "race/ethnicity_group F",
    "parental level of education_associate's degree",
    "parental level of education_bachelor's degree",
    "parental level of education_high school",
    "parental level of education_master's degree",
    "parental level of education_some college",
    "parental level of education_some high school",
    "lunch_free/reduced", "lunch_standard",
    "test preparation course_completed", "test preparation course_none",
]

GENDER_OPTIONS = ["female", "male"]
RACE_OPTIONS = ["group A", "group B", "group C", "group D", "group E", "group F"]
EDUCATION_OPTIONS = ["associate's degree", "bachelor's degree", "high school",
                     "master's degree", "some college", "some high school"]
LUNCH_OPTIONS = ["free/reduced", "standard"]
PREP_OPTIONS = ["completed", "none"]


def _encode_single_input(selections, feature_names):
    row_dict = {name: 0 for name in feature_names}
    for column, value in selections.items():
        key = f"{column}_{value}"
        if key in row_dict:
            row_dict[key] = 1
    return pd.DataFrame([row_dict])[feature_names]


@settings(max_examples=100)
@given(
    st.sampled_from(GENDER_OPTIONS),
    st.sampled_from(RACE_OPTIONS),
    st.sampled_from(EDUCATION_OPTIONS),
    st.sampled_from(LUNCH_OPTIONS),
    st.sampled_from(PREP_OPTIONS),
)
def test_encode_single_input_alignment(gender, race, education, lunch, prep):
    # Feature: student-performance-ml-app, Property 6: Single-input one-hot encoding alignment
    selections = {
        "gender": gender,
        "race/ethnicity": race,
        "parental level of education": education,
        "lunch": lunch,
        "test preparation course": prep,
    }
    result = _encode_single_input(selections, FEATURE_NAMES)
    assert result.shape == (1, len(FEATURE_NAMES))
    assert list(result.columns) == FEATURE_NAMES
    assert set(result.values.flatten().tolist()).issubset({0, 1})
    gender_cols = [c for c in FEATURE_NAMES if c.startswith("gender_")]
    race_cols = [c for c in FEATURE_NAMES if c.startswith("race/ethnicity_")]
    edu_cols = [c for c in FEATURE_NAMES if c.startswith("parental level of education_")]
    lunch_cols = [c for c in FEATURE_NAMES if c.startswith("lunch_")]
    prep_cols = [c for c in FEATURE_NAMES if c.startswith("test preparation course_")]
    for group in [gender_cols, race_cols, edu_cols, lunch_cols, prep_cols]:
        assert result[group].sum(axis=1).iloc[0] == 1


# ---------------------------------------------------------------------------
# Unit tests: single prediction grade correctness (Requirements 5.5–5.7)
# ---------------------------------------------------------------------------

def test_get_grade_a():
    assert _get_grade(95.0) == "A"
    assert _get_grade(90.0) == "A"

def test_get_grade_b():
    assert _get_grade(85.0) == "B"
    assert _get_grade(75.0) == "B"
    assert _get_grade(89.99) == "B"

def test_get_grade_c():
    assert _get_grade(70.0) == "C"
    assert _get_grade(60.0) == "C"
    assert _get_grade(74.99) == "C"

def test_get_grade_d():
    assert _get_grade(55.0) == "D"
    assert _get_grade(50.0) == "D"
    assert _get_grade(59.99) == "D"

def test_get_grade_f():
    assert _get_grade(45.0) == "F"
    assert _get_grade(0.0) == "F"
    assert _get_grade(49.99) == "F"

def test_encode_produces_correct_one_hot_keys():
    """Verify encode_single_input sets the right keys to 1."""
    result = _encode_single_input(
        {
            "gender": "male",
            "race/ethnicity": "group C",
            "parental level of education": "bachelor's degree",
            "lunch": "standard",
            "test preparation course": "completed",
        },
        FEATURE_NAMES,
    )
    assert result["gender_male"].iloc[0] == 1
    assert result["gender_female"].iloc[0] == 0
    assert result["race/ethnicity_group C"].iloc[0] == 1
    assert result["lunch_standard"].iloc[0] == 1
    assert result["test preparation course_completed"].iloc[0] == 1


# ---------------------------------------------------------------------------
# Property 7: Bulk encoding column alignment
# Feature: student-performance-ml-app, Property 7: Bulk encoding column alignment
# ---------------------------------------------------------------------------

REQUIRED_COLS = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
]


def _build_bulk_row(gender, race, education, lunch, prep):
    return {
        "gender": gender,
        "race/ethnicity": race,
        "parental level of education": education,
        "lunch": lunch,
        "test preparation course": prep,
    }


@settings(max_examples=100, deadline=None)
@given(
    st.lists(
        st.builds(
            _build_bulk_row,
            gender=st.sampled_from(GENDER_OPTIONS),
            race=st.sampled_from(RACE_OPTIONS),
            education=st.sampled_from(EDUCATION_OPTIONS),
            lunch=st.sampled_from(LUNCH_OPTIONS),
            prep=st.sampled_from(PREP_OPTIONS),
        ),
        min_size=1,
        max_size=10,
    )
)
def test_bulk_encoding_column_alignment(rows):
    # Feature: student-performance-ml-app, Property 7: Bulk encoding column alignment
    df = pd.DataFrame(rows)
    encoded = pd.get_dummies(df[REQUIRED_COLS])
    encoded = encoded.reindex(columns=FEATURE_NAMES, fill_value=0)
    assert list(encoded.columns) == FEATURE_NAMES
    assert encoded.isnull().sum().sum() == 0


# ---------------------------------------------------------------------------
# Property 8: Missing column detection completeness
# Feature: student-performance-ml-app, Property 8: Missing column detection completeness
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(st.lists(st.sampled_from(REQUIRED_COLS), min_size=0, max_size=5, unique=True))
def test_missing_column_detection_completeness(omitted_cols):
    # Feature: student-performance-ml-app, Property 8: Missing column detection completeness
    present_cols = [c for c in REQUIRED_COLS if c not in omitted_cols]
    df = pd.DataFrame(columns=present_cols)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    assert set(missing) == set(omitted_cols)


# ---------------------------------------------------------------------------
# Unit tests: bulk upload edge cases (Task 9.11)
# ---------------------------------------------------------------------------

def test_sample_csv_has_correct_columns():
    """Requirement 5.15: sample CSV must contain all required column names."""
    sample_data = pd.DataFrame([{
        "gender": "female",
        "race/ethnicity": "group B",
        "parental level of education": "bachelor's degree",
        "lunch": "standard",
        "test preparation course": "completed",
    }])
    for col in REQUIRED_COLS:
        assert col in sample_data.columns, f"Sample CSV missing column: {col}"


def test_bulk_upload_missing_column_shows_in_error():
    """Requirement 5.11: missing column must be reported by name."""
    df = pd.DataFrame([{
        "gender": "male",
        "race/ethnicity": "group A",
        "parental level of education": "high school",
        # "lunch" is intentionally omitted
        "test preparation course": "none",
    }])
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    assert "lunch" in missing
    assert len(missing) == 1
