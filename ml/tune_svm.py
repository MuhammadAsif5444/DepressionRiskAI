import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report


# ============================================================
# 1. LOAD DATASET
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "PHQ-9 Student Depression Dataset"
    / "PHQ-9_Dataset_5th Edition.csv"
)

df = pd.read_csv(DATA_FILE)

df.columns = df.columns.str.strip()


# ============================================================
# 2. FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["PHQ_Total", "PHQ_Severity"])

y = df["PHQ_Severity"]


# ============================================================
# 3. FEATURE TYPES
# ============================================================

categorical_features = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


# ============================================================
# 4. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)


# ============================================================
# 5. BASE SVM
# ============================================================

svm = SVC(
    class_weight="balanced"
)


# ============================================================
# 6. PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", svm)
    ]
)


# ============================================================
# 7. PARAMETER GRID
# ============================================================

parameter_grid = {

    "model__kernel": [
        "rbf",
        "linear"
    ],

    "model__C": [
        0.1,
        1,
        10,
        100
    ],

    "model__gamma": [
        "scale",
        "auto"
    ]
}


# ============================================================
# 8. CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 9. GRID SEARCH
# ============================================================

print("=" * 70)
print("SVM HYPERPARAMETER TUNING")
print("=" * 70)

print("\nTesting parameter combinations...")
print("This may take a little while.\n")


grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=parameter_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1
)


grid_search.fit(X, y)


# ============================================================
# 10. BEST PARAMETERS
# ============================================================

print("\n" + "=" * 70)
print("BEST SVM CONFIGURATION")
print("=" * 70)

print("\nBest parameters:")

for parameter, value in grid_search.best_params_.items():
    print(f"{parameter}: {value}")


# ============================================================
# 11. BEST SCORE
# ============================================================

print(
    f"\nBest Macro-F1: "
    f"{grid_search.best_score_:.4f}"
)


# ============================================================
# 12. SAVE RESULTS
# ============================================================

results = pd.DataFrame(
    grid_search.cv_results_
)

results = results.sort_values(
    by="rank_test_score"
)

RESULT_FILE = (
    PROJECT_ROOT
    / "ml"
    / "svm_tuning_results.csv"
)

results.to_csv(
    RESULT_FILE,
    index=False
)

print("\nAll tuning results saved to:")
print(RESULT_FILE)


# ============================================================
# 13. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("SVM TUNING COMPLETE")
print("=" * 70)