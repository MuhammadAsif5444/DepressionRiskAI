import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


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
# 5. MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),

    "SVM": SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced"
    )
}


# ============================================================
# 6. CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


scoring = {
    "accuracy": "accuracy",
    "precision": "precision_weighted",
    "recall": "recall_weighted",
    "f1": "f1_weighted",
    "macro_f1": "f1_macro"
}


results = []


# ============================================================
# 7. TRAIN AND VALIDATE
# ============================================================

for model_name, model in models.items():

    print("\n" + "=" * 70)
    print(f"VALIDATING: {model_name}")
    print("=" * 70)

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    result = {
        "Model": model_name,

        "Accuracy Mean": scores["test_accuracy"].mean(),
        "Accuracy Std": scores["test_accuracy"].std(),

        "Precision Mean": scores["test_precision"].mean(),
        "Recall Mean": scores["test_recall"].mean(),

        "F1 Mean": scores["test_f1"].mean(),
        "F1 Std": scores["test_f1"].std(),

        "Macro F1 Mean": scores["test_macro_f1"].mean(),
        "Macro F1 Std": scores["test_macro_f1"].std()
    }

    results.append(result)

    print(
        f"Accuracy: "
        f"{result['Accuracy Mean']:.4f} "
        f"+/- {result['Accuracy Std']:.4f}"
    )

    print(
        f"Weighted F1: "
        f"{result['F1 Mean']:.4f} "
        f"+/- {result['F1 Std']:.4f}"
    )

    print(
        f"Macro F1: "
        f"{result['Macro F1 Mean']:.4f} "
        f"+/- {result['Macro F1 Std']:.4f}"
    )


# ============================================================
# 8. RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Macro F1 Mean",
    ascending=False
)


print("\n\n")
print("=" * 70)
print("5-FOLD CROSS-VALIDATION RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 9. SAVE RESULTS
# ============================================================

RESULT_FILE = PROJECT_ROOT / "ml" / "cross_validation_results.csv"

results_df.to_csv(
    RESULT_FILE,
    index=False
)

print("\nResults saved to:")
print(RESULT_FILE)

print("\n" + "=" * 70)
print("CROSS-VALIDATION COMPLETE")
print("=" * 70)