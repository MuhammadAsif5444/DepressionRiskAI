import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


# ============================================================
# 1. LOAD DATA
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
# 2. FEATURES / TARGET
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
# 5. MODEL
# ============================================================

model = SVC(
    kernel="linear",
    C=100,
    gamma="scale",
    class_weight="balanced"
)


# ============================================================
# 6. PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", model)
    ]
)


# ============================================================
# 7. TRAIN
# ============================================================

pipeline.fit(X, y)


# ============================================================
# 8. GET FEATURE NAMES
# ============================================================

feature_names = (
    pipeline
    .named_steps["preprocessing"]
    .get_feature_names_out()
)


# ============================================================
# 9. GET SVM COEFFICIENTS
# ============================================================

svm_model = pipeline.named_steps["model"]

coefficients = svm_model.coef_

classes = svm_model.classes_


# ============================================================
# 10. ANALYZE FEATURES
# ============================================================

print("=" * 70)
print("EXPLAINABLE AI - FEATURE IMPORTANCE")
print("=" * 70)

print("\nClasses:")
print(classes)


for index, class_name in enumerate(classes):

    print("\n" + "-" * 70)
    print(f"TOP FEATURES FOR CLASS: {class_name}")
    print("-" * 70)

    class_coefficients = coefficients[index]

    top_indices = np.argsort(
        np.abs(class_coefficients)
    )[::-1][:15]

    for rank, feature_index in enumerate(
        top_indices,
        start=1
    ):

        feature = feature_names[feature_index]

        value = class_coefficients[feature_index]

        print(
            f"{rank:2}. "
            f"{feature:<70} "
            f"{value:+.4f}"
        )


# ============================================================
# 11. SAVE FEATURE IMPORTANCE
# ============================================================

rows = []

for index, class_name in enumerate(classes):

    class_coefficients = coefficients[index]

    for feature_index, feature in enumerate(feature_names):

        rows.append({
            "Class": class_name,
            "Feature": feature,
            "Coefficient": class_coefficients[feature_index],
            "Absolute Importance": abs(
                class_coefficients[feature_index]
            )
        })


importance_df = pd.DataFrame(rows)

importance_df = importance_df.sort_values(
    by=["Class", "Absolute Importance"],
    ascending=[True, False]
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "ml"
    / "feature_importance.csv"
)

importance_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE SAVED")
print("=" * 70)

print(OUTPUT_FILE)