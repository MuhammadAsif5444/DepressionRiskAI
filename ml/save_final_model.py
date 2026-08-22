import pandas as pd
import joblib

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "PHQ-9 Student Depression Dataset"
    / "PHQ-9_Dataset_5th Edition.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "ml"
    / "final_model.joblib"
)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("DEPRESSIONRISKAI - SAVE FINAL MODEL")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

df.columns = df.columns.str.strip()

print(f"Dataset shape: {df.shape}")


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=["PHQ_Total", "PHQ_Severity"]
)

y = df["PHQ_Severity"]


# ============================================================
# FEATURE TYPES
# ============================================================

categorical_features = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# PREPROCESSING
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
# FINAL SVM
# ============================================================

model = SVC(
    kernel="linear",
    C=100,
    gamma="scale",
    class_weight="balanced",
    probability=True
)


# ============================================================
# COMPLETE PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", model)
    ]
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining final SVM...")

pipeline.fit(X, y)

print("Training complete!")


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    pipeline,
    MODEL_FILE
)


print("\n" + "=" * 70)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 70)

print(f"\nModel location:")
print(MODEL_FILE)

print("\nModel type:")
print("Linear SVM")

print("\nParameters:")
print("C = 100")
print("Kernel = linear")
print("Gamma = scale")

print("\n" + "=" * 70)
print("MODEL CREATION COMPLETE")
print("=" * 70)