import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

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

print("=" * 70)
print("DEPRESSION RISK AI - BASELINE MODEL")
print("=" * 70)

print(f"\nDataset shape: {df.shape}")


# ============================================================
# 2. CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()


# ============================================================
# 3. DEFINE TARGET
# ============================================================

target = "PHQ_Severity"

# Remove PHQ_Total to prevent leakage.
# PHQ_Total is calculated from the PHQ-9 symptom responses.

features_to_remove = [
    "PHQ_Total",
    "PHQ_Severity"
]

X = df.drop(columns=features_to_remove)
y = df[target]


# ============================================================
# 4. DISPLAY TARGET DISTRIBUTION
# ============================================================

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget percentages:")
print((y.value_counts(normalize=True) * 100).round(2))


# ============================================================
# 5. IDENTIFY COLUMN TYPES
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
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 7. PREPROCESSING
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
# 8. LOGISTIC REGRESSION MODEL
# ============================================================

model = LogisticRegression(
    max_iter=2000,
    random_state=42
)


# ============================================================
# 9. CREATE PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", model)
    ]
)


# ============================================================
# 10. TRAIN
# ============================================================

print("\nTraining Logistic Regression...")

pipeline.fit(X_train, y_train)

print("Training complete!")


# ============================================================
# 11. PREDICTION
# ============================================================

y_pred = pipeline.predict(X_test)


# ============================================================
# 12. EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


# ============================================================
# 13. RESULTS
# ============================================================

print("\n" + "=" * 70)
print("MODEL RESULTS")
print("=" * 70)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# ============================================================
# 14. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


print("\n" + "=" * 70)
print("BASELINE EXPERIMENT COMPLETE")
print("=" * 70)