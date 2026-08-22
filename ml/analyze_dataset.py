import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "PHQ-9 Student Depression Dataset"
    / "PHQ-9_Dataset_5th Edition.csv"
)

df = pd.read_csv(DATA_FILE)

print("=" * 70)
print("DEPRESSION RISK AI - DATASET INSPECTION")
print("=" * 70)

print(f"\nRows: {len(df)}")
print(f"Columns: {len(df.columns)}")

print("\nCOLUMNS AND UNIQUE VALUES")
print("-" * 70)

for column in df.columns:
    print(f"\nCOLUMN: {column}")
    print(f"Data type: {df[column].dtype}")
    print(f"Missing: {df[column].isna().sum()}")
    print(f"Unique values: {df[column].nunique()}")

    if df[column].nunique() <= 15:
        print("Values:")
        print(df[column].value_counts(dropna=False).to_string())
    else:
        print("Sample values:")
        print(df[column].dropna().head(5).tolist())

print("\n" + "=" * 70)
print("DATASET INSPECTION COMPLETE")
print("=" * 70)