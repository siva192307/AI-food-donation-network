# =============================================================================
# train_model.py - AI Food Donation Network
# =============================================================================
# Trains a Random Forest Classifier to predict whether donated food is safe.
#
# Outputs
# -------
#   food_quality_model.pkl   – trained RandomForestClassifier
#   label_encoder.pkl        – LabelEncoder fitted on Food_Type column
#
# Usage
# -----
#   python train_model.py
# =============================================================================

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble          import RandomForestClassifier
from sklearn.model_selection   import train_test_split
from sklearn.preprocessing     import LabelEncoder
from sklearn.metrics           import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ── 1. Load dataset ────────────────────────────────────────────────────────────
DATASET_PATH = "food_quality_dataset.csv"

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}\n"
        "Run `python generate_dataset.py` first."
    )

print("=" * 60)
print("  AI Food Donation Network - Model Training")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)
print(f"\nDataset loaded   : {len(df):,} records")
print(f"Columns          : {list(df.columns)}")
print(f"\nClass distribution:\n{df['Food_Safe'].value_counts().to_string()}")

# ── 2. Encode categorical column (Food_Type) ──────────────────────────────────
le = LabelEncoder()
df["Food_Type_Enc"] = le.fit_transform(df["Food_Type"])

print(f"\nFood type classes: {list(le.classes_)}")

# ── 3. Define features and target ─────────────────────────────────────────────
FEATURE_COLS = ["Food_Type_Enc", "Quantity", "Preparation_Time",
                "Storage_Hours", "Temperature"]

X = df[FEATURE_COLS].values
y = (df["Food_Safe"] == "Yes").astype(int).values   # 1 = Safe, 0 = Unsafe

# ── 4. Train / test split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTrain samples    : {len(X_train):,}")
print(f"Test  samples    : {len(X_test):,}")

# ── 5. Train RandomForestClassifier ───────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=200,       # 200 trees for solid accuracy
    max_depth=None,         # let trees grow until pure leaves
    min_samples_split=4,    # avoid micro-splits
    min_samples_leaf=2,
    class_weight="balanced",# handles slight class imbalance
    random_state=42,
    n_jobs=-1,              # use all CPU cores
)

print("\nTraining Random Forest … ", end="", flush=True)
model.fit(X_train, y_train)
print("done")

# ── 6. Evaluate ────────────────────────────────────────────────────────────────
y_pred     = model.predict(X_test)
y_proba    = model.predict_proba(X_test)
accuracy   = accuracy_score(y_test, y_pred)

print("\n" + "-" * 60)
print(f"  Test Accuracy    : {accuracy * 100:.2f}%")
print("-" * 60)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Unsafe", "Safe"]))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(f"               Predicted")
print(f"               Unsafe  Safe")
print(f"  Actual Unsafe  {cm[0,0]:>5}  {cm[0,1]:>5}")
print(f"  Actual Safe    {cm[1,0]:>5}  {cm[1,1]:>5}")

# Feature importance
print("\nFeature Importances:")
feat_names = ["Food_Type", "Quantity", "Prep_Time", "Storage_Hours", "Temperature"]
importances = sorted(zip(feat_names, model.feature_importances_),
                     key=lambda x: x[1], reverse=True)
for name, imp in importances:
    bar = "#" * int(imp * 40)
    print(f"  {name:<18} {imp:.4f}  {bar}")

# ── 7. Persist model and encoder ──────────────────────────────────────────────
MODEL_PATH   = "food_quality_model.pkl"
ENCODER_PATH = "label_encoder.pkl"

joblib.dump(model, MODEL_PATH)
joblib.dump(le,    ENCODER_PATH)

print(f"\nModel saved      : {MODEL_PATH}")
print(f"Encoder saved    : {ENCODER_PATH}")
print("\nTraining complete.")
print("=" * 60)
