# ======================================
# Step 1 - Imports
# ======================================

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report

from xgboost import XGBClassifier

import joblib


# ======================================
# Step 2 - Load Dataset
# ======================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "UNSW_NB15_training-set.csv"
MODEL_DIR = PROJECT_ROOT / "service" / "models" / "artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATASET_PATH)

print("Dataset Shape:", df.shape)


# ======================================
# Step 3 - Drop Non Live Features
# ======================================

drop_cols = ["srcip", "dstip", "id", "Stime", "Ltime"]
df = df.drop(columns=drop_cols, errors="ignore")


# ======================================
# Step 4 - Create Binary Labels
# ======================================

df["binary_label"] = df["attack_cat"].apply(
    lambda x: 0 if x == "Normal" else 1
)

print("\nBinary Distribution:")
print(df["binary_label"].value_counts())


# ======================================
# Step 5 - Feature Split
# ======================================

X = df.drop(["attack_cat", "label", "binary_label"], axis=1)
y = df["binary_label"]


# ======================================
# Step 6 - Train Test Split
# ======================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain Size:", X_train.shape)
print("Test Size :", X_test.shape)


# ======================================
# Step 7 - Encode Categorical Features
# ======================================

cat_cols = X_train.select_dtypes(include="object").columns.tolist()

encoders = {}

for col in cat_cols:
    encoders[col] = LabelEncoder()

    X_train[col] = encoders[col].fit_transform(X_train[col])
    X_test[col] = encoders[col].transform(X_test[col])


# ======================================
# Step 8 - Create Binary Model
# ======================================

print("\nTraining Stage-1 Binary Model...")

model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    tree_method="hist",
    random_state=42
)


# ======================================
# Step 9 - Cross Validation
# ======================================

print("\nRunning 5-Fold Cross Validation...")

cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="recall"
)

print("Cross Validation Recall Scores:", cv_scores)
print("Average Cross Validation Recall:", cv_scores.mean())


# ======================================
# Step 10 - Train Model
# ======================================

model.fit(X_train, y_train)

print("\nTraining Complete")


# ======================================
# Step 11 - Prediction
# ======================================

y_pred = model.predict(X_test)


# ======================================
# Step 12 - Evaluation
# ======================================

print("\n============================")
print("STAGE-1 BINARY RESULTS")
print("============================")

print("\nAccuracy:",
      accuracy_score(y_test, y_pred))

print("\nRecall:",
      recall_score(y_test, y_pred))

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))


# ======================================
# Step 13 - Save Model + Encoders
# ======================================

print("\nSaving Stage-1 Model...")

joblib.dump(model, MODEL_DIR / "stage1_binary.pkl")
joblib.dump(encoders, MODEL_DIR / "stage1_encoders.pkl")

print("Stage-1 Model Saved Successfully")