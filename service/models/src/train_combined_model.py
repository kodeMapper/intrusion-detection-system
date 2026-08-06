# =========================================
# Imports
# =========================================

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier


# =========================================
# Load Dataset
# =========================================

print("Loading Datasets...")

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "service" / "models" / "artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

train_path = ROOT / "data" / "raw" / "UNSW_NB15_training-set.csv"
test_path = ROOT / "data" / "raw" / "UNSW_NB15_testing-set.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Train Shape:", train_df.shape)
print("Test Shape:", test_df.shape)


# =========================================
# Combine Dataset
# =========================================

print("\nCombining datasets...")

df = pd.concat([train_df, test_df], ignore_index=True)

print("Combined Shape:", df.shape)


# =========================================
# Drop Non Live Features
# =========================================

drop_cols = ["srcip","dstip","id","Stime","Ltime"]

df = df.drop(columns=drop_cols, errors="ignore")


# =========================================
# Drop Weak Attacks
# =========================================

weak_attacks = [
    "Analysis",
    "Backdoor",
    "Shellcode",
    "Worms"
]

df = df[~df["attack_cat"].isin(weak_attacks)]

print("\nAfter Dropping Weak Attacks")
print(df["attack_cat"].value_counts())


# =========================================
# Feature Split
# =========================================

X = df.drop(["attack_cat","label"], axis=1)
y = df["attack_cat"]


# =========================================
# Label Encoding
# =========================================

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("\nClass Mapping")

for i, label in enumerate(label_encoder.classes_):
    print(i, "->", label)


# =========================================
# Encode categorical
# =========================================

cat_cols = X.select_dtypes(include="object").columns

encoders = {}

for col in cat_cols:

    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

    encoders[col] = le


# =========================================
# Train Test Split
# =========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTrain Size:", X_train.shape)
print("Test Size:", X_test.shape)


# =========================================
# Models
# =========================================

print("\nTraining XGBoost")

xgb = XGBClassifier(

    n_estimators=600,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method="hist",
    eval_metric="mlogloss",
    random_state=42
)

xgb.fit(X_train, y_train)


print("\nTraining Random Forest")

rf = RandomForestClassifier(

    n_estimators=500,
    max_depth=20,
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train, y_train)


print("\nTraining LightGBM")

lgb = LGBMClassifier(

    n_estimators=600,
    learning_rate=0.05,
    num_leaves=64,
    random_state=42
)

lgb.fit(X_train, y_train)


# =========================================
# Ensemble
# =========================================

print("\nRunning Ensemble")

proba = (

    xgb.predict_proba(X_test)
    + rf.predict_proba(X_test)
    + lgb.predict_proba(X_test)

) / 3


pred = np.argmax(proba, axis=1)


# =========================================
# Evaluation
# =========================================

print("\nAccuracy")

print(
    accuracy_score(y_test, pred)
)

print("\nMacro Recall")

print(
    recall_score(
        y_test,
        pred,
        average="macro"
    )
)


print("\nClassification Report")

print(
    classification_report(
        y_test,
        pred,
        target_names=label_encoder.classes_
    )
)


# =========================================
# Save Models
# =========================================

print("\nSaving Models")

joblib.dump(xgb, MODEL_DIR / "combined_xgb.pkl")
joblib.dump(rf, MODEL_DIR / "combined_rf.pkl")
joblib.dump(lgb, MODEL_DIR / "combined_lgb.pkl")

joblib.dump(encoders, MODEL_DIR / "combined_encoders.pkl")
joblib.dump(label_encoder, MODEL_DIR / "combined_label.pkl")

print("Model Saved Successfully")