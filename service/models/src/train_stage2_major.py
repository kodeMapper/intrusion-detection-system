# ======================================
# Step 1 - Imports
# ======================================

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, recall_score,
    classification_report, balanced_accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTENC

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

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
# Step 3 - Drop non-live features
# ======================================

drop_cols = ["srcip", "dstip", "id", "Stime", "Ltime"]
df = df.drop(columns=drop_cols, errors="ignore")


# ======================================
# Step 4 - Select Major Attacks
# ======================================

major_attacks = ["Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance"]
df = df[df["attack_cat"].isin(major_attacks)]

print("\nMajor Dataset:")
print(df["attack_cat"].value_counts())


# ======================================
# Step 5 - Split
# ======================================

X = df.drop(["attack_cat", "label"], axis=1)
y = df["attack_cat"]

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=0.125,
    random_state=42,
    stratify=y_trainval
)

print("\nTrain Size :", X_train.shape)
print("Val Size   :", X_val.shape)
print("Test Size  :", X_test.shape)


# ======================================
# Step 6 - Encode categorical
# ======================================

cat_cols = X_train.select_dtypes(include="object").columns.tolist()
encoders = {}

for col in cat_cols:
    encoders[col] = LabelEncoder()

    X_train[col] = encoders[col].fit_transform(X_train[col])

    for df_part in [X_val, X_test]:
        df_part[col] = df_part[col].map(
            lambda v, enc=encoders[col]:
            enc.transform([v])[0] if v in enc.classes_ else -1
        )

categorical_indices = [X_train.columns.get_loc(c) for c in cat_cols]


# ======================================
# Step 7 - Encode Labels
# ======================================

label_encoder = LabelEncoder()

y_train_enc = label_encoder.fit_transform(y_train)
y_val_enc = label_encoder.transform(y_val)
y_test_enc = label_encoder.transform(y_test)

print("\nClass Mapping:")
for i, c in enumerate(label_encoder.classes_):
    print(i, "->", c)


# ======================================
# Step 8 - Class weights
# ======================================

classes = np.unique(y_train_enc)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train_enc
)

weight_dict = dict(zip(classes, weights))


# ======================================
# Step 9 - SMOTENC
# ======================================

print("\nApplying targeted SMOTENC...")

counts = pd.Series(y_train_enc).value_counts()
max_count = counts.max()

dos_idx = label_encoder.transform(["DoS"])[0]

sampling_strategy = {cls: max_count for cls in classes}

dos_boosted = min(int(counts[dos_idx]*3), int(max_count*2))
sampling_strategy[dos_idx] = dos_boosted

smote = SMOTENC(
    categorical_features=categorical_indices,
    sampling_strategy=sampling_strategy,
    random_state=42
)

X_smote, y_smote = smote.fit_resample(X_train, y_train_enc)

print("Post SMOTE Counts:")
for cls, cnt in zip(*np.unique(y_smote, return_counts=True)):
    print(label_encoder.classes_[cls], ":", cnt)


# ======================================
# Step 10 - Cross Validation
# ======================================

print("\nRunning Cross Validation...")

xgb_cv = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
    xgb_cv,
    X_smote,
    y_smote,
    cv=cv,
    scoring="recall_macro"
)

print("CV Recall:", scores)
print("Mean Recall:", scores.mean())


# ======================================
# Step 11 - Train Models
# ======================================

print("\nTraining XGBoost...")

xgb = XGBClassifier(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42
)

xgb.fit(X_smote, y_smote)


print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=18,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

rf.fit(X_smote, y_smote)


print("\nTraining LightGBM...")

lgbm = LGBMClassifier(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.05,
    class_weight="balanced",
    random_state=42
)

lgbm.fit(X_smote, y_smote)


# ======================================
# Step 12 - Soft Ensemble
# ======================================

def soft_vote(X):
    return (
        xgb.predict_proba(X) +
        rf.predict_proba(X) +
        lgbm.predict_proba(X)
    ) / 3


# ======================================
# Step 13 - Threshold Tuning
# ======================================

print("\nTuning Thresholds...")

val_proba = soft_vote(X_val)

thresholds = np.ones(len(label_encoder.classes_)) / len(label_encoder.classes_)


def predict_thresh(proba, thresh):
    scaled = proba/(thresh+1e-9)
    return np.argmax(scaled, axis=1)


for i in range(len(thresholds)):
    best = thresholds[i]
    best_recall = -1

    for t in np.linspace(0.05,2,80):
        temp = thresholds.copy()
        temp[i] = t

        pred = predict_thresh(val_proba,temp)

        score = recall_score(y_val_enc,pred,average="macro")

        if score > best_recall:
            best_recall = score
            best = t

    thresholds[i] = best


print("Final Thresholds:", thresholds)


# ======================================
# Step 14 - Evaluation
# ======================================

test_proba = soft_vote(X_test)

ensemble_pred = predict_thresh(test_proba,thresholds)


print("\nFinal Results")

print("Accuracy:",
accuracy_score(y_test_enc,ensemble_pred))

print("Macro Recall:",
recall_score(y_test_enc,ensemble_pred,average="macro"))

print("\nClassification Report\n")

print(classification_report(
y_test_enc,
ensemble_pred,
target_names=label_encoder.classes_
))


# ======================================
# Step 15 - Save Models
# ======================================

print("\nSaving Models...")

joblib.dump(xgb, MODEL_DIR / "stage2_major_xgb.pkl")
joblib.dump(rf, MODEL_DIR / "stage2_major_rf.pkl")
joblib.dump(lgbm, MODEL_DIR / "stage2_major_lgbm.pkl")

joblib.dump(label_encoder, MODEL_DIR / "stage2_major_label.pkl")
joblib.dump(encoders, MODEL_DIR / "stage2_major_encoders.pkl")
joblib.dump(thresholds, MODEL_DIR / "stage2_major_thresholds.pkl")

print("Stage-2 Major Saved")