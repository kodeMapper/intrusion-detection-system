# =====================================
# Imports
# =====================================

import numpy as np
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

from sklearn.feature_selection import SelectFromModel

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier


# =====================================
# Load Dataset
# =====================================

ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = ROOT / "data" / "raw" / "UNSW_NB15_training-set.csv"
MODEL_DIR = ROOT / "service" / "models" / "artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Loading Dataset...")

df = pd.read_csv(DATA_PATH)

print("Original Shape:", df.shape)


# =====================================
# Drop Weak Attacks
# =====================================

drop_attacks = [
    "Analysis",
    "Backdoor",
    "Shellcode",
    "Worms"
]

df = df[~df["attack_cat"].isin(drop_attacks)]

print("\nAfter Dropping Weak Attacks")
print(df["attack_cat"].value_counts())


# =====================================
# Drop Non Live Features
# =====================================

drop_cols = [
    "srcip",
    "dstip",
    "id",
    "Stime",
    "Ltime"
]

df = df.drop(columns=drop_cols, errors="ignore")


# =====================================
# Feature Split
# =====================================

X = df.drop(["attack_cat", "label"], axis=1)
y = df["attack_cat"]


# =====================================
# Train Test Split
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =====================================
# Encode Categorical
# =====================================

cat_cols = X_train.select_dtypes(include="object").columns

encoders = {}

for col in cat_cols:

    enc = LabelEncoder()

    X_train[col] = enc.fit_transform(X_train[col])

    X_test[col] = X_test[col].map(
        lambda x:
        enc.transform([x])[0]
        if x in enc.classes_
        else 0
    )

    encoders[col] = enc


# =====================================
# Encode Labels
# =====================================

label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)


print("\nClass Mapping")

for i, label in enumerate(label_encoder.classes_):

    print(i, "->", label)


# =====================================
# Feature Selection
# =====================================

print("\nTraining Initial XGBoost")

xgb_initial = XGBClassifier(
    n_estimators=300,
    max_depth=6
)

xgb_initial.fit(X_train, y_train)

selector = SelectFromModel(
    xgb_initial,
    threshold="median"
)

X_train = selector.transform(X_train)
X_test = selector.transform(X_test)


# =====================================
# Train Models
# =====================================

print("\nTraining XGBoost")

xgb = XGBClassifier(
    n_estimators=700,
    max_depth=9,
    learning_rate=0.04,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method="hist"
)

xgb.fit(X_train, y_train)


print("\nTraining Random Forest")

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=25,
    class_weight="balanced",
    n_jobs=-1
)

rf.fit(X_train, y_train)


print("\nTraining LightGBM")

lgbm = LGBMClassifier(
    n_estimators=700,
    learning_rate=0.04,
    class_weight="balanced"
)

lgbm.fit(X_train, y_train)


# =====================================
# Soft Voting Ensemble
# =====================================

print("\nRunning Ensemble")

proba = (
    xgb.predict_proba(X_test) +
    rf.predict_proba(X_test) +
    lgbm.predict_proba(X_test)
) / 3

preds = np.argmax(proba, axis=1)


# =====================================
# Evaluation
# =====================================

print("\nAccuracy")
print(accuracy_score(y_test, preds))


print("\nMacro Recall")
print(recall_score(y_test, preds, average="macro"))


print("\nMicro Recall (Total Recall)")
print(recall_score(y_test, preds, average="micro"))


print("\nClassification Report")

print(
    classification_report(
        y_test,
        preds,
        target_names=label_encoder.classes_
    )
)


# =====================================
# Save Models
# =====================================

print("\nSaving Models")

joblib.dump(xgb, MODEL_DIR / "final_xgb.pkl")
joblib.dump(rf, MODEL_DIR / "final_rf.pkl")
joblib.dump(lgbm, MODEL_DIR / "final_lgbm.pkl")

joblib.dump(selector, MODEL_DIR / "feature_selector.pkl")
joblib.dump(encoders, MODEL_DIR / "final_encoders.pkl")
joblib.dump(label_encoder, MODEL_DIR / "final_labels.pkl")

print("Model Saved Successfully")
