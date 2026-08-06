# ======================================
# Step 1 - Imports
# ======================================

import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


# ======================================
# Step 2 - Load Dataset
# ======================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "UNSW_NB15_training-set.csv"

df = pd.read_csv(DATASET_PATH)
print("Dataset Shape:", df.shape)


# ======================================
# Step 3 - Drop Non Live Features
# ======================================

drop_cols = ["srcip", "dstip", "id", "Stime", "Ltime", "attack_cat"]
df = df.drop(columns=drop_cols, errors="ignore")
print("After Drop:", df.shape)


# ======================================
# Step 4 - Feature Split
# ======================================

X = df.drop("label", axis=1)
y = df["label"]

print("Features:", X.shape)
print("\nFull dataset class distribution:")
print(y.value_counts().rename(index={0: "Normal", 1: "Attack"}))


# ======================================
# Step 5 - Train Test Split
# ======================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

X_train = X_train.copy()
X_test = X_test.copy()

print("\nTrain size:", X_train.shape)
print("Test size :", X_test.shape)


# ======================================
# Step 6 - Encode Categorical Features
# ======================================

cat_encoders = {}
for col in X_train.select_dtypes(include="object").columns:
    cat_encoders[col] = LabelEncoder()
    X_train[col] = cat_encoders[col].fit_transform(X_train[col])

    value_map = {
        class_name: class_id
        for class_id, class_name in enumerate(cat_encoders[col].classes_)
    }
    X_test[col] = X_test[col].map(value_map).fillna(-1).astype(int)


# ======================================
# Step 7 - Train Binary XGBoost Model
# ======================================

model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    min_child_weight=2,
    gamma=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
    device="cpu",
)

print("\nTraining binary XGBoost model...")
model.fit(X_train, y_train)
print("Training complete.")


# ======================================
# Step 8 - Prediction
# ======================================

y_pred = model.predict(X_test)


# ======================================
# Step 9 - Evaluation
# ======================================

print("\n" + "=" * 50)
print("BINARY NIDS EVALUATION RESULTS")
print("=" * 50)

print("\nAccuracy              :", accuracy_score(y_test, y_pred))
print("Balanced Accuracy     :", balanced_accuracy_score(y_test, y_pred))
print("Attack Precision      :", precision_score(y_test, y_pred, pos_label=1, zero_division=0))
print("Attack Recall         :", recall_score(y_test, y_pred, pos_label=1, zero_division=0))
print("Attack F1             :", f1_score(y_test, y_pred, pos_label=1, zero_division=0))
print("Weighted Recall       :", recall_score(y_test, y_pred, average="weighted", zero_division=0))
print("Weighted F1           :", f1_score(y_test, y_pred, average="weighted", zero_division=0))

print("\nClassification Report\n")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Normal", "Attack"],
    zero_division=0,
))


# ======================================
# Step 10 - Feature Importance
# ======================================

importances = pd.Series(model.feature_importances_, index=X_train.columns)
top_features = importances.sort_values(ascending=False).head(15)

print("\nTop 15 Feature Importances:")
print(top_features.to_string())
