# ======================================
# Step 1 - Imports
# ======================================

import pandas as pd
from pathlib import Path

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ======================================
# Step 2 - Load Dataset
# ======================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "UNSW_NB15_training-set.csv"

df = pd.read_csv(DATASET_PATH)

print("Dataset Shape:", df.shape)


# ======================================
# Step 3 - Drop Non Live Features
# ======================================

drop_cols = ["srcip", "dstip", "id", "Stime", "Ltime"]
df = df.drop(columns=drop_cols, errors="ignore")

print("After Drop:", df.shape)


# ======================================
# Step 4 - Feature Split
# ======================================

X = df.drop(["attack_cat", "label"], axis=1)
y = df["attack_cat"]

print("Features:", X.shape)
print("\nFull dataset class distribution:")
print(y.value_counts())


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

print("\nTrain size:", X_train.shape)
print("Test size :", X_test.shape)


# ======================================
# Step 6 - Encode Categorical Features
# ======================================
# Fit only on train, transform both to prevent leakage

encoders = {}
for col in X_train.select_dtypes(include="object").columns:
    encoders[col] = LabelEncoder()
    X_train[col] = encoders[col].fit_transform(X_train[col])
    X_test[col] = X_test[col].map(
        lambda val, enc=encoders[col]: (
            enc.transform([val])[0] if val in enc.classes_ else -1
        )
    )


# ======================================
# Step 7 - Balance with SMOTETomek
# ======================================
# SMOTE synthesizes new minority samples by interpolating between
# real neighbours — far better than duplicating rows.
# Tomek links then clean borderline majority samples.
#
# sampling_strategy dict: only oversample classes BELOW target.
# Majority classes (Normal, Generic, Exploits) are left untouched
# here and handled via class_weight in the model instead —
# this preserves their real distribution while lifting minorities.

TARGET_MINORITY = 5000

counts = y_train.value_counts()
sampling_strategy = {
    cls: TARGET_MINORITY
    for cls, count in counts.items()
    if count < TARGET_MINORITY
}

print("\nSMOTE sampling targets:", sampling_strategy)
print("\nApplying SMOTETomek (this may take ~1-2 min)...")

smote_tomek = SMOTETomek(
    smote=SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=5,          # neighbours used for synthesis
        random_state=42,
    ),
    random_state=42,
)

X_train, y_train = smote_tomek.fit_resample(X_train, y_train)

print("Done.")
print("\nTraining class distribution after SMOTETomek:")
print(pd.Series(y_train).value_counts())


# ======================================
# Step 8 - Train Model
# ======================================

model = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced",    # handles remaining majority imbalance
    random_state=42,
    min_samples_leaf=1,
    max_features="sqrt",
    n_jobs=-1,
)

print("\nTraining model...")
model.fit(X_train, y_train)
print("Training complete.")


# ======================================
# Step 9 - Prediction
# ======================================

y_pred = model.predict(X_test)


# ======================================
# Step 10 - Evaluation
# ======================================

print("\n" + "=" * 50)
print("EVALUATION RESULTS")
print("=" * 50)

print("\nAccuracy              :", accuracy_score(y_test, y_pred))
print("Balanced Accuracy     :", balanced_accuracy_score(y_test, y_pred))
print("Weighted Recall       :", recall_score(y_test, y_pred, average="weighted", zero_division=0))
print("Weighted F1           :", f1_score(y_test, y_pred, average="weighted", zero_division=0))
print("Macro Recall          :", recall_score(y_test, y_pred, average="macro", zero_division=0))
print("Macro F1              :", f1_score(y_test, y_pred, average="macro", zero_division=0))

all_classes = sorted(y_test.unique().tolist())
classes_excl_worms = [c for c in all_classes if c != "Worms"]

print(
    "\nMacro Recall (excl. Worms):",
    recall_score(y_test, y_pred, labels=classes_excl_worms, average="macro", zero_division=0),
)
print(
    "Macro F1     (excl. Worms):",
    f1_score(y_test, y_pred, labels=classes_excl_worms, average="macro", zero_division=0),
)

print("\nClassification Report\n")
print(classification_report(y_test, y_pred, zero_division=0))


# ======================================
# Step 11 - Feature Importance
# ======================================

feature_cols = X_train.columns if hasattr(X_train, "columns") else [f"f{i}" for i in range(X_train.shape[1])]
importances = pd.Series(model.feature_importances_, index=feature_cols)
top_features = importances.sort_values(ascending=False).head(15)

print("\nTop 15 Feature Importances:")
print(top_features.to_string())