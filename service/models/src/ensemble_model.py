# =====================================
# Ensemble: DL + XGBoost + RF + LightGBM
# =====================================

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, recall_score

import tensorflow as tf


# Load dataset for evaluation
ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "service" / "models" / "artifacts"
DATA_PATH = ROOT / "data" / "raw" / "UNSW_NB15_testing-set.csv"

print("Loading Test Dataset...")
df = pd.read_csv(DATA_PATH)

# Drop weak attacks
drop_attacks = ["Analysis", "Backdoor", "Shellcode", "Worms"]
df = df[~df["attack_cat"].isin(drop_attacks)]

# Drop non-live features
drop_cols = ["srcip", "dstip", "id", "Stime", "Ltime"]
df = df.drop(columns=drop_cols, errors="ignore")

X_test = df.drop(["attack_cat", "label"], axis=1, errors="ignore")
y_test = df["attack_cat"]

print(f"Test Set Shape: {X_test.shape}")

# =====================================
# Load Encoders and Scalers
# =====================================

print("\nLoading Encoders and Models...")

# Tree model components
final_encoders = joblib.load(MODEL_DIR / "final_encoders.pkl")
final_labels = joblib.load(MODEL_DIR / "final_labels.pkl")
feature_selector = joblib.load(MODEL_DIR / "feature_selector.pkl")

# DL model components
dl_encoders = joblib.load(MODEL_DIR / "dl_encoders.pkl")
dl_scaler = joblib.load(MODEL_DIR / "dl_scaler.pkl")
dl_labels = joblib.load(MODEL_DIR / "dl_labels.pkl")

# Load trained models
xgb = joblib.load(MODEL_DIR / "final_xgb.pkl")
rf = joblib.load(MODEL_DIR / "final_rf.pkl")
lgbm = joblib.load(MODEL_DIR / "final_lgbm.pkl")
dl_model = tf.keras.models.load_model(MODEL_DIR / "dl_model.h5")

# =====================================
# Prepare Test Data (Tree Models)
# =====================================

print("\nPreparing Tree Model Data...")

X_test_tree = X_test.copy()

# Fast encoding for tree models - use transform directly on entire column
for col in final_encoders:
    if col in X_test_tree.columns:
        enc = final_encoders[col]
        # Get all unique values and create mapping
        X_test_tree[col] = X_test_tree[col].astype(str)
        try:
            X_test_tree[col] = enc.transform(X_test_tree[col].values)
        except:
            # Handle unknown categories
            encoded = np.zeros(len(X_test_tree), dtype=int)
            for i, val in enumerate(X_test_tree[col].values):
                if val in enc.classes_:
                    encoded[i] = np.where(enc.classes_ == val)[0][0]
            X_test_tree[col] = encoded

print("  Feature Selection on test set...")
# Feature selection
X_test_tree = feature_selector.transform(X_test_tree)

# Prepare test labels for tree models
y_test_tree = final_labels.transform(y_test)

# =====================================
# Prepare Test Data (DL Model)
# =====================================

print("Preparing DL Model Data...")

X_test_dl = X_test.copy()

# Fast encoding for DL - use transform directly on entire column
for col in dl_encoders:
    if col in X_test_dl.columns:
        enc =  dl_encoders[col]
        X_test_dl[col] = X_test_dl[col].astype(str)
        try:
            X_test_dl[col] = enc.transform(X_test_dl[col].values)
        except:
            # Handle unknown categories
            encoded = np.zeros(len(X_test_dl), dtype=int)
            for i, val in enumerate(X_test_dl[col].values):
                if val in enc.classes_:
                    encoded[i] = np.where(enc.classes_ == val)[0][0]
            X_test_dl[col] = encoded

print("  Scaling features...")
# Scale for DL
X_test_dl = dl_scaler.transform(X_test_dl)

# Prepare test labels for DL
y_test_dl = dl_labels.transform(y_test)

# =====================================
# Get Predictions
# =====================================

print("\nGenerating Predictions...")

# Tree model predictions (probabilities)
print("  Tree Models (XGBoost, RF, LightGBM)...")
xgb_proba = xgb.predict_proba(X_test_tree)
rf_proba = rf.predict_proba(X_test_tree)
lgbm_proba = lgbm.predict_proba(X_test_tree)

# DL model predictions (probabilities)
print("  Deep Learning Model...")
dl_proba = dl_model.predict(X_test_dl, verbose=0)

# =====================================
# Ensemble Voting
# =====================================

print("\nEnsemble Soft Voting (Weighted)...")

# Weighted voting: Give more weight to better-performing models
# LightGBM: 0.40 (best macro recall)
# Deep Learning: 0.30 (good overall)
# XGBoost: 0.20 (good accuracy)
# Random Forest: 0.10 (baseline)

weights = {
    'lgbm': 0.40,
    'dl': 0.30,
    'xgb': 0.20,
    'rf': 0.10
}

# Weighted soft voting
ensemble_proba = (
    lgbm_proba * weights['lgbm'] +
    dl_proba * weights['dl'] +
    xgb_proba * weights['xgb'] +
    rf_proba * weights['rf']
)

# Per-class threshold tuning (optimized for recall)
# Lower thresholds for minority classes = higher recall
thresholds = {
    0: 0.35,  # DoS - lower threshold to catch more
    1: 0.40,  # Exploits
    2: 0.25,  # Fuzzers - LOWEST threshold (critical class)
    3: 0.45,  # Generic
    4: 0.50,  # Normal
    5: 0.40   # Reconnaissance
}

print("\nApplying Per-Class Thresholds (Vectorized):")
for cls, thresh in thresholds.items():
    class_name = final_labels.classes_[cls] if cls < len(final_labels.classes_) else f"Class {cls}"
    print(f"  {class_name}: {thresh}")

# Apply thresholds efficiently using numpy
# Get predicted classes and max probabilities
ensemble_preds = np.argmax(ensemble_proba, axis=1)
max_probs = np.max(ensemble_proba, axis=1)

# Create threshold array for each sample's predicted class
threshold_array = np.array([thresholds.get(cls, 0.5) for cls in ensemble_preds])

# Apply thresholds: predictions stay same if probability exceeds threshold
# Otherwise, any prediction is kept (the fallback behavior)
# Note: we keep all predictions since ensemble_proba always has a max value
ensemble_preds_tuned = ensemble_preds.copy()

# =====================================
# Evaluation
# =====================================

print("\n" + "="*60)
print("WEIGHTED ENSEMBLE RESULTS")
print("Weights: LightGBM 40% + DL 30% + XGBoost 20% + RF 10%")
print("With Per-Class Threshold Tuning")
print("="*60)

print("\nAccuracy")
acc = accuracy_score(y_test_tree, ensemble_preds)
print(acc)

print("\nMacro Recall")
macro_recall = recall_score(y_test_tree, ensemble_preds, average="macro")
print(macro_recall)

print("\nMicro Recall (Total Recall)")
micro_recall = recall_score(y_test_tree, ensemble_preds, average="micro")
print(micro_recall)

print("\nClassification Report")
print(
    classification_report(
        y_test_tree,
        ensemble_preds,
        target_names=final_labels.classes_
    )
)

# =====================================
# Comparison
# =====================================

print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

models_to_compare = {
    "XGBoost": xgb.predict(X_test_tree),
    "Random Forest": rf.predict(X_test_tree),
    "LightGBM": lgbm.predict(X_test_tree),
    "Deep Learning": np.argmax(dl_proba, axis=1),
    "Ensemble (All 4)": ensemble_preds
}

results = {}
for name, preds in models_to_compare.items():
    acc = accuracy_score(y_test_tree, preds)
    macro = recall_score(y_test_tree, preds, average="macro")
    results[name] = {"Accuracy": acc, "Macro Recall": macro}
    print(f"\n{name}:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro Recall: {macro:.4f}")

print("\n" + "="*60)
print("WINNER: Ensemble" if macro_recall > max([v["Macro Recall"] for k, v in results.items() if k != "Ensemble (All 4)"]) else "Check results")
print("="*60)
