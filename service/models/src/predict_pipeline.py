# =====================================
# Imports
# =====================================

import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# =====================================
# Load Models
# =====================================

print("Loading IDS Model...")

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "service" / "models" / "artifacts"

model_xgb = joblib.load(MODEL_DIR / "final_xgb.pkl")
model_rf = joblib.load(MODEL_DIR / "final_rf.pkl")
model_lgbm = joblib.load(MODEL_DIR / "final_lgbm.pkl")

encoders = joblib.load(MODEL_DIR / "final_encoders.pkl")
label_encoder = joblib.load(MODEL_DIR / "final_labels.pkl")

print("Models Loaded Successfully")


# =====================================
# Encoding Function
# =====================================

def encode_features(df):

    df = df.copy()

    for col in encoders:

        if col in df.columns:

            le = encoders[col]

            df[col] = df[col].apply(
                lambda x:
                le.transform([x])[0]
                if x in le.classes_
                else 0
            )

    return df


# =====================================
# Prediction Function
# =====================================

def predict(packet_df):

    packet_df = encode_features(packet_df)

    proba = (
        model_xgb.predict_proba(packet_df) +
        model_rf.predict_proba(packet_df) +
        model_lgbm.predict_proba(packet_df)
    ) / 3

    preds = np.argmax(proba, axis=1)

    return label_encoder.inverse_transform(preds)


# =====================================
# Testing With Dataset
# =====================================

if __name__ == "__main__":

    print("\nTesting Pipeline...")

    TEST_PATH = ROOT / "data" / "raw" / "UNSW_NB15_testing-set.csv"

    print(f"Loading test data from {TEST_PATH}...")

    df = pd.read_csv(TEST_PATH)

    print(f"Loaded {len(df):,} samples")


    # =====================================
    # Prepare Features
    # =====================================

    y_true = df["attack_cat"]

    drop_cols = [
        "srcip",
        "dstip",
        "id",
        "Stime",
        "Ltime",
        "attack_cat",
        "label"
    ]

    X = df.drop(columns=drop_cols, errors="ignore")


    # =====================================
    # Predict
    # =====================================

    print("Running predictions (this may take few minutes)...")

    preds = predict(X)


    # =====================================
    # Results
    # =====================================

    print("\nAccuracy:")
    print(accuracy_score(y_true, preds))


    print("\nClassification Report:\n")

    print(
        classification_report(
            y_true,
            preds,
            zero_division=0
        )
    )