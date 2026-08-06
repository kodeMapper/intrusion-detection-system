import numpy as np
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# Load dataset

ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT / "data" / "raw" / "UNSW_NB15_training-set.csv"
MODEL_DIR = ROOT / "service" / "models" / "artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Drop weak attacks

drop_attacks = ["Analysis","Backdoor","Shellcode","Worms"]
df = df[~df["attack_cat"].isin(drop_attacks)]

# Drop columns

drop_cols = ["srcip","dstip","id","Stime","Ltime"]
df = df.drop(columns=drop_cols, errors="ignore")

# Split

X = df.drop(["attack_cat","label"], axis=1)
y = df["attack_cat"]

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,stratify=y,random_state=42
)

# Encode categorical

encoders = {}
cat_cols = X_train.select_dtypes(include="object").columns

for col in cat_cols:
    enc = LabelEncoder()
    X_train[col] = enc.fit_transform(X_train[col])
    X_test[col] = X_test[col].map(
        lambda x: enc.transform([x])[0] if x in enc.classes_ else 0
    )
    encoders[col] = enc


# Encode labels

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)


# Scale

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply SMOTE to balance minority classes in training data
print("\nApplying SMOTE for class imbalance...")
print(f"Before SMOTE - Training samples: {X_train.shape[0]}")
print(f"Class distribution before SMOTE: {np.bincount(y_train)}")

smote = SMOTE(random_state=42, k_neighbors=5)
X_train, y_train = smote.fit_resample(X_train, y_train)

print(f"After SMOTE - Training samples: {X_train.shape[0]}")
print(f"Class distribution after SMOTE: {np.bincount(y_train)}")


# Compute class weights to handle imbalance
classes = np.unique(y_train)
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight = {cls: w for cls, w in zip(classes, weights)}

print("\nClass Weights (to balance minority classes):")
for cls, w in class_weight.items():
    print(f"  Class {cls}: {w:.2f}")

# Deep Learning Model with improved architecture

model = Sequential()

model.add(Dense(512, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(256, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(128, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(64, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(len(label_encoder.classes_), activation="softmax"))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nTraining Deep Learning Model with class weighting...")

# Callbacks for better training
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

lr_reduce = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

model.fit(
    X_train,
    y_train,
    epochs=150,
    batch_size=256,
    validation_split=0.1,
    class_weight=class_weight,
    callbacks=[early_stop, lr_reduce],
    verbose=1
)

# Prediction

preds = model.predict(X_test)
preds = np.argmax(preds, axis=1)

print("\nAccuracy")
print(accuracy_score(y_test, preds))

print("\nMacro Recall")
print(recall_score(y_test, preds, average="macro"))

print("\nClassification Report")
print(
    classification_report(
        y_test,
        preds,
        target_names=label_encoder.classes_
    )
)

# Save

model.save(MODEL_DIR / "dl_model.h5")

joblib.dump(scaler, MODEL_DIR / "dl_scaler.pkl")
joblib.dump(encoders, MODEL_DIR / "dl_encoders.pkl")
joblib.dump(label_encoder, MODEL_DIR / "dl_labels.pkl")

print("DL Model Saved")