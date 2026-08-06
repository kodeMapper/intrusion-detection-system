# ======================================
# Step 1 - Imports
# ======================================

import numpy as np
import pandas as pd
from pathlib import Path

from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score

from sklearn.utils.class_weight import compute_class_weight

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

import joblib

RANDOM_STATE = 42


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
# Step 4 - Merge Analysis + Backdoor
# ======================================

df["attack_cat"] = df["attack_cat"].replace({
    "Analysis": "Backdoor"
})


# ======================================
# Step 5 - Rare Classes
# ======================================

rare_classes = [
    "Backdoor",
    "Shellcode",
    "Worms"
]

df = df[df["attack_cat"].isin(rare_classes)].copy()

print("\nRare Dataset:")
print(df["attack_cat"].value_counts())


# ======================================
# Step 6 - Split
# ======================================

X = df.drop(["attack_cat", "label"], axis=1)
y = df["attack_cat"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

X_train = X_train.copy()
X_test = X_test.copy()

print("\nTrain Size:", X_train.shape)
print("Test Size :", X_test.shape)


# ======================================
# Step 7 - Encode Categorical
# ======================================

cat_cols = X_train.select_dtypes(include="object").columns.tolist()
categorical_indices = [X_train.columns.get_loc(col) for col in cat_cols]

encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

if cat_cols:
    X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = encoder.transform(X_test[cat_cols])


# ======================================
# Step 8 - Encode Target
# ======================================

label_encoder = LabelEncoder()

y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)

print("\nClass Mapping:")
for i, cls in enumerate(label_encoder.classes_):
    print(i, "->", cls)


# ======================================
# Step 9 - SMOTENC Setup
# ======================================

train_counts = pd.Series(y_train_enc).value_counts().sort_index()
min_train_count = train_counts.min()

smote_k_neighbors = min(3, min_train_count - 1)

print("\nTraining Class Counts:")
for class_id, count in train_counts.items():
    print(label_encoder.classes_[class_id], ":", count)

print("\nSMOTENC k_neighbors:", smote_k_neighbors)


# ======================================
# Step 10 - Cross Validation
# ======================================

print("\nRunning leakage-safe CV...")

cv_rf = RandomForestClassifier(
    n_estimators=250,
    max_depth=12,
    min_samples_leaf=4,
    max_features="sqrt",
    class_weight="balanced_subsample",
    n_jobs=1,
    random_state=RANDOM_STATE
)

cv_pipeline = ImbPipeline([
    ("smote", SMOTENC(
        categorical_features=categorical_indices,
        random_state=RANDOM_STATE,
        k_neighbors=smote_k_neighbors
    )),
    ("model", cv_rf)
])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

cv_scores = cross_val_score(
    cv_pipeline,
    X_train,
    y_train_enc,
    cv=cv,
    scoring="recall_macro",
    n_jobs=1
)

print("CV Macro Recall:", cv_scores)
print("Mean CV Macro Recall:", cv_scores.mean())


# ======================================
# Step 11 - Apply SMOTENC
# ======================================

print("\nApplying SMOTENC on training data only...")

smote = SMOTENC(
    categorical_features=categorical_indices,
    random_state=RANDOM_STATE,
    k_neighbors=smote_k_neighbors
)

X_train_aug, y_train_aug = smote.fit_resample(
    X_train,
    y_train_enc
)

X_train_aug = pd.DataFrame(
    X_train_aug,
    columns=X_train.columns
)

print("Augmented Class Counts:")
for class_id, count in pd.Series(y_train_aug).value_counts().sort_index().items():
    print(label_encoder.classes_[class_id], ":", count)


# ======================================
# Step 12 - Class Weights
# ======================================

classes = np.unique(y_train_aug)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train_aug
)

weight_dict = dict(zip(classes, weights))

sample_weights = np.array([
    weight_dict[i] for i in y_train_aug
])


# ======================================
# Step 13 - Train XGBoost
# ======================================

print("\nTraining XGBoost...")

xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=len(label_encoder.classes_),
    n_estimators=600,
    max_depth=4,
    learning_rate=0.04,
    min_child_weight=3,
    gamma=0.2,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=3.0,
    reg_alpha=0.5,
    eval_metric="mlogloss",
    tree_method="hist",
    n_jobs=-1,
    random_state=RANDOM_STATE
)

xgb.fit(
    X_train_aug,
    y_train_aug,
    sample_weight=sample_weights
)


# ======================================
# Step 14 - Train Random Forest
# ======================================

print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    min_samples_leaf=4,
    max_features="sqrt",
    class_weight="balanced_subsample",
    n_jobs=1,
    random_state=RANDOM_STATE
)

rf.fit(
    X_train_aug,
    y_train_aug
)


# ======================================
# Step 15 - Evaluation
# ======================================

def evaluate(name, model):

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_macro = recall_score(y_train_enc, train_pred, average="macro")
    test_macro = recall_score(y_test_enc, test_pred, average="macro")

    print("\n======================")
    print(name)
    print("======================")

    print("Train Accuracy:",
          accuracy_score(y_train_enc, train_pred))

    print("Test Accuracy:",
          accuracy_score(y_test_enc, test_pred))

    print("Train Macro Recall:", train_macro)
    print("Test Macro Recall :", test_macro)

    print("Recall Gap:", train_macro - test_macro)

    print("\nClassification Report\n")

    print(
        classification_report(
            y_test_enc,
            test_pred,
            target_names=label_encoder.classes_,
            zero_division=0
        )
    )


evaluate("Rare XGBoost", xgb)
evaluate("Rare Random Forest", rf)


# ======================================
# Step 16 - Save Models
# ======================================

print("\nSaving Stage-2 Rare Models...")

joblib.dump(xgb, MODEL_DIR / "stage2_rare_xgb.pkl")
joblib.dump(rf, MODEL_DIR / "stage2_rare_rf.pkl")
joblib.dump(label_encoder, MODEL_DIR / "stage2_rare_label.pkl")
joblib.dump(encoder, MODEL_DIR / "stage2_rare_encoder.pkl")

print("Stage-2 Rare Models Saved")