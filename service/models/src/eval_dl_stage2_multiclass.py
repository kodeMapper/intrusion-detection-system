import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

from dl_pipeline import TemporalTransformerClassifier, get_device

ROOT = Path(__file__).resolve().parents[3]
SEQUENCES_DIR = ROOT / "data" / "sequences"
MODEL_ARTIFACTS_DIR = ROOT / "service" / "models" / "artifacts"
MODEL_PATH = MODEL_ARTIFACTS_DIR / "dl_stage2_multiclass_v1.0.pt"
CONFIG_PATH = MODEL_ARTIFACTS_DIR / "dl_stage2_multiclass_config_v1.0.json"
REPORT_PATH = MODEL_ARTIFACTS_DIR / "dl_stage2_multiclass_eval_report_v1.0.json"


def load_attack_split(split_prefix: str, normal_label: int, original_to_stage2: dict[int, int]):
    X = np.load(SEQUENCES_DIR / f"{split_prefix}_w10_v1_20260612.npy").astype(np.float32)
    y_original = np.load(SEQUENCES_DIR / f"{split_prefix}_labels_w10_v1_20260612.npy")
    attack_mask = y_original != normal_label
    X_attack = X[attack_mask]
    y_attack_original = y_original[attack_mask]
    y_attack = np.array([original_to_stage2[int(label)] for label in y_attack_original], dtype=np.int64)
    return X_attack, y_attack


@torch.no_grad()
def predict_proba(model, X: np.ndarray, batch_size: int, device) -> np.ndarray:
    dataset = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    probs = []
    model.eval()
    for (X_batch,) in loader:
        logits = model(X_batch.to(device))
        probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
    return np.concatenate(probs)


def thresholded_predictions(probs: np.ndarray, confidence_threshold: float, num_classes: int) -> np.ndarray:
    preds = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)
    unclassified_label = num_classes
    preds = preds.copy()
    preds[confidence < confidence_threshold] = unclassified_label
    return preds


def evaluate_split(model, split_prefix: str, split_name: str, config: dict, device):
    attack_classes = config["attack_classes"]
    normal_label = int(config["normal_label"])
    original_to_stage2 = {int(k): int(v) for k, v in config["original_to_stage2_label"].items()}
    confidence_threshold = float(config["confidence_policy"]["confidence_threshold"])

    X, y_true = load_attack_split(split_prefix, normal_label, original_to_stage2)
    probs = predict_proba(model, X, batch_size=512, device=device)
    argmax_pred = np.argmax(probs, axis=1)
    thresholded_pred = thresholded_predictions(probs, confidence_threshold, len(attack_classes))

    argmax_report = classification_report(
        y_true,
        argmax_pred,
        labels=list(range(len(attack_classes))),
        target_names=attack_classes,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    argmax_matrix = confusion_matrix(y_true, argmax_pred, labels=list(range(len(attack_classes)))).tolist()

    thresholded_names = attack_classes + ["Attack-Unclassified"]
    thresholded_report = classification_report(
        y_true,
        thresholded_pred,
        labels=list(range(len(thresholded_names))),
        target_names=thresholded_names,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    thresholded_matrix = confusion_matrix(y_true, thresholded_pred, labels=list(range(len(thresholded_names)))).tolist()

    print(f"--- {split_name} Split ---")
    print("Argmax Confusion Matrix:")
    print(np.array(argmax_matrix))
    print("\nArgmax Classification Report:")
    print(
        classification_report(
            y_true,
            argmax_pred,
            labels=list(range(len(attack_classes))),
            target_names=attack_classes,
            digits=4,
            zero_division=0,
        )
    )
    print(f"Coverage at confidence threshold {confidence_threshold:.3f}: {np.mean(thresholded_pred != len(attack_classes)):.4f}")
    print("-" * 60)

    return {
        "argmax": {
            "confusion_matrix": argmax_matrix,
            "classification_report": argmax_report,
            "macro_recall": float(recall_score(y_true, argmax_pred, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(y_true, argmax_pred, average="macro", zero_division=0)),
            "per_class_recall": {name: float(argmax_report[name]["recall"]) for name in attack_classes},
        },
        "thresholded": {
            "confidence_threshold": confidence_threshold,
            "confusion_matrix": thresholded_matrix,
            "classification_report": thresholded_report,
            "coverage": float(np.mean(thresholded_pred != len(attack_classes))),
            "unclassified_rate": float(np.mean(thresholded_pred == len(attack_classes))),
        },
    }


def main():
    device = get_device()
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    model = TemporalTransformerClassifier(**config["model_hparams"])
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)

    reports = {
        "train": evaluate_split(model, "lstm_train", "Train", config, device),
        "validation": evaluate_split(model, "lstm_val", "Validation", config, device),
        "test": evaluate_split(model, "lstm_test", "Test", config, device),
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(reports, f, indent=4)
    print(f"Saved Stage 2 eval report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
