import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from dl_pipeline import TemporalTransformerClassifier, get_device

ROOT = Path(__file__).resolve().parents[3]
SEQUENCES_DIR = ROOT / "data" / "sequences"
MODEL_ARTIFACTS_DIR = ROOT / "service" / "models" / "artifacts"
MODEL_PATH = MODEL_ARTIFACTS_DIR / "dl_stage1_binary_v1.0.pt"
THRESHOLD_PATH = MODEL_ARTIFACTS_DIR / "dl_stage1_binary_threshold_v1.0.json"
REPORT_PATH = MODEL_ARTIFACTS_DIR / "dl_stage1_binary_eval_report_v1.0.json"


def load_split(split_prefix: str, normal_label: int):
    X = np.load(SEQUENCES_DIR / f"{split_prefix}_w10_v1_20260612.npy").astype(np.float32)
    y_multi = np.load(SEQUENCES_DIR / f"{split_prefix}_labels_w10_v1_20260612.npy")
    y_binary = (y_multi != normal_label).astype(np.int64)
    return X, y_binary


@torch.no_grad()
def predict(model, X: np.ndarray, batch_size: int, device) -> np.ndarray:
    dataset = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    probs = []
    model.eval()
    for (X_batch,) in loader:
        logits = model(X_batch.to(device))
        probs.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
    return np.concatenate(probs)


def evaluate_split(model, split_prefix: str, split_name: str, normal_label: int, threshold: float, device):
    X, y_true = load_split(split_prefix, normal_label)
    attack_prob = predict(model, X, batch_size=512, device=device)
    y_pred = (attack_prob >= threshold).astype(int)
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Attack"],
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    print(f"--- {split_name} Split ---")
    print("Confusion Matrix:")
    print(np.array(matrix))
    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Normal", "Attack"],
            digits=4,
            zero_division=0,
        )
    )
    print("-" * 50)
    return {"confusion_matrix": matrix, "classification_report": report}


def main():
    device = get_device()
    with open(THRESHOLD_PATH, "r") as f:
        threshold_config = json.load(f)
    threshold = threshold_config["attack_probability_threshold"]
    normal_label = int(threshold_config["normal_label"])

    model = TemporalTransformerClassifier(**threshold_config["model_hparams"])
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)

    reports = {
        "train": evaluate_split(model, "lstm_train", "Train", normal_label, threshold, device),
        "validation": evaluate_split(model, "lstm_val", "Validation", normal_label, threshold, device),
        "test": evaluate_split(model, "lstm_test", "Test", normal_label, threshold, device),
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(reports, f, indent=4)
    print(f"Saved Stage 1 eval report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
