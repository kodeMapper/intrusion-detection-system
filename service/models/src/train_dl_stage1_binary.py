import json
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

torch.set_float32_matmul_precision("medium")

from dl_pipeline import TemporalTransformerClassifier, get_device, set_seed

ROOT = Path(__file__).resolve().parents[3]
SEQUENCES_DIR = ROOT / "data" / "sequences"
DATA_ARTIFACTS_DIR = ROOT / "data" / "artifacts"
MODEL_ARTIFACTS_DIR = ROOT / "service" / "models" / "artifacts"
MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_ARTIFACTS_DIR / "dl_stage1_binary_v1.0.pt"
ONNX_PATH = MODEL_ARTIFACTS_DIR / "dl_stage1_binary_v1.0.onnx"
CHECKPOINT_NAME = "dl_stage1_binary_v1.0"
THRESHOLD_PATH = MODEL_ARTIFACTS_DIR / "dl_stage1_binary_threshold_v1.0.json"
REPORT_PATH = MODEL_ARTIFACTS_DIR / "dl_stage1_binary_report_v1.0.json"


def load_preprocessing_config() -> dict:
    with open(DATA_ARTIFACTS_DIR / "preprocessing_config_unsw_v1_20260612.json", "r") as f:
        config = json.load(f)
    normal_label = config["label_map"]["Normal"]
    if normal_label != 4:
        raise ValueError(f"Expected label_map['Normal'] == 4, got {normal_label}.")
    return config


def load_split(split_prefix: str, normal_label: int):
    X = np.load(SEQUENCES_DIR / f"{split_prefix}_w10_v1_20260612.npy").astype(np.float32)
    y_multi = np.load(SEQUENCES_DIR / f"{split_prefix}_labels_w10_v1_20260612.npy")
    y_binary = (y_multi != normal_label).astype(np.int64)
    return X, y_binary


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.int64, copy=False)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def class_weights_from_binary(y: np.ndarray) -> list[float]:
    counts = np.bincount(y, minlength=2).astype(np.float64)
    weights = counts.sum() / (len(counts) * np.maximum(counts, 1.0))
    weights = weights / weights.mean()
    return weights.astype(np.float32).tolist()


@torch.no_grad()
def predict_attack_probability(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    targets = []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        probs.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
        targets.append(y_batch.numpy())
    return np.concatenate(probs), np.concatenate(targets)


def tune_attack_threshold(
    y_true: np.ndarray,
    attack_prob: np.ndarray,
    min_normal_recall: float = 0.90,
) -> dict:
    candidates = np.unique(np.quantile(attack_prob, np.linspace(0.001, 0.999, 1000)))
    best = {"threshold": 0.5, "score": -np.inf}
    for threshold in candidates:
        y_pred = (attack_prob >= threshold).astype(int)
        normal_mask = y_true == 0
        attack_mask = y_true == 1
        normal_recall = recall_score(y_true[normal_mask], y_pred[normal_mask], pos_label=0, zero_division=0)
        if normal_recall < min_normal_recall:
            continue
        attack_recall = recall_score(y_true[attack_mask], y_pred[attack_mask], pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        score = 0.75 * attack_recall + 0.25 * f1
        if score > best["score"]:
            best = {
                "threshold": float(threshold),
                "score": float(score),
                "normal_recall": float(normal_recall),
                "attack_recall": float(attack_recall),
                "f1": float(f1),
                "precision": float(precision),
            }

    if best["score"] == -np.inf:
        threshold = float(np.quantile(attack_prob[y_true == 0], 0.90))
        y_pred = (attack_prob >= threshold).astype(int)
        best = {
            "threshold": threshold,
            "warning": "constraint_relaxed",
            "normal_recall": float(recall_score(y_true[y_true == 0], y_pred[y_true == 0], pos_label=0, zero_division=0)),
            "attack_recall": float(recall_score(y_true[y_true == 1], y_pred[y_true == 1], pos_label=1, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        }
    return best


def evaluate_split(model, X, y, threshold: float, batch_size: int, device) -> dict:
    loader = make_loader(X, y, batch_size=batch_size, shuffle=False)
    attack_prob, y_true = predict_attack_probability(model, loader, device)
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
    return {
        "confusion_matrix": matrix,
        "classification_report": report,
        "normal_recall": float(report["Normal"]["recall"]),
        "attack_recall": float(report["Attack"]["recall"]),
        "attack_precision": float(report["Attack"]["precision"]),
        "attack_f1": float(report["Attack"]["f1-score"]),
    }


def export_onnx(model, input_dim: int, seq_len: int, device):
    try:
        import onnx  # noqa: F401

        dummy_input = torch.randn(1, seq_len, input_dim, device=device)
        torch.onnx.export(
            model,
            dummy_input,
            ONNX_PATH,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        )
        print(f"Model successfully exported to ONNX: {ONNX_PATH}")
    except Exception as e:
        print(f"Warning: ONNX export failed. PyTorch model was saved, but ONNX export failed: {e}")


def train_stage1_binary():
    set_seed(42)
    config = load_preprocessing_config()
    normal_label = config["label_map"]["Normal"]
    device = get_device()

    batch_size = 256
    epochs = 80
    print(f"Training Stage 1 binary classifier on device: {device}")

    X_train, y_train = load_split("lstm_train", normal_label)
    X_val, y_val = load_split("lstm_val", normal_label)
    X_test, y_test = load_split("lstm_test", normal_label)

    val_counts = np.bincount(y_val, minlength=2)
    if val_counts.tolist() != [13941, 23697]:
        raise ValueError(f"Unexpected Stage 1 validation counts: {val_counts.tolist()}")

    input_dim = int(X_train.shape[2])
    seq_len = int(X_train.shape[1])
    class_weights = class_weights_from_binary(y_train)
    hparams = {
        "input_size": input_dim,
        "seq_len": seq_len,
        "num_classes": 2,
        "d_model": 128,
        "n_heads": 4,
        "num_layers": 2,
        "dim_feedforward": 256,
        "classifier_hidden": 128,
        "dropout": 0.15,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "focal_gamma": 1.5,
        "class_weights": class_weights,
    }

    model = TemporalTransformerClassifier(**hparams).to(device)
    train_loader = make_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size=batch_size, shuffle=False)

    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", patience=10, verbose=True),
        ModelCheckpoint(
            dirpath=MODEL_ARTIFACTS_DIR,
            filename=CHECKPOINT_NAME,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_weights_only=True,
        ),
    ]
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=callbacks,
        enable_progress_bar=True,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_path = callbacks[1].best_model_path
    if best_path:
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True)["state_dict"])
    model.to(device).eval()

    val_prob, val_targets = predict_attack_probability(model, val_loader, device)
    threshold_result = tune_attack_threshold(val_targets, val_prob, min_normal_recall=0.90)
    threshold = threshold_result["threshold"]

    torch.save(model.state_dict(), MODEL_PATH)
    export_onnx(model, input_dim=input_dim, seq_len=seq_len, device=device)

    threshold_config = {
        "model_type": "TemporalTransformerClassifier",
        "task": "stage1_binary_attack_gate",
        "model_hparams": hparams,
        "normal_label": int(normal_label),
        "label_policy": "0 = Normal, 1 = Attack where original label != label_map['Normal']",
        "attack_probability_threshold": threshold,
        "threshold_selection": {
            "optimize_for": "attack recall + f1",
            "min_normal_recall": 0.90,
            "validation_metrics": threshold_result,
        },
    }
    with open(THRESHOLD_PATH, "w") as f:
        json.dump(threshold_config, f, indent=4)

    reports = {
        "train": evaluate_split(model, X_train, y_train, threshold, batch_size, device),
        "validation": evaluate_split(model, X_val, y_val, threshold, batch_size, device),
        "test": evaluate_split(model, X_test, y_test, threshold, batch_size, device),
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(reports, f, indent=4)

    print("Stage 1 training complete.")
    print(f"Validation attack recall: {reports['validation']['attack_recall']:.4f}")
    print(f"Validation normal recall: {reports['validation']['normal_recall']:.4f}")


if __name__ == "__main__":
    train_stage1_binary()
