import json
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

torch.set_float32_matmul_precision("medium")

from dl_pipeline import TemporalTransformerClassifier, get_device, set_seed

ROOT = Path(__file__).resolve().parents[3]
SEQUENCES_DIR = ROOT / "data" / "sequences"
DATA_ARTIFACTS_DIR = ROOT / "data" / "artifacts"
MODEL_ARTIFACTS_DIR = ROOT / "service" / "models" / "artifacts"
MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

ATTACK_CLASSES = ["DoS", "Exploits", "Fuzzers", "Generic", "Reconnaissance"]

MODEL_PATH = MODEL_ARTIFACTS_DIR / "dl_stage2_multiclass_v1.0.pt"
ONNX_PATH = MODEL_ARTIFACTS_DIR / "dl_stage2_multiclass_v1.0.onnx"
CHECKPOINT_NAME = "dl_stage2_multiclass_v1.0"
CONFIG_PATH = MODEL_ARTIFACTS_DIR / "dl_stage2_multiclass_config_v1.0.json"
REPORT_PATH = MODEL_ARTIFACTS_DIR / "dl_stage2_multiclass_report_v1.0.json"


def load_preprocessing_config() -> dict:
    with open(DATA_ARTIFACTS_DIR / "preprocessing_config_unsw_v1_20260612.json", "r") as f:
        config = json.load(f)
    normal_label = config["label_map"]["Normal"]
    if normal_label != 4:
        raise ValueError(f"Expected label_map['Normal'] == 4, got {normal_label}.")
    return config


def build_stage2_label_maps(config: dict) -> tuple[int, dict[int, int], dict[str, int]]:
    label_map = config["label_map"]
    normal_label = label_map["Normal"]
    missing = [name for name in ATTACK_CLASSES if name not in label_map]
    if missing:
        raise ValueError(f"Missing attack classes in preprocessing label_map: {missing}")

    stage2_label_map = {name: idx for idx, name in enumerate(ATTACK_CLASSES)}
    original_to_stage2 = {int(label_map[name]): stage2_label_map[name] for name in ATTACK_CLASSES}
    if normal_label in original_to_stage2:
        raise ValueError("Normal label leaked into Stage 2 attack label map.")
    return int(normal_label), original_to_stage2, stage2_label_map


def load_attack_split(split_prefix: str, normal_label: int, original_to_stage2: dict[int, int]):
    X = np.load(SEQUENCES_DIR / f"{split_prefix}_w10_v1_20260612.npy").astype(np.float32)
    y_original = np.load(SEQUENCES_DIR / f"{split_prefix}_labels_w10_v1_20260612.npy")
    attack_mask = y_original != normal_label
    X_attack = X[attack_mask]
    y_attack_original = y_original[attack_mask]
    y_attack = np.array([original_to_stage2[int(label)] for label in y_attack_original], dtype=np.int64)
    return X_attack, y_attack


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.int64, copy=False)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def class_weights_from_labels(y: np.ndarray, num_classes: int) -> list[float]:
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"Every Stage 2 class must have samples. Counts: {counts.tolist()}")
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()
    return weights.astype(np.float32).tolist()


@torch.no_grad()
def predict_proba(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    targets = []
    for X_batch, y_batch in loader:
        logits = model(X_batch.to(device))
        probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
        targets.append(y_batch.numpy())
    return np.concatenate(probs), np.concatenate(targets)


def thresholded_predictions(probs: np.ndarray, confidence_threshold: float) -> np.ndarray:
    preds = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)
    unclassified_label = len(ATTACK_CLASSES)
    preds = preds.copy()
    preds[confidence < confidence_threshold] = unclassified_label
    return preds


def tune_confidence_threshold(y_true: np.ndarray, probs: np.ndarray) -> dict:
    argmax_pred = np.argmax(probs, axis=1)
    base_macro_recall = recall_score(y_true, argmax_pred, labels=list(range(len(ATTACK_CLASSES))), average="macro", zero_division=0)
    best = {
        "confidence_threshold": 0.0,
        "coverage": 1.0,
        "macro_recall": float(base_macro_recall),
        "unclassified_rate": 0.0,
    }

    for threshold in np.linspace(0.05, 0.95, 19):
        pred = thresholded_predictions(probs, threshold)
        coverage = float(np.mean(pred != len(ATTACK_CLASSES)))
        macro_recall = recall_score(
            y_true,
            pred,
            labels=list(range(len(ATTACK_CLASSES))),
            average="macro",
            zero_division=0,
        )
        if coverage >= 0.90 and macro_recall >= base_macro_recall - 0.02:
            best = {
                "confidence_threshold": float(threshold),
                "coverage": coverage,
                "macro_recall": float(macro_recall),
                "unclassified_rate": float(1.0 - coverage),
            }
    return best


def evaluate_probs(y_true: np.ndarray, probs: np.ndarray, confidence_threshold: float) -> dict:
    argmax_pred = np.argmax(probs, axis=1)
    argmax_report = classification_report(
        y_true,
        argmax_pred,
        labels=list(range(len(ATTACK_CLASSES))),
        target_names=ATTACK_CLASSES,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    argmax_matrix = confusion_matrix(y_true, argmax_pred, labels=list(range(len(ATTACK_CLASSES)))).tolist()

    thresholded_pred = thresholded_predictions(probs, confidence_threshold)
    thresholded_names = ATTACK_CLASSES + ["Attack-Unclassified"]
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

    per_class_recall = {name: float(argmax_report[name]["recall"]) for name in ATTACK_CLASSES}
    return {
        "argmax": {
            "confusion_matrix": argmax_matrix,
            "classification_report": argmax_report,
            "macro_recall": float(recall_score(y_true, argmax_pred, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(y_true, argmax_pred, average="macro", zero_division=0)),
            "per_class_recall": per_class_recall,
        },
        "thresholded": {
            "confidence_threshold": confidence_threshold,
            "confusion_matrix": thresholded_matrix,
            "classification_report": thresholded_report,
            "coverage": float(np.mean(thresholded_pred != len(ATTACK_CLASSES))),
            "unclassified_rate": float(np.mean(thresholded_pred == len(ATTACK_CLASSES))),
        },
    }


def evaluate_split(model, X, y, batch_size: int, confidence_threshold: float, device) -> dict:
    loader = make_loader(X, y, batch_size=batch_size, shuffle=False)
    probs, y_true = predict_proba(model, loader, device)
    return evaluate_probs(y_true, probs, confidence_threshold)


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


def train_stage2_multiclass():
    set_seed(42)
    config = load_preprocessing_config()
    normal_label, original_to_stage2, stage2_label_map = build_stage2_label_maps(config)
    device = get_device()

    batch_size = 256
    epochs = 100
    print(f"Training Stage 2 multiclass attack classifier on device: {device}")

    X_train, y_train = load_attack_split("lstm_train", normal_label, original_to_stage2)
    X_val, y_val = load_attack_split("lstm_val", normal_label, original_to_stage2)
    X_test, y_test = load_attack_split("lstm_test", normal_label, original_to_stage2)

    val_counts = np.bincount(y_val, minlength=len(ATTACK_CLASSES))
    if np.any(val_counts == 0):
        raise ValueError(f"Stage 2 validation set is missing attack classes: {val_counts.tolist()}")

    input_dim = int(X_train.shape[2])
    seq_len = int(X_train.shape[1])
    class_weights = class_weights_from_labels(y_train, len(ATTACK_CLASSES))
    hparams = {
        "input_size": input_dim,
        "seq_len": seq_len,
        "num_classes": len(ATTACK_CLASSES),
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
        EarlyStopping(monitor="val_loss", mode="min", patience=12, verbose=True),
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

    val_probs, val_targets = predict_proba(model, val_loader, device)
    confidence_policy = tune_confidence_threshold(val_targets, val_probs)
    confidence_threshold = confidence_policy["confidence_threshold"]

    torch.save(model.state_dict(), MODEL_PATH)
    export_onnx(model, input_dim=input_dim, seq_len=seq_len, device=device)

    reports = {
        "train": evaluate_split(model, X_train, y_train, batch_size, confidence_threshold, device),
        "validation": evaluate_split(model, X_val, y_val, batch_size, confidence_threshold, device),
        "test": evaluate_split(model, X_test, y_test, batch_size, confidence_threshold, device),
    }

    generic_recall = reports["validation"]["argmax"]["per_class_recall"]["Generic"]
    accepted = generic_recall > 0.05
    config_out = {
        "model_type": "TemporalTransformerClassifier",
        "task": "stage2_multiclass_attack_classifier",
        "model_hparams": hparams,
        "normal_label": int(normal_label),
        "attack_classes": ATTACK_CLASSES,
        "stage2_label_map": stage2_label_map,
        "original_to_stage2_label": {str(k): int(v) for k, v in original_to_stage2.items()},
        "confidence_policy": confidence_policy,
        "acceptance": {
            "accepted": accepted,
            "reason": "Generic recall must be above 0.05 and Normal must not be present in Stage 2 labels.",
            "validation_macro_recall": reports["validation"]["argmax"]["macro_recall"],
            "validation_macro_f1": reports["validation"]["argmax"]["macro_f1"],
            "validation_generic_recall": generic_recall,
        },
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config_out, f, indent=4)
    with open(REPORT_PATH, "w") as f:
        json.dump(reports, f, indent=4)

    print("Stage 2 training complete.")
    print(f"Validation macro recall: {reports['validation']['argmax']['macro_recall']:.4f}")
    print(f"Validation macro F1: {reports['validation']['argmax']['macro_f1']:.4f}")
    print(f"Validation Generic recall: {generic_recall:.4f}")
    if not accepted:
        raise RuntimeError("Stage 2 model rejected: Generic recall repeated the near-zero failure mode.")


if __name__ == "__main__":
    train_stage2_multiclass()
