import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from dl_pipeline import LSTMAutoencoder, TemporalOneClassVAE, evaluate_autoencoder, get_device

ROOT = Path(__file__).resolve().parents[3]
SEQUENCES_DIR = ROOT / "data" / "sequences"
MODELS_ARTIFACTS_DIR = ROOT / "service" / "models" / "artifacts"


def get_metrics(y_true, y_pred, split_name):
    print(f"--- {split_name} Split ---")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Normal (0)", "Anomaly/Attack (1)"],
            digits=4,
            zero_division=0,
        )
    )
    print("-" * 40)


def build_model(threshold_config, input_dim: int, seq_len: int):
    model_type = threshold_config.get("model_type", "LSTMAutoencoder")
    if model_type == "TemporalOneClassVAE":
        hparams = dict(threshold_config.get("model_hparams", {}))
        hparams.setdefault("input_size", input_dim)
        hparams.setdefault("seq_len", seq_len)
        return TemporalOneClassVAE(**hparams)

    return LSTMAutoencoder(
        input_size=input_dim,
        seq_len=seq_len,
        hidden_size=64,
        num_layers=2,
        latent_dim=32,
        dropout=0.2,
    )


def evaluate_split(split_prefix, split_name, model, threshold, device):
    X_path = SEQUENCES_DIR / f"{split_prefix}_w10_v1_20260612.npy"
    y_path = SEQUENCES_DIR / f"{split_prefix}_labels_w10_v1_20260612.npy"

    print(f"Loading {split_name} data...")
    X_split = np.load(X_path).astype(np.float32)
    y_true_multi = np.load(y_path)
    y_true_binary = (y_true_multi > 0).astype(int)

    dataset = TensorDataset(torch.from_numpy(X_split))
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)

    scores = evaluate_autoencoder(model, loader, device)
    y_pred = (scores > threshold).astype(int)
    get_metrics(y_true_binary, y_pred, split_name)


def main():
    device = get_device()
    print(f"Using device: {device}")

    with open(MODELS_ARTIFACTS_DIR / "dl_ae_threshold_v1.0.json", "r") as f:
        threshold_config = json.load(f)
    threshold = threshold_config["E_thresh"]
    model_type = threshold_config.get("model_type", "LSTMAutoencoder")
    print(f"Loaded {model_type} anomaly threshold (E_thresh): {threshold:.6f}")

    X_sample = np.load(SEQUENCES_DIR / "lstm_val_w10_v1_20260612.npy", mmap_mode="r")
    seq_len = int(X_sample.shape[1])
    input_dim = int(X_sample.shape[2])

    print(f"Loading {model_type} (seq_len={seq_len}, input_dim={input_dim})...")
    model = build_model(threshold_config, input_dim=input_dim, seq_len=seq_len)
    model.load_state_dict(
        torch.load(MODELS_ARTIFACTS_DIR / "dl_ae_v1.0.pt", map_location=device, weights_only=True)
    )
    if hasattr(model, "set_score_weights") and "score_weights" in threshold_config:
        model.set_score_weights(threshold_config["score_weights"])
    model.to(device)
    model.eval()

    print("\nStarting Evaluation...\n")
    evaluate_split("lstm_train", "Train", model, threshold, device)
    evaluate_split("lstm_val", "Validation", model, threshold, device)
    evaluate_split("lstm_test", "Test", model, threshold, device)


if __name__ == "__main__":
    main()
