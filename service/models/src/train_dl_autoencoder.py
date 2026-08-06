import json
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

torch.set_float32_matmul_precision("medium")

try:
    import mlflow

    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

from dl_pipeline import (
    AutoencoderDataModule,
    TemporalOneClassVAE,
    evaluate_autoencoder,
    find_optimal_threshold,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[3]
SEQUENCES_DIR = ROOT / "data" / "sequences"
ARTIFACTS_DIR = ROOT / "service" / "models" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
COMPONENT_NAMES = ["reconstruction_mse", "latent_center_distance", "kl_surprise"]


def _make_loader(X: np.ndarray, batch_size: int, shuffle: bool = False) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X.astype(np.float32, copy=False)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


@torch.no_grad()
def _collect_components(model: TemporalOneClassVAE, loader: DataLoader, device: torch.device) -> np.ndarray:
    chunks = []
    for (X_batch,) in loader:
        X_batch = X_batch.to(device)
        chunks.append(model.anomaly_components(X_batch).detach().cpu())
    return torch.cat(chunks, dim=0).numpy()


def _component_diagnostics(components: np.ndarray, y_binary: np.ndarray) -> dict:
    diagnostics = {}
    for label_name, mask in (("normal", y_binary == 0), ("attack", y_binary == 1)):
        split = components[mask]
        diagnostics[label_name] = {
            "median": dict(zip(COMPONENT_NAMES, np.median(split, axis=0).tolist())),
            "p90": dict(zip(COMPONENT_NAMES, np.percentile(split, 90, axis=0).tolist())),
        }
    return diagnostics


def train_autoencoder():
    set_seed(42)
    device_type = "gpu" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training TemporalOneClassVAE on device: {device_type}")

    batch_size = 256
    epochs = 200

    print("Loading sequence datasets...")
    X_train_seq = np.load(SEQUENCES_DIR / "lstm_train_w10_v1_20260612.npy")
    y_train_seq = np.load(SEQUENCES_DIR / "lstm_train_labels_w10_v1_20260612.npy")
    X_val_seq = np.load(SEQUENCES_DIR / "lstm_val_w10_v1_20260612.npy")
    y_val_seq = np.load(SEQUENCES_DIR / "lstm_val_labels_w10_v1_20260612.npy")

    train_normal_mask = y_train_seq == 0
    val_binary = (y_val_seq > 0).astype(np.int64)
    val_normal_mask = val_binary == 0

    X_train = X_train_seq[train_normal_mask].astype(np.float32)
    y_train = np.zeros(len(X_train), dtype=np.int64)
    X_val_normal = X_val_seq[val_normal_mask].astype(np.float32)
    y_val_normal = np.zeros(len(X_val_normal), dtype=np.int64)
    X_val_mixed = X_val_seq.astype(np.float32)

    input_dim = int(X_train.shape[2])
    seq_len = int(X_train.shape[1])
    print(f"Normal-only train sequences: {len(X_train):,}")
    print(f"Normal validation sequences: {len(X_val_normal):,}")
    print(f"Mixed threshold validation sequences: {len(X_val_mixed):,}")

    datamodule = AutoencoderDataModule(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val_normal,
        y_val=y_val_normal,
        batch_size=batch_size,
    )
    datamodule.setup("fit")

    model_hparams = {
        "input_size": input_dim,
        "seq_len": seq_len,
        "d_model": 128,
        "latent_dim": 16,
        "n_heads": 4,
        "num_layers": 2,
        "dim_feedforward": 256,
        "decoder_hidden": 256,
        "decoder_layers": 2,
        "dropout": 0.1,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "beta_max": 0.02,
        "kl_warmup_epochs": 20,
        "center_loss_weight": 0.2,
        "score_weights": (0.10, 0.60, 0.30),
    }
    model = TemporalOneClassVAE(**model_hparams).to(device)

    print("Initializing one-class latent center from normal training traffic...")
    model.initialize_center(datamodule.train_dataloader(), device)

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.0,
        patience=15,
        verbose=True,
        mode="min",
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=ARTIFACTS_DIR,
        filename="dl_ae_v1.0",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
        save_weights_only=True,
    )

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator=device_type,
        devices=1,
        callbacks=[early_stop_callback, checkpoint_callback],
        enable_progress_bar=True,
    )

    if HAS_MLFLOW:
        mlflow.set_tracking_uri(f"file://{ROOT}/experiments/mlruns")
        mlflow.set_experiment("KodeMapper_DL_IDS_Expansion")
        mlflow.start_run(run_name="TemporalOneClassVAE_v1.0")
        mlflow.log_params(
            {
                "model": "TemporalOneClassVAE",
                "batch_size": batch_size,
                "threshold_metric": "Balanced F1 with normal recall constraint",
                "binary_label_policy": "attack = y > 0",
                **model_hparams,
            }
        )

    print("Starting normal-only one-class training...")
    trainer.fit(model, datamodule=datamodule)

    checkpoint = torch.load(
        checkpoint_callback.best_model_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    print("Calibrating anomaly score on normal training traffic...")
    calibration = model.calibrate_score(datamodule.train_dataloader(), device)

    print("Evaluating mixed validation scores and selecting threshold...")
    val_eval_loader = _make_loader(X_val_mixed, batch_size=batch_size, shuffle=False)
    val_scores = evaluate_autoencoder(model, val_eval_loader, device)
    val_components = _collect_components(model, val_eval_loader, device)
    normal_scores = val_scores[val_binary == 0]
    attack_scores = val_scores[val_binary == 1]

    result = find_optimal_threshold(
        normal_scores,
        attack_scores,
        optimize_for="balanced",
        min_normal_recall=0.90,
    )
    threshold = result["threshold"]
    print(f"Calculated Optimal Anomaly Threshold (E_thresh): {threshold:.6f}")
    if "attack_recall" in result:
        print(f"  Validation Attack Recall : {result['attack_recall']:.3f}")
        print(f"  Validation Normal Recall : {result['normal_recall']:.3f}")
        print(f"  Validation F1            : {result['f1']:.3f}")

    if HAS_MLFLOW:
        mlflow.log_metric("anomaly_threshold", threshold)
        for metric_name in ("attack_recall", "normal_recall", "f1", "precision", "score"):
            if metric_name in result:
                mlflow.log_metric(f"val_{metric_name}", result[metric_name])

    threshold_config = {
        "E_thresh": threshold,
        "method": "balanced_f1_optimizer",
        "model_type": "TemporalOneClassVAE",
        "model_hparams": {**model_hparams, "score_weights": list(model_hparams["score_weights"])},
        "score_method": "calibrated_recon_center_kl",
        "score_components": ["reconstruction_mse", "latent_center_distance", "kl_surprise"],
        "score_weights": list(model_hparams["score_weights"]),
        "score_calibration": calibration,
        "component_diagnostics": _component_diagnostics(val_components, val_binary),
        "binary_label_policy": "0 = normal, y > 0 = attack",
        "threshold_selection": {
            "optimize_for": "balanced",
            "min_normal_recall": 0.90,
            "validation_metrics": result,
        },
    }
    with open(ARTIFACTS_DIR / "dl_ae_threshold_v1.0.json", "w") as f:
        json.dump(threshold_config, f, indent=4)

    torch.save(model.state_dict(), ARTIFACTS_DIR / "dl_ae_v1.0.pt")

    try:
        import onnx  # noqa: F401

        dummy_input = torch.randn(1, seq_len, input_dim, device=device)
        torch.onnx.export(
            model,
            dummy_input,
            ARTIFACTS_DIR / "dl_ae_v1.0.onnx",
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size", 1: "seq_len"},
                "output": {0: "batch_size", 1: "seq_len"},
            },
        )
        print(f"Model successfully exported to ONNX: {ARTIFACTS_DIR / 'dl_ae_v1.0.onnx'}")
    except Exception as e:
        print(
            "Warning: ONNX export failed. PyTorch model was successfully saved "
            f"as .pt, but ONNX export failed: {e}"
        )

    if HAS_MLFLOW:
        mlflow.end_run()

    print("Autoencoder training, calibration, and serialization complete!")


if __name__ == "__main__":
    train_autoencoder()
