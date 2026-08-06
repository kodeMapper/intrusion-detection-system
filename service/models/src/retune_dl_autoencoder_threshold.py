import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from dl_pipeline import TemporalOneClassVAE, find_optimal_threshold, get_device

ROOT = Path(__file__).resolve().parents[3]
SEQUENCES_DIR = ROOT / "data" / "sequences"
ARTIFACTS_DIR = ROOT / "service" / "models" / "artifacts"
THRESHOLD_PATH = ARTIFACTS_DIR / "dl_ae_threshold_v1.0.json"
MODEL_PATH = ARTIFACTS_DIR / "dl_ae_v1.0.pt"

COMPONENT_NAMES = ["reconstruction_mse", "latent_center_distance", "kl_surprise"]
DEFAULT_CANDIDATES = {
    "latent_heavy_10_60_30": (0.10, 0.60, 0.30),
    "no_recon_00_70_30": (0.00, 0.70, 0.30),
    "balanced_latent_15_50_35": (0.15, 0.50, 0.35),
    "current_45_35_20": (0.45, 0.35, 0.20),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retune TemporalOneClassVAE score weights and threshold without retraining."
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--min-normal-recall", type=float, default=0.90)
    parser.add_argument("--target-attack-recall", type=float, default=0.60)
    parser.add_argument("--apply", action="store_true", help="Persist the best weights if the target is met.")
    parser.add_argument(
        "--force-apply",
        action="store_true",
        help="Persist the best weights even if target attack recall is not met.",
    )
    return parser.parse_args()


def select_indices(y_binary: np.ndarray, max_samples: int | None) -> np.ndarray:
    if max_samples is None or max_samples >= len(y_binary):
        return np.arange(len(y_binary))

    normal_idx = np.flatnonzero(y_binary == 0)
    attack_idx = np.flatnonzero(y_binary == 1)
    normal_take = min(len(normal_idx), max(1, max_samples // 2))
    attack_take = min(len(attack_idx), max_samples - normal_take)
    if attack_take <= 0:
        attack_take = min(len(attack_idx), 1)
        normal_take = min(len(normal_idx), max_samples - attack_take)

    chosen = np.concatenate([normal_idx[:normal_take], attack_idx[:attack_take]])
    return np.sort(chosen)


def load_model(config: dict, device: torch.device) -> TemporalOneClassVAE:
    hparams = dict(config["model_hparams"])
    model = TemporalOneClassVAE(**hparams)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    if "score_weights" in config:
        model.set_score_weights(config["score_weights"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def compute_components(model: TemporalOneClassVAE, X_val: np.ndarray, batch_size: int, device: torch.device):
    dataset = TensorDataset(torch.from_numpy(X_val.astype(np.float32, copy=False)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    chunks = []
    for (X_batch,) in loader:
        X_batch = X_batch.to(device)
        chunks.append(model.anomaly_components(X_batch).detach().cpu())
    return torch.cat(chunks, dim=0).numpy()


def scaled_components(components: np.ndarray, config: dict) -> np.ndarray:
    calibration = config.get("score_calibration", {})
    median = np.asarray(calibration.get("median"), dtype=np.float32)
    iqr = np.asarray(calibration.get("iqr"), dtype=np.float32)
    if median.shape != (3,) or iqr.shape != (3,):
        raise ValueError("Threshold config must contain 3-value score_calibration median and iqr.")
    return np.maximum((components - median) / (iqr + 1e-6), 0.0)


def evaluate_candidates(scaled: np.ndarray, y_binary: np.ndarray, min_normal_recall: float):
    results = {}
    for name, weights in DEFAULT_CANDIDATES.items():
        weights_arr = np.asarray(weights, dtype=np.float32)
        scores = scaled @ (weights_arr / weights_arr.sum())
        result = find_optimal_threshold(
            scores[y_binary == 0],
            scores[y_binary == 1],
            optimize_for="balanced",
            min_normal_recall=min_normal_recall,
        )
        results[name] = {
            "weights": list(weights),
            "threshold": result["threshold"],
            "metrics": result,
        }
    return results


def component_diagnostics(components: np.ndarray, y_binary: np.ndarray) -> dict:
    diagnostics = {}
    for label_name, mask in (("normal", y_binary == 0), ("attack", y_binary == 1)):
        split = components[mask]
        diagnostics[label_name] = {
            "median": dict(zip(COMPONENT_NAMES, np.median(split, axis=0).tolist())),
            "p90": dict(zip(COMPONENT_NAMES, np.percentile(split, 90, axis=0).tolist())),
        }
    return diagnostics


def choose_best(results: dict) -> tuple[str, dict]:
    return max(results.items(), key=lambda item: item[1]["metrics"].get("score", -np.inf))


def apply_best(
    config: dict,
    best_name: str,
    best_result: dict,
    all_results: dict,
    diagnostics: dict,
    min_normal_recall: float,
):
    updated = copy.deepcopy(config)
    best_weights = best_result["weights"]
    updated["E_thresh"] = best_result["threshold"]
    updated["score_weights"] = best_weights
    updated.setdefault("model_hparams", {})["score_weights"] = best_weights
    updated["score_method"] = "calibrated_recon_center_kl"
    updated["score_components"] = COMPONENT_NAMES
    updated["retune_results"] = {
        "selected": best_name,
        "candidates": all_results,
        "component_diagnostics": diagnostics,
    }
    updated["threshold_selection"] = {
        "optimize_for": "balanced",
        "min_normal_recall": min_normal_recall,
        "validation_metrics": best_result["metrics"],
    }

    with open(THRESHOLD_PATH, "w") as f:
        json.dump(updated, f, indent=4)

    device = torch.device("cpu")
    model = TemporalOneClassVAE(**updated["model_hparams"])
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True), strict=False)
    model.set_score_weights(best_weights)
    torch.save(model.state_dict(), MODEL_PATH)


def main():
    args = parse_args()
    device = get_device()
    print(f"Retuning TemporalOneClassVAE threshold on device: {device}")

    with open(THRESHOLD_PATH, "r") as f:
        config = json.load(f)
    if config.get("model_type") != "TemporalOneClassVAE":
        raise ValueError("Retuning utility only supports TemporalOneClassVAE artifacts.")

    X_val = np.load(SEQUENCES_DIR / "lstm_val_w10_v1_20260612.npy")
    y_val = np.load(SEQUENCES_DIR / "lstm_val_labels_w10_v1_20260612.npy")
    y_binary = (y_val > 0).astype(np.int64)
    idx = select_indices(y_binary, args.max_samples)
    X_val = X_val[idx].astype(np.float32)
    y_binary = y_binary[idx]
    print(f"Using {len(X_val):,} validation sequences ({int((y_binary == 0).sum()):,} normal, {int((y_binary == 1).sum()):,} attack).")

    model = load_model(config, device)
    components = compute_components(model, X_val, args.batch_size, device)
    scaled = scaled_components(components, config)
    results = evaluate_candidates(scaled, y_binary, args.min_normal_recall)
    diagnostics = component_diagnostics(components, y_binary)
    best_name, best_result = choose_best(results)

    print("\nCandidate results:")
    for name, result in results.items():
        metrics = result["metrics"]
        print(
            f"{name}: threshold={metrics['threshold']:.6f}, "
            f"attack_recall={metrics.get('attack_recall', -1):.4f}, "
            f"normal_recall={metrics.get('normal_recall', -1):.4f}, "
            f"f1={metrics.get('f1', -1):.4f}, "
            f"precision={metrics.get('precision', -1):.4f}, "
            f"score={metrics.get('score', -1):.4f}"
        )

    print(f"\nBest candidate: {best_name} with weights={best_result['weights']}")
    best_attack_recall = best_result["metrics"].get("attack_recall", 0.0)
    target_met = best_attack_recall >= args.target_attack_recall
    if args.apply:
        if target_met or args.force_apply:
            apply_best(config, best_name, best_result, results, diagnostics, args.min_normal_recall)
            print(f"Applied {best_name} to {THRESHOLD_PATH} and updated model score_weights buffer.")
        else:
            print(
                "Not applying changes because target attack recall was not met. "
                "Use --force-apply to persist the best candidate anyway."
            )


if __name__ == "__main__":
    main()
