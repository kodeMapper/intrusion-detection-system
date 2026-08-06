import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from pathlib import Path

def evaluate_lstm(model, dataloader, device):
    """
    Evaluates LSTM predictions and computes Macro Recall.
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            logits = model(X_batch)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
            
    macro_recall = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    return macro_recall, all_targets, all_preds

def evaluate_autoencoder(model, dataloader, device):
    """
    Computes anomaly scores for all samples.

    New one-class models expose anomaly_score(). Older autoencoders fall back
    to MSE reconstruction error for backward compatibility.
    """
    model.eval()
    criterion = nn.MSELoss(reduction='none')
    errors = []
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                X_batch = batch[0]
            else:
                X_batch = batch
            X_batch = X_batch.to(device)
            if hasattr(model, "anomaly_score"):
                sample_errors = model.anomaly_score(X_batch).detach().cpu().numpy()
                errors.extend(sample_errors)
                continue

            out = model(X_batch)
            reconstructed = out[0] if isinstance(out, tuple) else out
            
            # Compute MSE per sample
            loss = criterion(reconstructed, X_batch)
            if X_batch.dim() == 3:
                # Average over sequence length and features
                sample_errors = loss.mean(dim=(1, 2)).cpu().numpy()
            else:
                # Average over features
                sample_errors = loss.mean(dim=1).cpu().numpy()
            errors.extend(sample_errors)
            
    return np.array(errors)


from sklearn.metrics import f1_score, precision_score

def find_optimal_threshold(
    normal_errors: np.ndarray,
    attack_errors: np.ndarray,
    n_thresholds: int = 500,
    min_normal_recall: float = 0.90,
    optimize_for: str = "balanced",
) -> dict:
    """
    Sweep thresholds between the 1st and 99.9th percentile of all errors.
    Returns the threshold that maximizes the objective while keeping
    Normal Recall >= min_normal_recall.
    """
    all_errors = np.concatenate([normal_errors, attack_errors])
    labels = np.concatenate([
        np.zeros(len(normal_errors)),   # 0 = normal
        np.ones(len(attack_errors)),    # 1 = attack
    ])

    lo = np.percentile(all_errors, 1.0)
    hi = np.percentile(all_errors, 99.9)
    candidates = np.linspace(lo, hi, n_thresholds)

    best = {"threshold": None, "score": -np.inf}

    for thresh in candidates:
        preds = (all_errors > thresh).astype(int)

        # Normal Recall = fraction of normals correctly classified as normal
        normal_mask = labels == 0
        normal_recall = (preds[normal_mask] == 0).mean()
        if normal_recall < min_normal_recall:
            continue  # Hard constraint — skip thresholds that alarm too many normals

        attack_mask = labels == 1
        attack_recall = preds[attack_mask].mean()
        f1 = f1_score(labels, preds, zero_division=0)
        precision = precision_score(labels, preds, zero_division=0)

        if optimize_for == "f1":
            score = f1
        elif optimize_for == "attack_recall":
            score = attack_recall
        elif optimize_for == "balanced":
            score = 0.4 * f1 + 0.6 * attack_recall
        else:
            raise ValueError(f"Unknown optimize_for: {optimize_for}")

        if score > best["score"]:
            best = {
                "threshold": float(thresh),
                "score": float(score),
                "f1": float(f1),
                "normal_recall": float(normal_recall),
                "attack_recall": float(attack_recall),
                "precision": float(precision),
            }

    if best["threshold"] is None:
        # Fallback: no threshold met the normal_recall constraint
        fallback = float(np.percentile(normal_errors, 99))
        best = {"threshold": fallback, "warning": "constraint_relaxed"}

    return best
