import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        if self.reduction == 'sum': return focal_loss.sum()
        return focal_loss

# Optionally use MLflow if installed
try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

from dl_pipeline import (
    FlowLSTM, FlowSequenceDataset, 
    set_seed, get_device, evaluate_lstm
)

ROOT = Path(__file__).resolve().parents[3]
SEQUENCES_DIR = ROOT / "data" / "sequences"
ARTIFACTS_DIR = ROOT / "service" / "models" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def train_lstm():
    set_seed(42)
    device = get_device()
    print(f"Training LSTM on device: {device}")

    # Hyperparameters
    batch_size = 512
    epochs = 100
    lr = 1e-3
    patience = 15
    
    print("Loading datasets...")
    X_train = np.load(SEQUENCES_DIR / "lstm_train_w10_v1_20260612.npy")
    y_train = np.load(SEQUENCES_DIR / "lstm_train_labels_w10_v1_20260612.npy")
    X_val = np.load(SEQUENCES_DIR / "lstm_val_w10_v1_20260612.npy")
    y_val = np.load(SEQUENCES_DIR / "lstm_val_labels_w10_v1_20260612.npy")
    
    input_dim = X_train.shape[2]
    num_classes = len(np.unique(y_train))
    print(f"Input dim: {input_dim}, Classes: {num_classes}, Train Samples: {len(X_train)}")

    train_dataset = FlowSequenceDataset(X_train, y_train)
    val_dataset = FlowSequenceDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = FlowLSTM(input_dim=input_dim, hidden_dim=128, num_layers=2, num_classes=num_classes)
    model = model.to(device)

    # Calculate class weights for Focal Loss
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / (class_counts + 1e-6)
    # Normalize weights to avoid scaling learning rate drastically
    class_weights = class_weights / class_weights.sum() * num_classes
    alpha_tensor = torch.FloatTensor(class_weights).to(device)
    
    criterion = FocalLoss(alpha=alpha_tensor, gamma=2.0)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if HAS_MLFLOW:
        mlflow.set_tracking_uri(f"file://{ROOT}/experiments/mlruns")
        mlflow.set_experiment("KodeMapper_DL_IDS_Expansion")
        active_run = mlflow.start_run(run_name="FlowLSTM_v1.0")
        mlflow.log_params({
            "model": "FlowLSTM",
            "batch_size": batch_size,
            "learning_rate": lr,
            "seq_len": 10,
            "hidden_dim": 128
        })
    else:
        active_run = None

    best_val_recall = 0.0
    epochs_no_improve = 0

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        val_recall, _, _ = evaluate_lstm(model, val_loader, device)
        scheduler.step()
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss:.4f} | Val Macro Recall: {val_recall:.4f}")
        
        if HAS_MLFLOW:
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_macro_recall", val_recall, step=epoch)

        if val_recall > best_val_recall:
            best_val_recall = val_recall
            epochs_no_improve = 0
            # Save PyTorch state dictionary
            torch.save(model.state_dict(), ARTIFACTS_DIR / "dl_lstm_v1.0.pt")
            
            # Export to ONNX (optional)
            try:
                import onnx
                dummy_input = torch.randn(1, 10, input_dim, device=device)
                torch.onnx.export(
                    model, dummy_input, 
                    ARTIFACTS_DIR / "dl_lstm_v1.0.onnx",
                    export_params=True,
                    opset_version=14,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                )
                print(f"Model successfully exported to ONNX: {ARTIFACTS_DIR / 'dl_lstm_v1.0.onnx'}")
            except Exception as e:
                print(f"Warning: ONNX export failed. PyTorch model was successfully saved as .pt, but ONNX export failed: {e}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    if HAS_MLFLOW:
        mlflow.end_run()

    print("LSTM Training complete! Best Validation Recall:", best_val_recall)

if __name__ == "__main__":
    train_lstm()
