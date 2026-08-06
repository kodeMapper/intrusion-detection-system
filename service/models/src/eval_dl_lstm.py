import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

from dl_pipeline import FlowLSTM, FlowSequenceDataset, get_device, evaluate_lstm

ROOT = Path(__file__).resolve().parents[3]
SEQUENCES_DIR = ROOT / "data" / "sequences"
ARTIFACTS_DIR = ROOT / "data" / "artifacts"
MODELS_ARTIFACTS_DIR = ROOT / "service" / "models" / "artifacts"

def get_metrics(y_true, y_pred, split_name, target_names):
    print(f"--- {split_name} Split ---")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    # Some classes might not appear in a specific split, so use labels array
    labels = list(range(len(target_names)))
    print(classification_report(y_true, y_pred, labels=labels, target_names=target_names, digits=4, zero_division=0))
    print("-" * 50)

def main():
    device = get_device()
    print(f"Using device: {device}")

    # 1. Load config for label map
    with open(ARTIFACTS_DIR / "preprocessing_config_unsw_v1_20260612.json", "r") as f:
        config = json.load(f)
        
    label_map = config["label_map"]
    # Invert label map to get class names in order
    target_names = ["" for _ in range(len(label_map))]
    for name, idx in label_map.items():
        target_names[idx] = name
        
    print(f"Classes: {target_names}")

    # 2. Define data loading helper
    def load_and_evaluate(split_prefix, split_name, model):
        X_path = SEQUENCES_DIR / f"{split_prefix}_w10_v1_20260612.npy"
        y_path = SEQUENCES_DIR / f"{split_prefix}_labels_w10_v1_20260612.npy"
        
        print(f"Loading {split_name} data...")
        X = np.load(X_path)
        y = np.load(y_path)
        
        dataset = FlowSequenceDataset(X, y)
        loader = DataLoader(dataset, batch_size=512, shuffle=False)
        
        _, all_targets, all_preds = evaluate_lstm(model, loader, device)
        get_metrics(all_targets, all_preds, split_name, target_names)

    # 3. Load Model
    # Need input_dim and num_classes
    # Load just one array to get input_dim
    X_sample = np.load(SEQUENCES_DIR / "lstm_val_w10_v1_20260612.npy", mmap_mode='r')
    input_dim = X_sample.shape[2]
    num_classes = len(target_names)
    
    print(f"Loading LSTM model (input_dim={input_dim}, num_classes={num_classes})...")
    model = FlowLSTM(input_dim=input_dim, hidden_dim=128, num_layers=2, num_classes=num_classes)
    model.load_state_dict(torch.load(MODELS_ARTIFACTS_DIR / "dl_lstm_v1.0.pt", map_location=device, weights_only=True))
    model.to(device)
    
    print("\nStarting Evaluation...\n")
    load_and_evaluate("lstm_train", "Train", model)
    load_and_evaluate("lstm_val", "Validation", model)
    load_and_evaluate("lstm_test", "Test", model)

if __name__ == "__main__":
    main()
