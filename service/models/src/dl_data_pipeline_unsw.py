import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed" / "unsw"
SEQUENCES_DIR = ROOT / "data" / "sequences"
AE_DIR = ROOT / "data" / "autoencoder"
ARTIFACTS_DIR = ROOT / "data" / "artifacts"

for d in [PROCESSED_DIR, SEQUENCES_DIR, AE_DIR, ARTIFACTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def create_sequences(X, y, window_size=10, stride=1):
    sequences = []
    labels = []
    for i in range(0, len(X) - window_size + 1, stride):
        seq = X[i : i + window_size]
        label = y[i + window_size - 1]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

def main():
    print("Loading raw UNSW-NB15 data...")
    train_path = RAW_DATA_DIR / "UNSW_NB15_training-set.csv"
    test_path = RAW_DATA_DIR / "UNSW_NB15_testing-set.csv"
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df = pd.concat([df_train, df_test], ignore_index=True)
    
    # 1. Feature Drop
    # Clean whitespace from attack_cat
    df['attack_cat'] = df['attack_cat'].str.strip()
    
    # Feature Engineering
    print("Engineering derived features...")
    if 'sport' in df.columns:
        # Some rows might have '-' or '0x', convert to numeric coercing errors
        sport_num = pd.to_numeric(df['sport'], errors='coerce').fillna(65535)
        df['is_wellknown_sport'] = (sport_num < 1024).astype(int)
    if 'dsport' in df.columns:
        dsport_num = pd.to_numeric(df['dsport'], errors='coerce').fillna(65535)
        df['is_wellknown_dsport'] = (dsport_num < 1024).astype(int)
        
    if all(c in df.columns for c in ['sbytes', 'dbytes']):
        df['bytes_ratio'] = df['sbytes'] / (df['dbytes'] + 1)
    if all(c in df.columns for c in ['spkts', 'dpkts']):
        df['pkts_ratio'] = df['spkts'] / (df['dpkts'] + 1)
    if all(c in df.columns for c in ['sbytes', 'dbytes', 'spkts', 'dpkts']):
        df['pkt_size_avg'] = (df['sbytes'] + df['dbytes']) / (df['spkts'] + df['dpkts'] + 1)
        
    print("Dropping non-live features...")
    drop_cols = ["srcip", "dstip", "id", "sport", "dsport", "Ltime"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # Drop weak attack classes
    weak_classes = ["Analysis", "Backdoor", "Shellcode", "Worms"]
    df = df[~df['attack_cat'].isin(weak_classes)]
    
    # Create target columns
    y = df['attack_cat']
    y_binary = df['label'] # For AE
    df = df.drop(columns=['attack_cat', 'label'])
    
    # Temporal / Stratified split
    print("Sorting by Stime and splitting...")
    if 'Stime' in df.columns:
        # Sort indices by Stime
        sorted_indices = df.sort_values('Stime').index
        df = df.loc[sorted_indices]
        y = y.loc[sorted_indices]
        y_binary = y_binary.loc[sorted_indices]
        df = df.drop(columns=['Stime'])
    
        # Chronological 70/15/15 split
        n = len(df)
        train_end = int(0.70 * n)
        val_end = int(0.85 * n)
        
        train_idx = df.index[:train_end]
        val_idx = df.index[train_end:val_end]
        test_idx = df.index[val_end:]
    else:
        print("Stime not found. Falling back to Stratified Random Split to prevent class starvation.")
        # Stratified 70/15/15 split
        train_idx, temp_idx = train_test_split(df.index, test_size=0.30, stratify=y, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=y.loc[temp_idx], random_state=42)
        
        # Sort indices within each split to maintain chronological order for sequences
        train_idx = train_idx.sort_values()
        val_idx = val_idx.sort_values()
        test_idx = test_idx.sort_values()

    # Save split indices
    split_indices = {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test": test_idx.tolist()
    }
    with open(ARTIFACTS_DIR / "split_indices_unsw_v1_20260612.json", "w") as f:
        json.dump(split_indices, f)
        
    X_train = df.loc[train_idx]
    y_train = y.loc[train_idx]
    y_bin_train = y_binary.loc[train_idx]
    
    X_val = df.loc[val_idx]
    y_val = y.loc[val_idx]
    
    X_test = df.loc[test_idx]
    y_test = y.loc[test_idx]
    
    # Identify cat/num columns
    cat_cols = ["proto", "service", "state"]
    cat_cols = [c for c in cat_cols if c in X_train.columns]
    num_cols = [c for c in X_train.columns if c not in cat_cols]
    
    # Encoding
    print("Encoding categorical features...")
    if cat_cols:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded_train = encoder.fit_transform(X_train[cat_cols])
        encoded_val = encoder.transform(X_val[cat_cols])
        encoded_test = encoder.transform(X_test[cat_cols])
        
        feature_names = encoder.get_feature_names_out(cat_cols)
        
        # Merge back
        X_train_enc = pd.DataFrame(encoded_train, columns=feature_names, index=X_train.index)
        X_val_enc = pd.DataFrame(encoded_val, columns=feature_names, index=X_val.index)
        X_test_enc = pd.DataFrame(encoded_test, columns=feature_names, index=X_test.index)
        
        X_train = pd.concat([X_train[num_cols], X_train_enc], axis=1)
        X_val = pd.concat([X_val[num_cols], X_val_enc], axis=1)
        X_test = pd.concat([X_test[num_cols], X_test_enc], axis=1)
        
        joblib.dump(encoder, ARTIFACTS_DIR / "onehot_encoder_unsw_v1_20260612.pkl")
    
    # Scaling
    print("Scaling numeric features...")
    scaler = QuantileTransformer(output_distribution='normal', n_quantiles=1000, random_state=42)
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_val[num_cols] = scaler.transform(X_val[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler_unsw_v1_20260612.pkl")
    
    # Save feature list
    with open(ARTIFACTS_DIR / "feature_list_unsw_v1_20260612.json", "w") as f:
        json.dump(X_train.columns.tolist(), f)
        
    # Autoencoder extraction
    print("Extracting Autoencoder data...")
    ae_train = X_train[y_bin_train == 0] # Normal only
    ae_train.to_parquet(AE_DIR / "ae_train_normal_unsw_v1_20260612.parquet")
    X_val.to_parquet(AE_DIR / "ae_val_mixed_unsw_v1_20260612.parquet")
    
    # Sequence building
    print("Building LSTM sequences...")
    # Convert labels to ints
    unique_labels = sorted(y.unique().tolist())
    label_map = {l: i for i, l in enumerate(unique_labels)}
    y_train_int = y_train.map(label_map).values
    y_val_int = y_val.map(label_map).values
    y_test_int = y_test.map(label_map).values
    
    X_train_seq, y_train_seq = create_sequences(X_train.values, y_train_int)
    X_val_seq, y_val_seq = create_sequences(X_val.values, y_val_int)
    X_test_seq, y_test_seq = create_sequences(X_test.values, y_test_int)
    
    np.save(SEQUENCES_DIR / "lstm_train_w10_v1_20260612.npy", X_train_seq)
    np.save(SEQUENCES_DIR / "lstm_train_labels_w10_v1_20260612.npy", y_train_seq)
    np.save(SEQUENCES_DIR / "lstm_val_w10_v1_20260612.npy", X_val_seq)
    np.save(SEQUENCES_DIR / "lstm_val_labels_w10_v1_20260612.npy", y_val_seq)
    np.save(SEQUENCES_DIR / "lstm_test_w10_v1_20260612.npy", X_test_seq)
    np.save(SEQUENCES_DIR / "lstm_test_labels_w10_v1_20260612.npy", y_test_seq)
    
    # Save config
    config = {
        "version": "v1",
        "date": "2026-06-12",
        "dataset": "unsw_nb15",
        "random_seed": 42,
        "split_ratio": {"train": 0.70, "val": 0.15, "test": 0.15},
        "dropped_columns": drop_cols + ["Stime"],
        "dropped_attack_classes": weak_classes,
        "categorical_features": cat_cols,
        "categorical_encoding": "one_hot",
        "scaler_type": "QuantileTransformer(normal)",
        "lstm_window_size": 10,
        "lstm_stride": 1,
        "lstm_label_strategy": "last_flow",
        "ae_training_data": "normal_only",
        "label_map": label_map
    }
    with open(ARTIFACTS_DIR / "preprocessing_config_unsw_v1_20260612.json", "w") as f:
        json.dump(config, f, indent=4)
        
    print("Pipeline Execution Complete!")

if __name__ == "__main__":
    main()
