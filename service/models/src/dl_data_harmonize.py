import pandas as pd
import numpy as np
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = ROOT / "data" / "raw"
PROCESSED_COMBINED_DIR = ROOT / "data" / "processed" / "combined"
REPORTS_DIR = ROOT / "data" / "reports"

PROCESSED_COMBINED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

UNSW_MAPPING = {
    'dur': 'flow_duration',
    'sbytes': 'total_fwd_bytes',
    'dbytes': 'total_bwd_bytes',
    'spkts': 'total_fwd_pkts',
    'dpkts': 'total_bwd_pkts',
    'smeansz': 'fwd_pkt_len_mean',
    'dmeansz': 'bwd_pkt_len_mean',
    'sinpkt': 'fwd_iat_mean',
    'dinpkt': 'bwd_iat_mean',
    'proto': 'protocol',
    'rate': 'flow_rate'
}

CICIDS_MAPPING = {
    'Flow Duration': 'flow_duration',
    'Total Length of Fwd Packets': 'total_fwd_bytes',
    'TotLen Fwd Pkts': 'total_fwd_bytes',
    'Total Length of Bwd Packets': 'total_bwd_bytes',
    'TotLen Bwd Pkts': 'total_bwd_bytes',
    'Total Fwd Packets': 'total_fwd_pkts',
    'Total Backward Packets': 'total_bwd_pkts',
    'Total Bwd packets': 'total_bwd_pkts',
    'Fwd Packet Length Mean': 'fwd_pkt_len_mean',
    'Bwd Packet Length Mean': 'bwd_pkt_len_mean',
    'Fwd IAT Mean': 'fwd_iat_mean',
    'Bwd IAT Mean': 'bwd_iat_mean',
    'Flow IAT Mean': 'flow_iat_mean',
    'Protocol': 'protocol',
    'Flow Packets/s': 'flow_rate'
}

UNSW_LABEL_MAPPING = {
    'Normal': 'normal',
    'DoS': 'dos',
    'Reconnaissance': 'reconnaissance',
    'Exploits': 'exploits',
    'Generic': 'botnet'
}

def clean_cicids_label(l):
    if pd.isna(l): return 'unknown'
    l = str(l).strip()
    if l == 'BENIGN': return 'normal'
    if 'DoS' in l or 'DDoS' in l: return 'dos'
    if 'PortScan' in l: return 'reconnaissance'
    if 'Web Attack' in l: return 'exploits'
    if 'Patator' in l or 'Brute Force' in l: return 'brute_force'
    if 'Bot' in l: return 'botnet'
    return 'unknown'

def clean_unsw_label(l):
    if pd.isna(l): return 'unknown'
    l = str(l).strip()
    return UNSW_LABEL_MAPPING.get(l, 'unknown')

def process_unsw():
    print("Processing UNSW-NB15...")
    train_path = RAW_DATA_DIR / "UNSW_NB15_training-set.csv"
    test_path = RAW_DATA_DIR / "UNSW_NB15_testing-set.csv"
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df = pd.concat([df_train, df_test], ignore_index=True)
    
    # Drop weak classes
    weak_classes = ["Analysis", "Backdoor", "Shellcode", "Worms"]
    df['attack_cat'] = df['attack_cat'].str.strip().fillna('Normal')
    df = df[~df['attack_cat'].isin(weak_classes)].copy()
    
    # Map label
    df['unified_label'] = df['attack_cat'].apply(clean_unsw_label)
    df = df[df['unified_label'] != 'unknown']
    
    # Rename columns
    df = df.rename(columns=UNSW_MAPPING)
    
    # Protocol mapping (UNSW uses names, CICIDS uses numbers typically. Let's keep names and map numbers)
    # Actually, we can just leave protocol as is, or we'll map commonly used numbers to names later.
    
    cols_to_keep = list(UNSW_MAPPING.values()) + ['unified_label']
    df = df[[c for c in cols_to_keep if c in df.columns]].copy()
    df['_dataset_source'] = 'unsw'
    return df

def process_cicids():
    print("Processing CICIDS2017...")
    cicids_dir = RAW_DATA_DIR / "cicids2017"
    csv_files = list(cicids_dir.glob("*.csv"))
    
    dfs = []
    for f in csv_files:
        print(f"  Sampling {f.name}...")
        try:
            df_part = pd.read_csv(f, skipinitialspace=True, encoding='latin1')
            df_part.columns = df_part.columns.str.strip()
            # 5% sample for speed in harmonization pipeline
            if len(df_part) > 10000:
                df_part = df_part.sample(frac=0.05, random_state=42)
            dfs.append(df_part)
        except Exception as e:
            print(f"Error reading {f.name}: {e}")
            
    df = pd.concat(dfs, ignore_index=True)
    
    if 'Label' in df.columns:
        label_col = 'Label'
    else:
        label_col = [c for c in df.columns if 'label' in c.lower()][0]
        
    df['unified_label'] = df[label_col].apply(clean_cicids_label)
    df = df[df['unified_label'] != 'unknown']
    
    # Replace Infinity with NaN
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with NaN or negative durations
    df = df.dropna()
    if 'Flow Duration' in df.columns:
        df = df[df['Flow Duration'] > 0]
        # Convert microsec to sec
        df['Flow Duration'] = df['Flow Duration'] / 1e6
        
    # Protocol mapping: 6 is TCP, 17 is UDP, 0 is HOPOPT / ICMP? Usually 1 is ICMP
    proto_map = {6: 'tcp', 17: 'udp', 1: 'icmp'}
    if 'Protocol' in df.columns:
        df['Protocol'] = df['Protocol'].map(proto_map).fillna('other')
        
    df = df.rename(columns=CICIDS_MAPPING)
    
    cols_to_keep = list(UNSW_MAPPING.values()) + ['unified_label'] # Use UNSW values as unified list
    # Ensure columns exist
    df = df[[c for c in cols_to_keep if c in df.columns]].copy()
    df['_dataset_source'] = 'cicids'
    return df

def main():
    df_unsw = process_unsw()
    df_cicids = process_cicids()
    
    print("Harmonizing datasets...")
    # Find common features
    common_features = list(set(df_unsw.columns).intersection(set(df_cicids.columns)))
    
    # Reorder columns
    df_unsw = df_unsw[common_features]
    df_cicids = df_cicids[common_features]
    
    df_combined = pd.concat([df_unsw, df_cicids], ignore_index=True)
    
    # Generate feature map report
    overlap_data = {"Feature": common_features}
    overlap_df = pd.DataFrame(overlap_data)
    overlap_df.to_csv(REPORTS_DIR / "cross_dataset_feature_map.csv", index=False)
    
    from sklearn.model_selection import train_test_split
    
    print("Splitting harmonized dataset...")
    # Stratified 70/30 split into train/val
    train_df, val_df = train_test_split(
        df_combined, 
        test_size=0.30, 
        stratify=df_combined['unified_label'], 
        random_state=42
    )
    
    # Save as parquet
    print("Saving to parquet...")
    train_df.to_parquet(PROCESSED_COMBINED_DIR / "train_harmonized_v1.parquet", index=False)
    val_df.to_parquet(PROCESSED_COMBINED_DIR / "val_harmonized_v1.parquet", index=False)
    
    print("Harmonization Complete!")
    print(f"Train set shape: {train_df.shape}")
    print(f"Val set shape: {val_df.shape}")

if __name__ == "__main__":
    main()
