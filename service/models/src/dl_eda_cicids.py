import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = ROOT / "data" / "raw" / "cicids2017"
REPORTS_DIR = ROOT / "data" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def run_cicids_eda():
    print("Loading CICIDS2017 dataset...")
    csv_files = list(RAW_DATA_DIR.glob("*.csv"))
    
    # Load a fraction of each file to save memory
    dfs = []
    for f in csv_files:
        print(f"Sampling {f.name}...")
        try:
            # Strip whitespace from column names while reading
            df_part = pd.read_csv(f, skipinitialspace=True, encoding='latin1')
            df_part.columns = df_part.columns.str.strip()
            
            # Sample 20% to keep memory usage reasonable
            if len(df_part) > 10000:
                df_part = df_part.sample(frac=0.2, random_state=42)
            dfs.append(df_part)
        except Exception as e:
            print(f"Error reading {f.name}: {e}")
            
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total sampled rows: {len(df)}")
    
    # 1. Class distribution
    print("1. Generating class distribution...")
    plt.figure(figsize=(10, 6))
    if 'Label' in df.columns:
        label_col = 'Label'
    else:
        label_col = [c for c in df.columns if 'label' in c.lower()][0]
        
    sns.countplot(y=label_col, data=df, order=df[label_col].value_counts().index)
    plt.title("Class Distribution (CICIDS2017)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "class_distribution_cicids.png")
    plt.close()

    # 2. Missing/NaN/Inf audit
    print("2. Auditing missing values and Infinity...")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    
    # Check for inf
    num_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = df[num_cols].apply(lambda x: np.isinf(x).sum())
    inf_counts = inf_counts[inf_counts > 0]
    
    audit_df = pd.DataFrame({'Missing': missing, 'Infinity': inf_counts})
    audit_df.to_csv(REPORTS_DIR / "missing_values_cicids.csv")
    
    # 3. Duplicate rows
    print("3. Checking duplicates...")
    dup_count = df.duplicated().sum()
    with open(REPORTS_DIR / "eda_log_cicids.txt", "w") as f:
        f.write(f"Duplicate rows found in sample: {dup_count}\n")
    
    # 4. Feature histograms (sample top 15 numeric)
    print("4. Generating histograms...")
    top_cols = num_cols[:15]
    
    # Replace Inf with NaN just for plotting
    df_plot = df[top_cols].replace([np.inf, -np.inf], np.nan)
    
    fig, axes = plt.subplots(5, 3, figsize=(15, 20))
    axes = axes.flatten()
    for i, col in enumerate(top_cols):
        sns.histplot(df_plot[col].dropna(), bins=50, ax=axes[i], log_scale=(False, True))
        axes[i].set_title(col)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "feature_histograms_cicids.png")
    plt.close()

    # 5. Correlation heatmap
    print("5. Generating correlation heatmap...")
    corr = df_plot.corr()
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr, cmap='coolwarm', center=0)
    plt.title("Correlation Heatmap (CICIDS2017)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "correlation_heatmap_cicids.png")
    plt.close()

    print("CICIDS2017 EDA Complete! Reports saved to data/reports/")

if __name__ == "__main__":
    run_cicids_eda()
