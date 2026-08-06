import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = ROOT / "data" / "raw"
REPORTS_DIR = ROOT / "data" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def run_unsw_eda():
    print("Loading UNSW-NB15 dataset...")
    # Load and combine train/test to get full view
    train_path = RAW_DATA_DIR / "UNSW_NB15_training-set.csv"
    test_path = RAW_DATA_DIR / "UNSW_NB15_testing-set.csv"
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df = pd.concat([df_train, df_test], ignore_index=True)
    
    # 1. Class distribution
    print("1. Generating class distribution...")
    plt.figure(figsize=(10, 6))
    sns.countplot(y='attack_cat', data=df, order=df['attack_cat'].value_counts().index)
    plt.title("Class Distribution (UNSW-NB15)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "class_distribution_unsw.png")
    plt.close()

    # 2. Missing/NaN/Inf audit
    print("2. Auditing missing values...")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    
    # Check for inf
    inf_counts = df.select_dtypes(include=[np.number]).apply(lambda x: np.isinf(x).sum())
    inf_counts = inf_counts[inf_counts > 0]
    
    audit_df = pd.DataFrame({'Missing': missing, 'Infinity': inf_counts})
    audit_df.to_csv(REPORTS_DIR / "missing_values_unsw.csv")
    
    # 3. Duplicate rows
    print("3. Checking duplicates...")
    dup_count = df.duplicated().sum()
    with open(REPORTS_DIR / "eda_log_unsw.txt", "w") as f:
        f.write(f"Duplicate rows found: {dup_count}\n")
    
    # 4. Feature histograms (sample top 15 numeric)
    print("4. Generating histograms...")
    num_cols = df.select_dtypes(include=[np.number]).columns
    top_cols = num_cols[:15] # Just picking first 15 for demo
    
    fig, axes = plt.subplots(5, 3, figsize=(15, 20))
    axes = axes.flatten()
    for i, col in enumerate(top_cols):
        sns.histplot(df[col], bins=50, ax=axes[i], log_scale=(False, True))
        axes[i].set_title(col)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "feature_histograms_unsw.png")
    plt.close()

    # 5. Correlation heatmap
    print("5. Generating correlation heatmap...")
    corr = df[num_cols].corr()
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr, cmap='coolwarm', center=0)
    plt.title("Correlation Heatmap (UNSW-NB15)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "correlation_heatmap_unsw.png")
    plt.close()

    # 6. Constant features
    print("6. Detecting constant features...")
    stds = df[num_cols].std()
    constant_features = stds[stds < 1e-6].index.tolist()
    with open(REPORTS_DIR / "eda_log_unsw.txt", "a") as f:
        f.write(f"Constant numeric features: {constant_features}\n")

    # 7. Label quality
    print("7. Checking label quality...")
    unique_labels = df['attack_cat'].unique().tolist()
    with open(REPORTS_DIR / "eda_log_unsw.txt", "a") as f:
        f.write(f"Unique raw labels: {unique_labels}\n")
        
    # Strip whitespace
    df['attack_cat'] = df['attack_cat'].str.strip()
    clean_labels = df['attack_cat'].unique().tolist()
    with open(REPORTS_DIR / "eda_log_unsw.txt", "a") as f:
        f.write(f"Cleaned unique labels: {clean_labels}\n")

    # 8. Per-class feature means
    print("8. Calculating per-class means...")
    class_means = df.groupby('attack_cat')[num_cols].mean()
    class_means.to_csv(REPORTS_DIR / "per_class_feature_means_unsw.csv")

    # 9. Box plots for AE relevant features
    print("9. Generating box plots for AE...")
    ae_features = ['dur', 'sbytes', 'dbytes', 'rate', 'sttl']
    df_sample = df.sample(n=min(10000, len(df)), random_state=42)
    fig, axes = plt.subplots(1, len(ae_features), figsize=(20, 5))
    for i, col in enumerate(ae_features):
        sns.boxplot(x='label', y=col, data=df_sample, ax=axes[i])
        axes[i].set_yscale('log')
        axes[i].set_title(col)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "ae_feature_boxplots_unsw.png")
    plt.close()

    print("UNSW-NB15 EDA Complete! Reports saved to data/reports/")

if __name__ == "__main__":
    run_unsw_eda()
