import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = ROOT / "data" / "raw" / "nslkdd"
REPORTS_DIR = ROOT / "data" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def run_nslkdd_eda():
    print("Loading NSL-KDD dataset...")
    train_path = RAW_DATA_DIR / "KDDTrain+.txt"
    test_path = RAW_DATA_DIR / "KDDTest+.txt"
    
    # NSL-KDD usually doesn't have headers in the txt file
    try:
        df_train = pd.read_csv(train_path, header=None)
        df_test = pd.read_csv(test_path, header=None)
        df = pd.concat([df_train, df_test], ignore_index=True)
    except Exception as e:
        print(f"Error loading NSL-KDD: {e}")
        return
        
    print(f"Total rows: {len(df)}")
    
    # 1. Class distribution (Label is typically the 41st column, index 41)
    label_col = 41
    print("1. Generating class distribution...")
    plt.figure(figsize=(10, 6))
    
    # Group rare attacks for better visualization if needed, but let's just plot top 20
    top_labels = df[label_col].value_counts().nlargest(20).index
    sns.countplot(y=df[df[label_col].isin(top_labels)][label_col], order=top_labels)
    plt.title("Class Distribution (NSL-KDD) - Top 20")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "class_distribution_nslkdd.png")
    plt.close()

    # 2. Missing/NaN/Inf audit
    print("2. Auditing missing values and Infinity...")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = df[num_cols].apply(lambda x: np.isinf(x).sum())
    inf_counts = inf_counts[inf_counts > 0]
    
    audit_df = pd.DataFrame({'Missing': missing, 'Infinity': inf_counts})
    audit_df.to_csv(REPORTS_DIR / "missing_values_nslkdd.csv")
    
    # 3. Duplicate rows
    print("3. Checking duplicates...")
    dup_count = df.duplicated().sum()
    with open(REPORTS_DIR / "eda_log_nslkdd.txt", "w") as f:
        f.write(f"Duplicate rows found: {dup_count}\n")
    
    # 4. Feature histograms (sample top 15 numeric)
    print("4. Generating histograms...")
    top_num_cols = num_cols[:15]
    
    fig, axes = plt.subplots(5, 3, figsize=(15, 20))
    axes = axes.flatten()
    for i, col in enumerate(top_num_cols):
        sns.histplot(df[col].dropna(), bins=50, ax=axes[i], log_scale=(False, True))
        axes[i].set_title(f"Feature {col}")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "feature_histograms_nslkdd.png")
    plt.close()

    # 5. Correlation heatmap
    print("5. Generating correlation heatmap...")
    corr = df[num_cols].corr()
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr, cmap='coolwarm', center=0)
    plt.title("Correlation Heatmap (NSL-KDD)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "correlation_heatmap_nslkdd.png")
    plt.close()

    print("NSL-KDD EDA Complete! Reports saved to data/reports/")

if __name__ == "__main__":
    run_nslkdd_eda()
