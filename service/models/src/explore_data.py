import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "UNSW_NB15_training-set.csv"

df = pd.read_csv(DATASET_PATH)

print("Dataset Shape:", df.shape)
print(df.head())
print(df.columns)
print(df["attack_cat"].value_counts())

df.info()
df.isnull().sum()
