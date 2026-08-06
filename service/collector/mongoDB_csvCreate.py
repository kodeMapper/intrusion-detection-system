# =====================================
# Generate Live Dataset (for testing pipeline - MongoDB)
# =====================================

import pandas as pd
from pathlib import Path

print("Generating Live Dataset...")


TOTAL_SAMPLES = 1000
NORMAL_RATIO = 0.80
ATTACK_RATIO = 0.20
RANDOM_STATE = 42


# =====================================
# Fix Project Root
# =====================================

ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = ROOT / "data" / "raw" / "UNSW_NB15_training-set.csv"

print("Loading Dataset From:")
print(DATASET_PATH)


# =====================================
# Load Dataset
# =====================================

if not DATASET_PATH.exists():
    print("Dataset Not Found")
    exit()

df = pd.read_csv(DATASET_PATH)

print("\nOriginal Dataset Shape:")
print(df.shape)


# =====================================
# Remove Weak Attacks
# =====================================

weak = [

    "Analysis",
    "Backdoor",
    "Shellcode",
    "Worms"

]

df = df[~df["attack_cat"].isin(weak)]


print("\nAfter Removing Weak Attacks")

print(df["attack_cat"].value_counts())


# =====================================
# Create Live Dataset (1000 Samples)
# =====================================

normal_count = int(TOTAL_SAMPLES * NORMAL_RATIO)
attack_count = int(TOTAL_SAMPLES * ATTACK_RATIO)

generic_count = attack_count // 4
dos_count = attack_count // 4
fuzzers_count = attack_count // 4
other_count = attack_count - (generic_count + dos_count + fuzzers_count)


def sample_rows(dataframe, count):
    replace = count > len(dataframe)
    return dataframe.sample(count, random_state=RANDOM_STATE, replace=replace)


normal_df = df[df["attack_cat"] == "Normal"]
generic_df = df[df["attack_cat"] == "Generic"]
dos_df = df[df["attack_cat"] == "DoS"]
fuzzers_df = df[df["attack_cat"] == "Fuzzers"]
other_df = df[
    (~df["attack_cat"].isin(["Normal", "Generic", "DoS", "Fuzzers"]))
]

live_data = pd.concat(
    [
        sample_rows(normal_df, normal_count),
        sample_rows(generic_df, generic_count),
        sample_rows(dos_df, dos_count),
        sample_rows(fuzzers_df, fuzzers_count),
        sample_rows(other_df, other_count),
    ],
    ignore_index=True,
)


# =====================================
# Shuffle Dataset
# =====================================

live_data = live_data.sample(frac=1, random_state=RANDOM_STATE)


# =====================================
# Display Distribution
# =====================================

print("\nLive Dataset Distribution:")

print(live_data["attack_cat"].value_counts())


# =====================================
# Save Dataset
# =====================================

SAVE_PATH = ROOT / "tests" / "live_test_dataset.csv"

SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

live_data.to_csv(SAVE_PATH, index=False)


print("\nLive Dataset Saved At:")

print(SAVE_PATH)


print("\nLive Dataset Generation Completed")