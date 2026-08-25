# Deep Learning Datasets Plan — Implementation-Ready Specification

### KodeMapper AI-Driven IDPS · DL Pipeline Branch

---

> **Audience:** Developer implementing the DL branch. This document assumes familiarity with the existing classical ML pipeline ([train_combined_model.py](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/service/models/src/train_combined_model.py), [train_from_scratch.py](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/service/models/src/train_from_scratch.py), [ensemble_model.py](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/service/models/src/ensemble_model.py)) and the project architecture ([02_tech_and_architecture.md](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/documentation/02_tech_and_architecture.md)).

---

## Table of Contents

1. [Dataset Selection and Role Assignment](#1-dataset-selection-and-role-assignment)
2. [Feature-Selection Philosophy](#2-feature-selection-philosophy)
3. [Exact Feature Keep/Drop Specification](#3-exact-feature-keepdrop-specification)
4. [Categorical Feature Encoding Strategy](#4-categorical-feature-encoding-strategy)
5. [Class Imbalance Handling](#5-class-imbalance-handling)
6. [Rare and Low-Value Attack Class Policy](#6-rare-and-low-value-attack-class-policy)
7. [Train / Validation / Test Split Strategy](#7-train--validation--test-split-strategy)
8. [LSTM Sequence Construction](#8-lstm-sequence-construction)
9. [Autoencoder Training-Data Policy](#9-autoencoder-training-data-policy)
10. [Exploratory Data Analysis Plan](#10-exploratory-data-analysis-plan)
11. [Cross-Dataset Harmonization Strategy](#11-cross-dataset-harmonization-strategy)
12. [Artifact and Versioning Strategy](#12-artifact-and-versioning-strategy)
13. [Recommended Directory Structure](#13-recommended-directory-structure)

---

## 1. Dataset Selection and Role Assignment

### 1.1 The Three Candidate Datasets

| Property | UNSW-NB15 | CICIDS2017 | NSL-KDD |
|---|---|---|---|
| **Year** | 2015 | 2017 | 2009 (cleaned KDD99) |
| **Records** | ~2.54M (full), ~82K train / ~175K test (splits) | ~2.83M flows | ~125K train / ~22K test |
| **Attack families** | 9 categories + Normal | 14 attack types + Benign | 4 super-classes + Normal |
| **Feature count** | 49 raw | 78 raw | 41 raw |
| **Flow-level features** | Yes — Argus/Bro-derived flow features | Yes — CICFlowMeter-derived | Partially — many connection-level, some content-based |
| **Live-feasible features** | ~38 after dropping IPs, timestamps, IDs | ~55 after cleanup | ~30 but many are payload-derived or host-specific |
| **Label quality** | Good — minor whitespace issues in `attack_cat` | Good but some NaN/Infinity in computed features | Good — predefined train/test |
| **Schema overlap with live capture** | High — matches Zeek/CICFlowMeter output | High — CICFlowMeter output directly | Low — KDD99 features like `num_shells`, `su_attempted` require host telemetry |

### 1.2 Evaluation Criteria (First Principles)

We evaluate each dataset against five requirements that matter for a production IDPS:

| Criterion | Weight | Rationale |
|---|---|---|
| **Live-feature availability** | Critical | If a dataset's discriminative features cannot be extracted from live packets/flows, a model trained on it will fail at inference time. |
| **Schema compatibility with existing pipeline** | High | The ML engine already preprocesses UNSW-NB15 ([train_combined_model.py L56-58](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/service/models/src/train_combined_model.py#L56-L58)). Reusing schemas reduces integration cost. |
| **Label quality and class balance** | High | Clean, unambiguous labels reduce noise. Sufficient per-class volume enables robust per-class evaluation. |
| **Attack diversity and modernity** | Medium | Broader attack coverage improves generalization. Modern attack types are more relevant than legacy patterns. |
| **Research comparability** | Low | Useful for paper benchmarks, but secondary to operational performance. |

### 1.3 Role Assignment — Justified from Criteria

#### 🥇 UNSW-NB15 → Primary (Training + Evaluation)

**Justification:**

1. **Live-feature fit:** UNSW-NB15's feature set (flow duration, byte/packet counts, TTL, TCP flags, service, state, connection-count windows) maps directly to what Zeek `conn.log` and CICFlowMeter produce. The existing pipeline already proves this — [live_predictor_worker.py](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/service/models/src/live_predictor_worker.py) runs inference using exactly these features.

2. **Schema already integrated:** The entire preprocessing, encoding, and feature-selection pipeline is built for UNSW-NB15 columns. The DL branch must share the same preprocessing artifacts (encoders, scalers) to participate in the ensemble ([ensemble_model.py L44-51](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/service/models/src/ensemble_model.py#L44-L51)). Starting from a different primary dataset would require a parallel preprocessing stack — unnecessary complexity.

3. **Label quality:** After dropping the four weak classes (Analysis, Backdoor, Shellcode, Worms — see [Section 6](#6-rare-and-low-value-attack-class-policy)), the remaining 6 classes (Normal, Generic, Exploits, Fuzzers, DoS, Reconnaissance) each have ≥3,400 samples, which is sufficient for supervised DL.

4. **Proven baseline:** The classical ensemble achieves 89.75% accuracy / 81.58% macro recall on this dataset ([ml.md L459-475](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/docs_sahil/ml.md#L459-L475)). The DL branch has a concrete target to beat or complement.

#### 🥈 CICIDS2017 → Secondary (Augmentation + Generalization Testing)

**Justification:**

1. **Attack diversity:** CICIDS2017 includes attack types absent from UNSW-NB15 — Brute Force (FTP/SSH), DDoS variants, Web Attacks, Botnet, Infiltration. Training the LSTM on CICIDS flows (after harmonization) exposes the model to different temporal attack patterns.

2. **Feature overlap:** The core flow features (duration, fwd/bwd bytes, fwd/bwd packets, IAT stats, flag counts) exist in both datasets. Approximately 20–25 features can be aligned directly (see [Section 11](#11-cross-dataset-harmonization-strategy)).

3. **Generalization validation:** The cross-dataset experiments defined in [05_experimental_plan L189-196](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/documentation/05_experimental_plan_and_metrics.md#L189-L196) require CICIDS2017 as a test target. A DL model that performs well on UNSW-NB15 but collapses on CICIDS2017 is operationally useless.

4. **Not primary because:** The schema is different (78 CICFlowMeter columns vs. 49 UNSW columns), column names are inconsistent, and some CICIDS features contain NaN/Infinity values that require extra cleaning. Using it as primary would add integration friction without proportional benefit.

#### 📏 NSL-KDD → Benchmark Only (No Training)

**Justification:**

1. **Feature incompatibility:** NSL-KDD contains host-level features (`num_shells`, `su_attempted`, `is_host_login`, `num_root`, `num_file_creations`) that require operating-system telemetry. These do not exist in network flow data and would be zero/absent during live inference. A model that learns to rely on these features will fail silently in production.

2. **Attack obsolescence:** KDD99/NSL-KDD attacks (Neptune, Smurf, Pod) represent 1990s-era exploits. Training on these teaches outdated signatures that do not generalize to modern attacks.

3. **Benchmark value:** NSL-KDD remains the single most widely published-against IDS benchmark. Running the trained model against it (binary: attack vs. normal) provides comparison numbers for the literature review section.

> [!IMPORTANT]
> **NSL-KDD must never enter the training pipeline.** Use it only for post-training evaluation by mapping to binary classification (normal vs. attack) with the harmonized feature subset.

### 1.4 Summary Decision Matrix

| Dataset | Train | Validate | Test (Primary) | Cross-Dataset Test | Benchmark Comparison |
|---|---|---|---|---|---|
| **UNSW-NB15** | ✅ | ✅ | ✅ | — | ✅ |
| **CICIDS2017** | ✅ (harmonized subset) | ✅ (harmonized) | — | ✅ | ✅ |
| **NSL-KDD** | ❌ | ❌ | ❌ | ✅ (binary only) | ✅ |

---

## 2. Feature-Selection Philosophy

### 2.1 The Live-Feasible Principle

Every feature used by the DL models must satisfy this test:

> **Can this feature be computed from live network packets/flows captured by Zeek or CICFlowMeter at the monitoring point, without access to the dataset's metadata, host OS, or ground-truth labels?**

If the answer is no, the feature is dropped — regardless of its training-time predictive power.

**Why this is non-negotiable:** The existing [live_predictor_worker.py](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/service/models/src/live_predictor_worker.py) receives flow records from the collector → preprocessor pipeline ([02_tech_and_architecture.md](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/documentation/02_tech_and_architecture.md) data flow diagram). The DL models will receive the same feature vectors. If a feature doesn't exist in the live flow record, it will be zero or missing — and the model's learned weights for it become noise.

### 2.2 Three-Tier Feature Categorization

| Tier | Rule | Action |
|---|---|---|
| **Tier 1 — Live-feasible, behavioral** | Describes traffic behavior, computable from packet headers/flow stats | Keep |
| **Tier 2 — Metadata / identity** | IPs, ports (raw), timestamps, row IDs | Drop (identity leakage, non-generalizable) |
| **Tier 3 — Host-derived or dataset-only** | Features requiring OS-level data, or computed only in dataset post-processing | Drop |

### 2.3 Consistency with Classical Pipeline

The existing pipeline drops `srcip`, `dstip`, `id`, `Stime`, `Ltime` ([train_combined_model.py L56](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/service/models/src/train_combined_model.py#L56)). The DL pipeline must drop the same columns **plus** apply additional scrutiny because:

- Neural networks are more susceptible to overfitting on spurious correlations than tree ensembles.
- Raw port numbers (`sport`, `dsport`) are high-cardinality integers that trees handle via splits but dense layers memorize.
- The DL branch adds StandardScaler normalization ([train_dl_model.py L70-72](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/service/models/src/train_dl_model.py#L70-L72)), which means features with extreme outliers or non-informative distributions (constant, near-constant) can distort the scale.

---

## 3. Exact Feature Keep/Drop Specification

### 3.1 UNSW-NB15 Features — Full Decision Table

| Feature | Type | Decision | Rationale |
|---|---|---|---|
| `srcip` | Identity | **DROP** | IP address — not generalizable, privacy concern |
| `dstip` | Identity | **DROP** | Same as above |
| `sport` | Port | **DROP** | High cardinality integer (0–65535). Encode as `is_wellknown_sport` binary flag instead. |
| `dsport` | Port | **DROP** | Same as above. Encode as `is_wellknown_dsport` binary flag. |
| `id` | Metadata | **DROP** | Row index — zero information |
| `Stime` | Timestamp | **DROP** | Absolute time — not available in live. Used only for sequence ordering (before drop). |
| `Ltime` | Timestamp | **DROP** | Same as above |
| `attack_cat` | Label | **SEPARATE** | Target variable — never a feature |
| `label` | Label | **SEPARATE** | Binary target — used for autoencoder normal/attack split |
| `dur` | Flow | **KEEP** | Connection duration — core behavioral feature |
| `proto` | Categorical | **KEEP** | Protocol — TCP/UDP/ICMP. One-hot encode. |
| `service` | Categorical | **KEEP** | Service type (HTTP, FTP, DNS, etc.). One-hot encode top-N. |
| `state` | Categorical | **KEEP** | Connection state. One-hot encode. |
| `spkts` | Flow | **KEEP** | Source packets — volume indicator |
| `dpkts` | Flow | **KEEP** | Destination packets |
| `sbytes` | Flow | **KEEP** | Source bytes — key for DoS/DDoS detection |
| `dbytes` | Flow | **KEEP** | Destination bytes |
| `rate` | Flow | **KEEP** | Packets per second — core flow feature |
| `sttl` | Flow | **KEEP** | Source TTL — reflects OS fingerprint and routing |
| `dttl` | Flow | **KEEP** | Destination TTL |
| `sload` | Flow | **KEEP** | Source bits per second |
| `dload` | Flow | **KEEP** | Destination bits per second |
| `sloss` | Flow | **KEEP** | Source packets lost — retransmission indicator |
| `dloss` | Flow | **KEEP** | Destination packets lost |
| `sinpkt` | Flow | **KEEP** | Source inter-packet arrival time |
| `dinpkt` | Flow | **KEEP** | Destination inter-packet arrival time |
| `sjit` | Flow | **KEEP** | Source jitter |
| `djit` | Flow | **KEEP** | Destination jitter |
| `swin` | Flow | **KEEP** | Source TCP window size |
| `dwin` | Flow | **KEEP** | Destination TCP window size |
| `stcpb` | Flow | **KEEP** | Source TCP base sequence number |
| `dtcpb` | Flow | **KEEP** | Destination TCP base sequence number |
| `smeansz` | Flow | **KEEP** | Mean source packet size |
| `dmeansz` | Flow | **KEEP** | Mean destination packet size |
| `trans_depth` | Flow | **KEEP** | Transaction pipeline depth (HTTP) |
| `res_bdy_len` | Flow | **KEEP** | Response body length |
| `synack` | Flow | **KEEP** | SYN-ACK round-trip time |
| `ackdat` | Flow | **KEEP** | ACK data round-trip time |
| `tcprtt` | Flow | **KEEP** | TCP round-trip time |
| `ct_state_ttl` | Window | **KEEP** | Connection count per state/TTL — top feature per [ml.md L303-314](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/docs_sahil/ml.md#L303-L314) |
| `ct_flw_http_mthd` | Window | **KEEP** | HTTP method count |
| `is_ftp_login` | Binary | **KEEP** | FTP login flag |
| `ct_ftp_cmd` | Window | **KEEP** | FTP command count |
| `ct_srv_src` | Window | **KEEP** | Connections to same service from source |
| `ct_srv_dst` | Window | **KEEP** | Connections to same service at destination |
| `ct_dst_ltm` | Window | **KEEP** | Connections to same destination in last time window |
| `ct_src_ltm` | Window | **KEEP** | Connections from same source in last time window |
| `ct_src_dport_ltm` | Window | **KEEP** | Connections from source to same dest port in time window |
| `ct_dst_sport_ltm` | Window | **KEEP** | Connections to dest from same source port in time window |
| `ct_dst_src_ltm` | Window | **KEEP** | Connections between same src-dst pair in time window |
| `is_sm_ips_ports` | Binary | **KEEP** | Same IP and port flag |

### 3.2 Derived Features to Engineer

| New Feature | Formula | Purpose |
|---|---|---|
| `is_wellknown_sport` | `1 if sport < 1024 else 0` | Captures privilege without memorizing port numbers |
| `is_wellknown_dsport` | `1 if dsport < 1024 else 0` | Same for destination |
| `bytes_ratio` | `sbytes / (dbytes + 1)` | Asymmetry indicator — high ratio ≈ exfiltration or flooding |
| `pkts_ratio` | `spkts / (dpkts + 1)` | Packet asymmetry |
| `pkt_size_avg` | `(sbytes + dbytes) / (spkts + dpkts + 1)` | Average packet size across both directions |

### 3.3 Post-Selection Feature Count

After applying all keep/drop/derive decisions:

- **Raw UNSW-NB15 columns kept:** ~38
- **Derived features added:** 5
- **One-hot expanded features (proto, service, state):** ~15–25 depending on cardinality cap
- **Total input dimensionality (DL models):** approximately **55–65 features**

---

## 4. Categorical Feature Encoding Strategy

### 4.1 Encoding Method: One-Hot Encoding for DL

The existing classical pipeline uses `LabelEncoder` for categorical features ([train_combined_model.py L103-112](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/service/models/src/train_combined_model.py#L103-L112)). This is appropriate for tree models (which can split on ordinal values) but **inappropriate for neural networks** because:

- LabelEncoder assigns arbitrary integer codes (e.g., TCP=0, UDP=1, ICMP=2). A dense layer will interpret "UDP is between TCP and ICMP" — which is nonsensical.
- One-hot encoding creates orthogonal dimensions, which is correct for categorical data in gradient-based optimization.

### 4.2 Implementation Specification

```python
# Categorical features in UNSW-NB15
CATEGORICAL_FEATURES = ["proto", "service", "state"]

# Cardinality caps (to prevent dimension explosion)
CARDINALITY_CAPS = {
    "proto": None,     # Only ~3 values (tcp, udp, icmp) — keep all
    "service": 15,     # Top 15 services + "other" bucket
    "state": None,     # ~10 states — keep all
}
```

**Rules:**
1. Fit the one-hot encoder **only on training data**. Unknown categories at inference time map to all-zeros (the "other" bucket).
2. Save the fitted encoder as `dl_onehot_encoder_v{N}.pkl`.
3. For `service`, group all services outside the top-15 by frequency into an `_other` category before encoding.
4. The encoder must be **shared** between LSTM and Autoencoder pipelines — both models consume the same feature vectors.

### 4.3 Why Not Target Encoding or Embedding Layers?

- **Target encoding** leaks label information into features — unacceptable for a security system where false negatives carry high cost and overfitting is dangerous.
- **Embedding layers** are viable for the LSTM but not for the Autoencoder (which uses dense layers). Maintaining two different encoding strategies for the same feature creates drift risk. Use one-hot for both.

---

## 5. Class Imbalance Handling

### 5.1 The Problem — Quantified

UNSW-NB15 combined dataset after dropping weak classes (from [ml.md L80-91](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/docs_sahil/ml.md#L80-L91)):

| Class | Approx. Count | % of Total |
|---|---|---|
| Normal | ~93,000 | ~56% |
| Generic | ~40,000 | ~24% |
| Exploits | ~33,000 | ~20% |
| Fuzzers | ~18,000 | ~11% |
| DoS | ~12,000 | ~7% |
| Reconnaissance | ~10,000 | ~6% |

> [!WARNING]
> A model trained naively on this distribution will learn to predict Normal/Generic for everything and achieve ~80% accuracy while missing most real attacks. **Accuracy is not an acceptable metric for this system.**

### 5.2 Three-Layer Imbalance Mitigation

#### Layer 1 — SMOTE (Training Data Augmentation)

Apply SMOTE to the **training set only** (never validation or test).

```python
from imblearn.over_sampling import SMOTE

# Target: bring minority classes to at least 30% of majority class size
TARGET_RATIO = 0.3  # relative to largest class count
smote = SMOTE(
    sampling_strategy={
        cls: max(count, int(max_count * TARGET_RATIO))
        for cls, count in class_counts.items()
        if count < int(max_count * TARGET_RATIO)
    },
    k_neighbors=5,
    random_state=42
)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

**Apply SMOTE before sequence construction for LSTM**, not after — otherwise synthetic sequences will have inconsistent temporal ordering.

Wait — actually, for LSTM: SMOTE cannot be applied to pre-formed sequences because the synthetic interpolation would produce physically impossible flow sequences. Instead, use **class weights** (Layer 2) for LSTM. SMOTE is only used for the Autoencoder's supervised validation and for comparison experiments.

#### Layer 2 — Class Weights in Loss Function

For both LSTM and Autoencoder (when used in a supervised fine-tuning stage), compute inverse-frequency class weights:

```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(zip(np.unique(y_train), weights))
```

Pass `class_weight_dict` to PyTorch's `CrossEntropyLoss` via the `weight` parameter (as a tensor).

The existing DL script already does this ([train_dl_model.py L86-93](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/service/models/src/train_dl_model.py#L86-L93)) — carry it forward.

#### Layer 3 — Decision Threshold Tuning

The default argmax threshold (predict the class with highest probability) is suboptimal for security. Instead:

1. After training, generate prediction probabilities on the **validation set**.
2. For each class, sweep thresholds from 0.1 to 0.9 and compute per-class recall and precision.
3. Select per-class thresholds that maximize recall while keeping precision ≥ 30% (or per operational tolerance).
4. Store thresholds as a JSON artifact: `dl_thresholds_v{N}.json`.

The existing ensemble already applies per-class thresholds ([ensemble_model.py L165-172](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/service/models/src/ensemble_model.py#L165-L172)). The DL branch must produce similar per-class threshold configs.

---

## 6. Rare and Low-Value Attack Class Policy

### 6.1 Classes to Drop from DL Training

Consistent with the existing pipeline ([train_combined_model.py L65-72](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/service/models/src/train_combined_model.py#L65-L72)):

| Class | Sample Count (Full) | Decision | Reason |
|---|---|---|---|
| `Analysis` | ~677 | **DROP** | Ambiguous behavioral patterns, overlaps with Reconnaissance |
| `Backdoor` | ~583 | **DROP** | Too few samples for DL generalization; patterns are host-level |
| `Shellcode` | ~378 | **DROP** | Payload-based — undetectable at flow level |
| `Worms` | ~44 | **DROP** | Statistically meaningless — cannot train any model on 44 samples |

### 6.2 What Happens to Dropped-Class Traffic at Inference?

The Autoencoder covers this gap. Traffic from dropped classes will have **high reconstruction error** (because the Autoencoder has never seen these patterns). The anomaly score will flag them as suspicious, even though the supervised LSTM cannot classify them.

This is the architectural reason for running both models in parallel — the LSTM handles known attack types, and the Autoencoder catches everything else as anomalies.

### 6.3 Retained Classes

| Index | Class | Approximate Count |
|---|---|---|
| 0 | Normal | 93,000 |
| 1 | Generic | 40,000 |
| 2 | Exploits | 33,000 |
| 3 | Fuzzers | 18,000 |
| 4 | DoS | 12,000 |
| 5 | Reconnaissance | 10,000 |

---

## 7. Train / Validation / Test Split Strategy

### 7.1 Split Ratio

```
70% Training  │  15% Validation  │  15% Test
```

**Why 70/15/15 instead of 80/10/10:**
- The validation set serves dual duty: (a) early stopping / LR scheduling during training, and (b) threshold calibration for both the LSTM's per-class thresholds and the Autoencoder's anomaly threshold. With 10% validation, rare classes (DoS, Reconnaissance) may have fewer than 600 validation samples — insufficient for reliable threshold calibration.
- 15% test provides enough statistical power for per-class confidence intervals on recall.

### 7.2 Splitting Rules

1. **Stratified by `attack_cat`:** Every split must preserve the original class proportions. Use `sklearn.model_selection.train_test_split` with `stratify=y`.

2. **Temporal ordering for LSTM sequences:** Sort the full dataset by `Stime` **before** splitting. Then apply stratified split. This prevents the LSTM from seeing "future" flows during training. If stratification conflicts with temporal ordering (it will — some classes cluster in time), use the following compromise:
   - Sort by `Stime`.
   - Take the first 70% of rows as the training pool, the next 15% as validation, the last 15% as test.
   - If any class is absent from a split, fall back to stratified random split with `random_state=42` and document the compromise.

3. **Fixed seed:** `random_state=42` everywhere. Same seed used in existing pipeline.

4. **Split indices saved as artifact:** Save as `split_indices_v{N}.json` containing the row indices for each split. This ensures reproducibility and allows re-creating exact splits.

### 7.3 Per-Model Split Usage

| Split | LSTM | Autoencoder |
|---|---|---|
| **Training (70%)** | Full training data (all classes) for supervised classification | **Normal traffic only** — filter `label == 0` from training split |
| **Validation (15%)** | All classes — for early stopping + threshold tuning | All classes — for reconstruction error distribution and threshold tuning |
| **Test (15%)** | All classes — final evaluation, no tuning | All classes — final anomaly detection evaluation |

---

## 8. LSTM Sequence Construction

### 8.1 Why Sequences?

Network attacks like port scans, brute force, and DDoS manifest as **patterns across multiple consecutive flows** — not individual flows in isolation. An LSTM processes these patterns by reading a window of flows in order.

### 8.2 Sequence Construction Algorithm

```
Input: DataFrame with N rows, sorted by Stime, with columns [features..., attack_cat]
Parameters:
  - WINDOW_SIZE = 10  (start here; experiment with 15 and 20)
  - STRIDE = 1        (sliding window, overlapping)

For i in range(0, N - WINDOW_SIZE + 1, STRIDE):
    sequence = features[i : i + WINDOW_SIZE]     # shape: (10, n_features)
    label = attack_cat[i + WINDOW_SIZE - 1]       # label of the LAST flow in the window
    → Append (sequence, label) to dataset
```

**Shape output:** `(num_sequences, WINDOW_SIZE, n_features)` — this is the 3D tensor the LSTM expects.

### 8.3 Label Assignment

Assign the label of the **last flow** in the window. This is the standard causal-safe approach — the model predicts what is happening *now* based on the current and preceding flows.

**Why not majority vote?** Majority vote over the window would dilute attack labels — a window with 1 attack flow and 9 normal flows would be labeled "normal", causing the model to miss attack onsets.

### 8.4 Data Leakage Prevention — Critical Rules

> [!CAUTION]
> Violating any of these rules will produce inflated metrics that collapse in production.

1. **Split before sequencing:** Apply the 70/15/15 split on the **flat DataFrame** (individual flows). Then build sequences **within each split** independently. Never allow a sequence to span the train/validation or validation/test boundary.

2. **Fit scaler on training sequences only:** The `StandardScaler` (or `MinMaxScaler`) must be fit on the training split's feature values. Transform validation and test using the **same** fitted scaler.

3. **No future information:** Sort by `Stime` before splitting. Sequences in the training set must come from earlier time periods than validation and test sequences.

4. **Drop `Stime` after sorting:** Use `Stime` for ordering only — it is not a feature. Drop it before building the feature matrix.

### 8.5 Sequence Construction Parameters

| Parameter | Default | Experiment Range | Rationale |
|---|---|---|---|
| `WINDOW_SIZE` | 10 | [5, 10, 15, 20] | 10 flows ≈ a few seconds of traffic; captures most scan/brute patterns |
| `STRIDE` | 1 | [1, 5, 10] | Stride=1 maximizes training data but increases compute; stride=5 for faster iteration |
| `PAD_SHORT` | True | — | If a split has fewer than WINDOW_SIZE flows, zero-pad from the left |

### 8.6 Edge Case: Mixed-Source Sequences

In a live system, flows from different sources may be interleaved. For training, consider grouping flows by source-destination pair before sequencing (so each sequence represents one "conversation"). However, for UNSW-NB15, the dataset is not structured this way — flows are already semi-sorted by capture order. Start with simple chronological sliding windows; refine to per-pair grouping only if evaluation shows improvement.

---

## 9. Autoencoder Training-Data Policy

### 9.1 Core Principle

The Autoencoder is an **anomaly detector, not a classifier**. It learns the distribution of normal traffic. Anything it cannot reconstruct well is flagged as anomalous.

### 9.2 Training Data: Normal Traffic Only

```python
# From the training split (70%), extract only normal traffic
ae_train = X_train[y_train == "Normal"]  # or label == 0

# Validation set: contains BOTH normal and attack traffic
ae_val = X_val  # full validation set with all classes

# Test set: full test set with all classes
ae_test = X_test
```

### 9.3 Why Normal-Only Training?

1. If the Autoencoder sees attack traffic during training, it learns to reconstruct attacks — which means attacks will have LOW reconstruction error and be missed.
2. The unsupervised approach catches **novel attacks** that the LSTM (supervised) has never seen. This is the architectural justification for running both models.
3. This aligns with the project's goal of detecting "both known and unknown attacks" ([01_project_overview.md L22](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/documentation/01_project_overview.md#L22)).

### 9.4 Anomaly Threshold Calibration

After training, compute reconstruction error on the **validation set**:

```python
# Compute reconstruction error for each sample
val_predictions = autoencoder.predict(X_val)
reconstruction_errors = np.mean((X_val - val_predictions) ** 2, axis=1)

# Separate errors by class
normal_errors = reconstruction_errors[y_val == "Normal"]
attack_errors = reconstruction_errors[y_val != "Normal"]

# Method 1: Statistical threshold
threshold = np.mean(normal_errors) + 2.0 * np.std(normal_errors)

# Method 2: Percentile-based threshold (more robust to outliers)
threshold = np.percentile(normal_errors, 97)  # 97th percentile of normal errors
```

**Threshold sweep:** Try multipliers from 1.5 to 3.0 standard deviations (or percentiles from 90th to 99th). Select the one that maximizes recall on the validation attack traffic while keeping FPR on validation normal traffic below 5%.

### 9.5 Autoencoder Input

The Autoencoder receives the same **flat feature vector** as the classical models — **not sequences**. Each flow is independently encoded and reconstructed. Shape: `(batch_size, n_features)`.

---

## 10. Exploratory Data Analysis Plan

### 10.1 EDA Checklist — Execute Before Any Training

| # | Step | Output Artifact | Pass Criteria |
|---|---|---|---|
| 1 | Class distribution bar chart (per dataset) | `reports/class_distribution_{dataset}.png` | Verify counts match published numbers |
| 2 | Missing/NaN/Inf audit per column | `reports/missing_values_{dataset}.csv` | If column >30% missing → drop; else impute with training-set median |
| 3 | Duplicate row count and removal | Logged to `reports/eda_log_{dataset}.txt` | Remove exact duplicates; log count |
| 4 | Feature histograms (top 15 features by importance from classical pipeline) | `reports/feature_histograms_{dataset}.png` | Visual check for extreme skew; apply `log1p` if skew > 5 |
| 5 | Correlation heatmap (all numeric features) | `reports/correlation_heatmap_{dataset}.png` | If Pearson r > 0.95 between two features → drop one |
| 6 | Constant/near-constant feature detection (std < 1e-6) | Logged | Drop any constant features |
| 7 | Label quality check — unique values, whitespace, encoding | Logged | Strip whitespace from `attack_cat`; verify no unlabeled rows |
| 8 | Per-class feature mean comparison (normal vs. each attack) | `reports/per_class_feature_means_{dataset}.csv` | Identify discriminative features per attack type |
| 9 | Box plots of reconstruction-relevant features for AE | `reports/ae_feature_boxplots.png` | Verify normal traffic is clustered; attacks are outliers |
| 10 | Cross-dataset feature overlap analysis | `reports/cross_dataset_feature_map.csv` | Document which features exist in both UNSW and CICIDS |

### 10.2 EDA for CICIDS2017 — Additional Steps

| Step | Rationale |
|---|---|
| Check for `Infinity` values in `Flow Bytes/s`, `Flow Packets/s` | Known CICIDS issue — replace with `NaN` then impute |
| Check for negative flow durations | Known data quality issue — drop those rows |
| Verify day-by-day label distribution (Monday=benign, Tuesday=brute force, etc.) | Prevents accidental temporal bias if splitting by day |

---

## 11. Cross-Dataset Harmonization Strategy

### 11.1 The Golden Rule

> **Never blindly concatenate datasets with different schemas. Only merge features you have manually verified represent the same physical measurement.**

### 11.2 Unified Feature Mapping (UNSW-NB15 ↔ CICIDS2017)

| Unified Name | UNSW-NB15 Column | CICIDS2017 Column | Verified? | Notes |
|---|---|---|---|---|
| `flow_duration` | `dur` | `Flow Duration` (÷1e6 → seconds) | ✅ | CICIDS uses microseconds |
| `total_fwd_bytes` | `sbytes` | `Total Fwd Packets` × `Fwd Packet Length Mean` or `TotLen Fwd Pkts` | ⚠️ | Verify formula |
| `total_bwd_bytes` | `dbytes` | `TotLen Bwd Pkts` | ✅ | Direct match |
| `total_fwd_pkts` | `spkts` | `Total Fwd Packets` | ✅ | Direct match |
| `total_bwd_pkts` | `dpkts` | `Total Bwd Packets` | ✅ | Direct match |
| `fwd_pkt_len_mean` | `smeansz` | `Fwd Packet Length Mean` | ✅ | Direct match |
| `bwd_pkt_len_mean` | `dmeansz` | `Bwd Packet Length Mean` | ✅ | Direct match |
| `flow_iat_mean` | `sinpkt` (approximately) | `Flow IAT Mean` | ⚠️ | UNSW `sinpkt` is source-only; CICIDS is bidirectional. Use with caution. |
| `fwd_iat_mean` | `sinpkt` | `Fwd IAT Mean` | ✅ | Match |
| `bwd_iat_mean` | `dinpkt` | `Bwd IAT Mean` | ✅ | Match |
| `syn_flag_count` | Derived from `state` | `SYN Flag Count` | ⚠️ | UNSW encodes in `state`; extract if possible |
| `protocol` | `proto` | `Protocol` | ✅ | Both use standard protocol numbers/names |
| `flow_rate` | `rate` | Compute: `(Total Fwd Packets + Total Bwd Packets) / Flow Duration` | ⚠️ | Derived — verify units match |

### 11.3 Harmonization Process

```
Step 1: Load each dataset independently
Step 2: Apply dataset-specific cleaning (NaN/Inf for CICIDS, whitespace for UNSW)
Step 3: Map to unified column names using the table above
Step 4: Drop all columns not in the unified list
Step 5: Normalize each dataset independently (fit scaler per dataset)
Step 6: Add `_dataset_source` column ("unsw" / "cicids") for tracking
Step 7: Concatenate
Step 8: Drop `_dataset_source` before training
Step 9: Save as data/processed/combined/train_harmonized_v{N}.parquet
```

### 11.4 Unified Label Mapping for Cross-Dataset Use

| Unified Label | UNSW-NB15 | CICIDS2017 |
|---|---|---|
| `normal` | Normal | BENIGN |
| `dos` | DoS | DoS Hulk, DoS Slowloris, DoS SlowHTTPTest, DoS GoldenEye, DDoS |
| `reconnaissance` | Reconnaissance | PortScan |
| `exploits` | Exploits | Web Attack – Brute Force, Web Attack – XSS, Web Attack – SQL Injection |
| `brute_force` | — | FTP-Patator, SSH-Patator |
| `botnet` | Generic (approximate) | Bot |

> [!WARNING]
> The label mappings for `exploits` and `brute_force` are approximate. CICIDS "Web Attack" is not the same as UNSW "Exploits". For cross-dataset experiments, use **binary classification only** (normal vs. attack) to avoid semantic label mismatch.

### 11.5 What to Never Do

- Never merge raw CSVs without verifying unit compatibility (microseconds vs. seconds, bytes vs. bits).
- Never train on CICIDS and test on UNSW with multi-class labels — the attack taxonomies do not align.
- Never use the harmonized combined dataset as the primary evaluation — evaluate on each dataset separately first, then combine.

---

## 12. Artifact and Versioning Strategy

### 12.1 Naming Convention

```
{step}_{dataset}_{version}_{date}.{extension}
```

Examples:
```
scaler_unsw_v1_20260612.pkl
onehot_encoder_unsw_v1_20260612.pkl
train_split_indices_unsw_v1_20260612.json
lstm_sequences_unsw_w10_v1_20260612.npy
ae_train_normal_unsw_v1_20260612.parquet
preprocessing_config_v1_20260612.json
```

### 12.2 Mandatory Artifacts to Save

| Artifact | Format | Purpose |
|---|---|---|
| Fitted `StandardScaler` | `.pkl` (joblib) | Reproduce exact scaling at inference |
| Fitted one-hot encoder | `.pkl` (joblib) | Reproduce exact encoding at inference |
| Feature list (ordered) | `.json` | Document exact input features and order |
| Split indices | `.json` | Reproduce exact splits |
| LSTM sequence arrays | `.npy` | Cache built sequences for fast re-training |
| AE normal-only training set | `.parquet` | Reproduce AE training data |
| Preprocessing config | `.json` | All hyperparams: window size, stride, cardinality caps, SMOTE params, threshold |
| EDA reports | `.png` + `.csv` | Audit trail |
| SHA-256 checksums of raw data | `checksums.sha256` | Verify data integrity ([05_experimental_plan L267](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/documentation/05_experimental_plan_and_metrics.md#L267)) |

### 12.3 `preprocessing_config.json` Template

```json
{
    "version": "v1",
    "date": "2026-06-12",
    "dataset": "unsw_nb15",
    "random_seed": 42,
    "split_ratio": {"train": 0.70, "val": 0.15, "test": 0.15},
    "dropped_columns": ["srcip", "dstip", "id", "Stime", "Ltime", "sport", "dsport"],
    "dropped_attack_classes": ["Analysis", "Backdoor", "Shellcode", "Worms"],
    "categorical_features": ["proto", "service", "state"],
    "categorical_encoding": "one_hot",
    "service_cardinality_cap": 15,
    "scaler_type": "StandardScaler",
    "smote_applied": true,
    "smote_target_ratio": 0.3,
    "smote_k_neighbors": 5,
    "lstm_window_size": 10,
    "lstm_stride": 1,
    "lstm_label_strategy": "last_flow",
    "ae_training_data": "normal_only",
    "ae_threshold_method": "percentile_97",
    "derived_features": ["is_wellknown_sport", "is_wellknown_dsport", "bytes_ratio", "pkts_ratio", "pkt_size_avg"]
}
```

---

## 13. Recommended Directory Structure

```
data/
├── raw/                              # Unmodified downloads
│   ├── UNSW_NB15_training-set.csv
│   ├── UNSW_NB15_testing-set.csv
│   ├── cicids2017/                   # Multiple day CSVs
│   │   ├── Monday-WorkingHours.pcap_ISCX.csv
│   │   ├── Tuesday-WorkingHours.pcap_ISCX.csv
│   │   └── ...
│   ├── nslkdd/
│   │   ├── KDDTrain+.txt
│   │   └── KDDTest+.txt
│   └── checksums.sha256
│
├── processed/                        # Cleaned, encoded, split
│   ├── unsw/
│   │   ├── train_v1.parquet
│   │   ├── val_v1.parquet
│   │   └── test_v1.parquet
│   ├── cicids/
│   │   ├── train_v1.parquet
│   │   ├── val_v1.parquet
│   │   └── test_v1.parquet
│   └── combined/                     # Harmonized union
│       ├── train_harmonized_v1.parquet
│       └── val_harmonized_v1.parquet
│
├── sequences/                        # LSTM-ready 3D arrays
│   ├── lstm_train_w10_v1.npy
│   ├── lstm_train_labels_w10_v1.npy
│   ├── lstm_val_w10_v1.npy
│   ├── lstm_val_labels_w10_v1.npy
│   ├── lstm_test_w10_v1.npy
│   └── lstm_test_labels_w10_v1.npy
│
├── autoencoder/                      # AE-specific data
│   ├── ae_train_normal_v1.parquet
│   └── ae_val_mixed_v1.parquet
│
├── artifacts/                        # Preprocessing artifacts
│   ├── scaler_unsw_v1.pkl
│   ├── onehot_encoder_unsw_v1.pkl
│   ├── feature_list_v1.json
│   ├── split_indices_v1.json
│   └── preprocessing_config_v1.json
│
└── reports/                          # EDA outputs
    ├── class_distribution_unsw.png
    ├── class_distribution_cicids.png
    ├── correlation_heatmap_unsw.png
    ├── missing_values_unsw.csv
    ├── missing_values_cicids.csv
    ├── feature_statistics_unsw.csv
    ├── cross_dataset_feature_map.csv
    └── eda_log.txt
```

This structure aligns with the existing `data/` directory ([project structure](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/README.md#L36-L37)) and the `experiments/` convention described in [03_implementation_plan.md L24](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/documentation/03_implementation_plan.md#L24).

---

## Pre-Implementation Checklist

Before writing any training code, verify:

- [ ] Raw data downloaded and checksummed in `data/raw/`
- [ ] EDA completed for UNSW-NB15 — all 10 checklist items
- [ ] EDA completed for CICIDS2017 — including Infinity/NaN audit
- [ ] Non-live features dropped (srcip, dstip, id, Stime, Ltime, sport, dsport)
- [ ] Derived features engineered (is_wellknown_sport, bytes_ratio, etc.)
- [ ] Weak attack classes dropped (Analysis, Backdoor, Shellcode, Worms)
- [ ] Categorical features one-hot encoded; encoder saved
- [ ] StandardScaler fit on training split only; scaler saved
- [ ] 70/15/15 stratified split created; indices saved
- [ ] LSTM sequences built within each split (no cross-boundary leakage)
- [ ] Autoencoder training set contains only normal traffic from training split
- [ ] CICIDS2017 harmonized feature mapping documented and verified
- [ ] All artifacts saved with version-date naming convention
- [ ] `preprocessing_config.json` saved documenting all parameters

---

*Document Version: 2.0 | Project: KodeMapper AI-IDPS Deep Learning Pipeline*
*Supersedes: dl_datasets_plan.md v1.0 (beginner-level overview)*
