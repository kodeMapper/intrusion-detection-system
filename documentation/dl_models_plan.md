# Deep Learning Models Plan — Implementation-Ready Specification

### KodeMapper AI-Driven IDPS · DL Pipeline Branch

---

> **Audience:** Developer implementing the DL branch. This document assumes familiarity with the existing classical ML pipeline ([train_stage2_major.py](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/service/models/src/train_stage2_major.py), [train_stage1_binary.py](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/service/models/src/train_stage1_binary.py), and [ensemble_model.py](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/service/models/src/ensemble_model.py)) and the datasets design plan ([dl_datasets_plan.md](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/documentation/dl_datasets_plan.md)).

---

## Table of Contents

1. [Architectural Overview & Design Philosophy](#1-architectural-overview--design-philosophy)
2. [Alignment with the Two-Layer (Stage-1/Stage-2) ML Architecture](#2-alignment-with-the-two-layer-stage-1stage-2-ml-architecture)
3. [Sequential LSTM Specification](#3-sequential-lstm-specification)
4. [Unsupervised Anomaly Autoencoder Specification](#4-unsupervised-anomaly-autoencoder-specification)
5. [Training, Optimization, & Regularization Policies](#5-training-optimization--regularization-policies)
6. [Hyperparameter Tuning Strategy](#6-hyperparameter-tuning-strategy)
7. [Training Curves & Experiment Tracking](#7-training-curves--experiment-tracking)
8. [Decision Integration & Hybrid Inference (Ensemble Strategy)](#8-decision-integration--hybrid-inference-ensemble-strategy)
9. [Production Concerns & Mitigation Strategies](#9-production-concerns--mitigation-strategies)
10. [Serialization & Directory Layout](#10-serialization--directory-layout)
11. [Step-by-Step Implementation Scripts](#11-step-by-step-implementation-scripts)

---

## 1. Architectural Overview & Design Philosophy

The deep learning (DL) branch implements two distinct neural architectures, each designed to fill a specific gap that traditional tree-based models (XGBoost, LightGBM) cannot address. Unlike classical models that treat each flow in isolation and rely on known signatures, the DL branch introduces temporal sequential dependencies and unsupervised anomaly detection.

```
                   Raw Network Flows (from Preprocessor)
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │  Harmonized Preprocessing   │
                      │  (One-Hot & scaling)        │
                      └──────────────┬──────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
          ┌──────────────┐                       ┌──────────────┐
          │  Seq. LSTM   │                       │ Autoencoder  │
          │ (Temporal)   │                       │  (Unsup. AE) │
          └──────┬───────┘                       └──────┬───────┘
                 │                                      │
            Class Probas                           Recon Error
                 │                                      │
                 └───────────────────┬───────────────────┘
                                     ▼
                      ┌─────────────────────────────┐
                      │  Hybrid Decision Fusion &    │
                      │  Ensemble Voting            │
                      └──────────────┬──────────────┘
                                     ▼
                                Alert / Pass
```

### 1.1 Model Roles & Justifications

* **Sequential LSTM (Long Short-Term Memory):** Learns the temporal relationships across consecutive network flows. This is critical for detecting multi-stage attacks, scanning/reconnaissance sweeps, and slow-rate DoS/exfiltration attempts that are invisible in isolated flow records.
* **Unsupervised Anomaly Autoencoder (AE):** Trained exclusively on normal traffic. It measures the reconstruction error of incoming flows. If the reconstruction error exceeds a dynamically calculated threshold, the flow is flagged as an anomaly. This serves as a vital safety net to capture zero-day attacks and dropped minor classes (`Analysis`, `Backdoor`, `Shellcode`, `Worms`) that have too few training samples for supervised classification.

### 1.2 Common Preprocessing Bridge

To ensure optimal performance and avoid feature drift, both models consume feature vectors preprocessed by the same pipeline:
1. Feature Keep/Drop selection as defined in [dl_datasets_plan.md](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/documentation/dl_datasets_plan.md).
2. Cardinality-capped One-Hot Encoding for categorical features (`proto`, `service`, `state`).
3. Standard scaling (fitted on train split normal traffic for Autoencoder, and full train split for LSTM).

---

## 2. Alignment with the Two-Layer (Stage-1/Stage-2) ML Architecture

The existing classical ML engine operates as a two-stage pipeline to solve a specific operational challenge:

1. **Stage 1 (Binary Filter):** Maximizes throughput and minimizes false negatives. It acts as a gatekeeper (`stage1_binary.pkl`), trained to separate all normal traffic from potential attack traffic (binary classification). Normal traffic is passed immediately, preventing unnecessary processing downstream.
2. **Stage 2 (Multi-Class Ensemble):** Gated behind Stage 1. Once traffic is flagged as an anomaly or attack, the multi-class ensemble (`stage2_major_xgb.pkl`, `stage2_major_rf.pkl`, `stage2_major_lgbm.pkl`) classifies the exact attack type (`Generic`, `Exploits`, `Fuzzers`, `DoS`, `Reconnaissance`). This classification determines the specific SOAR-lite playbook (e.g. rate-limiting for DoS, port blocking for Reconnaissance).

### 2.1 Integrating Deep Learning into the Two-Stage Paradigm

The deep learning models align directly with this two-layer design and enhance it at both levels. They run in parallel to the classical models and are integrated through the following structure:

```
                            Incoming Raw Flow
                                    │
                                    ▼
                      ┌──────────────────────────┐
                      │ Preprocessing & Scaling  │
                      └─────────────┬────────────┘
                                    │
                                    ├───┐
                                    ▼   ▼
                     ┌──────────────────┐  ┌──────────────────┐
                     │Stage-1 Classifier│  │   Autoencoder    │
                     │ (Binary XGBoost) │  │(Unsupervised AE) │
                     └──────────┬───────┘  └────────┬─────────┘
                                │                   │
                          Prob > Thresh?    Recon Error > Ethresh?
                                │                   │
                                ├───┬───────────────┘
                                │   │ (Any Yes = Flagged)
                                ▼
                           Is Flagged?
                               ├─── No ──▶ [PASS / Legitimate]
                               └─── Yes
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │   Stage-2 Classifiers   │
                       ├─────────────────────────┤
                       │  - LightGBM (35%)       │
                       │  - XGBoost (30%)        │
                       │  - LSTM (25%)           │
                       │  - Random Forest (10%)  │
                       └────────────┬────────────┘
                                    │
                                    ▼
                        [Ensemble Voting Prob]
                                    │
                         Tuned Per-Class Thresh?
                               ├─── Yes ──▶ [Trigger Alert & Playbook]
                               └─── No  ──▶ [PASS / Legitimate]
```

### 2.2 Functional Stage Mapping of DL Models

* **Stage 1 Alignment:** The **Anomaly Autoencoder (AE)** is integrated at Stage 1. It operates in parallel with the Stage 1 binary XGBoost classifier. While the binary XGBoost utilizes supervised patterns to flag known attack signatures, the unsupervised Autoencoder uses reconstruction error to catch novel zero-day threats and rare attack classes that were dropped from supervised training (`Analysis`, `Backdoor`, `Shellcode`, `Worms`). If *either* the Stage 1 binary classifier *or* the Autoencoder flags a flow, it is forwarded to Stage 2.
* **Stage 2 Alignment:** The **FlowLSTM** is integrated at Stage 2. Once a flow passes Stage 1, it is processed by the full ensemble. The LSTM adds temporal sequence history, providing crucial context that tree-based models lack. It participates in the soft voting ensemble alongside XGBoost, Random Forest, and LightGBM.

---

## 3. Sequential LSTM Specification

The LSTM processes sequences of flow vectors to detect temporal signatures.

### 3.1 Sequence Construction Policy

* **Sequence Length ($T$):** Formulated as a sequence of the last $T = 10$ flows.
* **Grouping Strategy:** Sequences are grouped per source IP address (`srcip`) to capture attack paths. If grouped globally, temporal dynamics from unrelated hosts get mixed. Thus, we maintain a rolling FIFO queue of length $10$ per active host in the preprocessor.
* **Padding Policy:** For hosts with fewer than $10$ historical flows, we pad the sequence at the beginning (pre-padding) with zero vectors.
* **Label Assignment:** The label of the sequence is the label of the **most recent (last) flow** in the sequence. This ensures real-time evaluation capability.

```
Time step t-9:  [ Flow Feature Vector (scaled) ]
Time step t-8:  [ Flow Feature Vector (scaled) ]
  ...
Time step t:    [ Flow Feature Vector (scaled) ]  <── Target Label associated here
```

### 3.2 LSTM Topology

```
Input Sequence: Shape (Batch, T=10, d_in)
  ├── bidirectional LSTM (128 hidden units, 2 layers, dropout = 0.3)
  ├── Tensor Slice: Take the final hidden state of the forward/backward directions
  ├── Dense Layer (64 units)
  ├── ReLU Activation
  ├── Dropout (p = 0.3)
  └── Dense Out (6 units) ──▶ Softmax (Class Probabilities)
```

### 3.3 PyTorch Class Definition

```python
import torch
import torch.nn as nn

class FlowLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, num_classes: int = 6):
        super(FlowLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        # Bidirectional doubling input to dense layer (hidden_dim * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Seq_Len, Input_Dim)
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Extract the last time-step output for both directions
        # lstm_out shape: (Batch, Seq_Len, Hidden_Dim * 2)
        last_step_out = lstm_out[:, -1, :]
        
        # Class probabilities
        logits = self.fc(last_step_out)
        return logits
```

---

## 4. Unsupervised Anomaly Autoencoder Specification

The Autoencoder learns a compressed representation (latent space) of normal network traffic. Since it is trained exclusively on benign traffic, it fails to reconstruct attack behaviors accurately, resulting in a high reconstruction error.

### 4.1 Topology

```
Input Flow Vector: Shape (d_in)
  │
  ├── Encoder:
  │     ├── Dense (64 units) ──▶ ReLU
  │     ├── Dense (32 units) ──▶ ReLU
  │     └── Dense (16 units) [Latent space] ──▶ ReLU
  │
  └── Decoder:
        ├── Dense (32 units) ──▶ ReLU
        ├── Dense (64 units) ──▶ ReLU
        └── Dense (d_in) [Reconstructed output] ──▶ Linear
```

### 4.2 Mathematical Formulation of Anomaly Score

The anomaly score is the Mean Squared Error (MSE) between the input vector $x$ and the reconstructed vector $\hat{x}$:

$$E(x) = \frac{1}{d} \sum_{i=1}^{d} (x_i - \hat{x}_i)^2$$

Where $d$ is the number of features in the input vector.

### 4.3 PyTorch Class Definition

```python
class AnomalyAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super(AnomalyAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
```

### 4.4 Threshold Tuning Policy ($E_{thresh}$)

The anomaly threshold is determined dynamically during training:
1. Compute the MSE reconstruction errors on the **benign validation subset** after training.
2. Sort the errors and locate the **99th percentile** ($P_{99}$) of the error distribution.
3. Set the anomaly threshold $E_{thresh} = P_{99}$.
4. In production, any flow $x$ with $E(x) > E_{thresh}$ is flagged as anomalous.

---

## 5. Training, Optimization, & Regularization Policies

To ensure models converge reliably, we establish strict hyperparameters:

### 5.1 Supervised Model (LSTM)

* **Loss Function:** Weighted Cross-Entropy Loss to handle class imbalances.
  
  $$\text{Loss} = -\sum_{c=1}^{C} w_c \cdot y_c \log(\hat{y}_c)$$
  
  Where $w_c$ is computed using the inverse class frequency.
* **Optimizer:** AdamW with weight decay $1\times 10^{-4}$ (to enforce weight regularization).
* **Learning Rate (LR):** Initial LR of $1\times 10^{-3}$ with a dynamic scheduler:
  * `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)`
* **Early Stopping:** Terminate training if `val_loss` does not improve for 15 epochs.
* **Batch Size:** 256.

### 5.2 Unsupervised Autoencoder

* **Loss Function:** Mean Squared Error (MSE) Loss.
* **Training Data:** Strictly filtered to contain only records where `label == 0` (Normal/Benign).
* **Optimizer:** Adam with learning rate $1\times 10^{-3}$.
* **Regularization:** L2 regularization via weight decay of $1\times 10^{-5}$ on decoder weights to prevent identity mapping memorization.

---

## 6. Hyperparameter Tuning Strategy

To find the optimal network configurations without manual trial-and-error, we define a structured Hyperparameter Search Space. We utilize `Optuna` to run 50 trials per model on the training validation splits, optimizing for **Macro Recall** (LSTM) and **F1 Anomaly Score** (Autoencoder).

### 6.1 Search Space Specifications

| Model | Hyperparameter | Search Range | Scale Type |
|---|---|---|---|
| **Flow LSTM** | Hidden Dimensions | $\{64, 128, 256\}$ | Discrete |
| | LSTM Layers | $\{1, 2, 3\}$ | Discrete |
| | Sequence Length ($T$) | $\{5, 10, 15, 20\}$ | Discrete |
| | Bidirectional Flag | $\{\text{True}, \text{False}\}$ | Categorical |
| | Learning Rate | $[10^{-4}, 10^{-2}]$ | Logarithmic |
| **Autoencoder**| Latent Dimensions | $\{8, 16, 24, 32\}$ | Discrete |
| | Encoder Layers | $\{2, 3, 4\}$ | Discrete |
| | Weight Decay | $[10^{-6}, 10^{-4}]$ | Logarithmic |

### 6.2 Run Gating and Selection
Each hyperparameter trial is evaluated on the validation split. The model weights with the highest validation score are serialized, and their configurations are pushed to the MLflow Experiment Registry.

---

## 7. Training Curves & Experiment Tracking

Every training run must produce verifiable performance logs to diagnose underfitting, overfitting, and gradient issues.

### 7.1 Required Curves and Visualizations

1. **Loss vs. Iteration (Batch level):** Plots training cost per step. Essential for catching optimization spikes, exploding gradients, and checking learning rate stability.
2. **Train vs. Validation Loss (Epoch level):** Plots both loss values side-by-side. Used to define early-stopping boundaries.
3. **Per-Class Recall / F1-Score vs. Epoch:** Evaluates if minority classes are being learned over time, showing the impact of SMOTE and class weighting.
4. **AE Reconstruction Error Distribution Histogram:** Separates the reconstruction error distributions of normal validation flows and attack validation flows. The separation gap verifies the classification viability of $E_{thresh}$.

### 7.2 Expected Visual Metrics (Conceptual Representations)

#### Loss vs. Iterations (LSTM Training)
```
  Loss
  ▲
  │  *   *
  │   * *  *
  │    *    ** *
  │             ** *   *
  │                 **** * *
  │                         ******
  └───────────────────────────────────▶ Iterations
```

#### Reconstruction Error Distribution (Benign vs. Attack)
```
  Frequency
  ▲
  │     Benign Flows
  │      ┌───┐                 Attack Flows
  │     ┌┘   └┐                     ┌──┐
  │    ┌┘     └┐                   ┌┘  └┐
  │   ┌┘       └┐                 ┌┘    └┐
  │  ┌┘         └┐      ▲        ┌┘      └┐
  └──┴───────────┴──────┼────────┴────────┴──▶ Reconstruction MSE
                        │
                  E_thresh (P99 Benign)
```

### 7.3 MLflow Logging Scheme

All training runs are tracked via MLflow:

```python
import mlflow

mlflow.set_experiment("KodeMapper_DL_IDS_Expansion")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("model_type", "FlowLSTM")
    mlflow.log_param("seq_len", 10)
    mlflow.log_param("learning_rate", 0.001)
    
    # Log epoch metrics
    for epoch in range(epochs):
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_macro_recall", val_recall, step=epoch)
```

---

## 8. Decision Integration & Hybrid Inference (Ensemble Strategy)

During live deployment, predictions from classical models and the LSTM are integrated into a multi-tiered decision system in Stage 2.

### 8.1 Weighted Voting Integration

The probabilities of the 4 multi-class models are averaged using highly optimized performance weights. The classical tree models receive the bulk of the weight for static feature excellence, while the LSTM holds significant sway for its temporal context.

$$P_{ensemble}(c) = \sum_{m \in M} w_m \cdot P_m(c)$$

| Model ($m$) | Architecture Type | Assigned Weight ($w_m$) | Rationale |
|---|---|---|---|
| **LightGBM** | Classical Tree Ensemble | **0.35** | Historically achieves the highest macro recall baseline; robust on tabular. |
| **XGBoost** | Classical Tree Ensemble | **0.30** | Extremely accurate tree ensemble, acts as the primary baseline stabilizer. |
| **LSTM** | Deep Sequential | **0.25** | Crucial weight because it is the **only** model that understands temporal history and sequencing. |
| **Random Forest**| Classical Tree Ensemble | **0.10** | A stable baseline model used to smooth out predictions. |

### 8.2 Per-Class Threshold Map

To maximize recall on minority attacks (e.g. `Fuzzers`, `DoS`), we replace the standard `argmax` operation with tuned per-class thresholds:

```python
# Tuned thresholds - prediction is accepted only if probability exceeds value.
# Otherwise, falls back to highest normalized class score.
CLASS_THRESHOLDS = {
    "DoS": 0.35,
    "Exploits": 0.40,
    "Fuzzers": 0.25,
    "Generic": 0.45,
    "Normal": 0.50,
    "Reconnaissance": 0.40
}
```

### 8.3 Backend and Dashboard Flow

1. The FastAPI backend orchestrates incoming flows from Redis.
2. It executes Stage 1 inference (XGBoost + Autoencoder). If flagged by either, it invokes the Stage 2 ensemble.
3. The resulting alert is pushed via WebSockets to the React Dashboard.
4. The dashboard displays the probability vector and reconstruction error alongside SHAP feature attributions, allowing security analysts to inspect why an alert was triggered.

---

## 9. Production Concerns & Mitigation Strategies

Operating DL models on live, high-speed network traffic introduces operational risks that do not appear in offline validation.

### 9.1 Temporal and Spatial Data Leakage

* **The Leakage Vector:** In sequence construction for the LSTM, sliding windows overlap (e.g., Sequence 1 covers flows 1–10, Sequence 2 covers flows 2–11). If sequences are shuffled *before* splitting into train/validation sets, adjacent overlapping sequences will be split across sets, causing the validation set to memorize training flow patterns.
* **Mitigation Strategy:** Perform the dataset split **chronologically** on raw flow records *before* sequence construction. We split the first 70% of flows for training, the next 15% for validation, and the final 15% for testing. Sequences are then built strictly within the boundaries of each partition. No sequence window can bridge across the train/validation or validation/test boundaries.

### 9.2 Concept Drift and Model Decay

* **The Drift Vector:** Network traffic profiles drift constantly due to software updates, changing user behavior, and new application protocols. This leads to false positives in the Autoencoder as normal traffic patterns shift, and missed threats in the LSTM.
* **Mitigation Strategy:** 
  1. **Statistical Drift Monitor:** Execute a daily Kolmogorov-Smirnov (KS) test on the distributions of core numerical features (`rate`, `sbytes`, `dbytes`) comparing live traffic against the training baseline. If the KS statistic exceeds $0.15$ for multiple core features, trigger a drift alert.
  2. **Autoencoder Reconstruction Monitor:** Track the rolling 24-hour mean of the Autoencoder's reconstruction error on traffic classified as normal. A sustained increase in average normal reconstruction error indicates that the baseline has drifted and the models require retraining.

### 9.3 Reproducibility and Seed Control

To ensure deterministic builds, we enforce a strict seed initialization block at the top of all scripts:

```python
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## 10. Serialization & Directory Layout

To support reproducibility, versioning, and direct load/unload capabilities, all serialization paths are strictly predefined.

### 10.1 Model Serialization Naming Convention

```
dl_{model_type}_v{major}.{minor}.pt
```

* `dl_lstm_v1.0.onnx`: ONNX compiled temporal LSTM.
* `dl_ae_v1.0.onnx`: ONNX compiled unsupervised Autoencoder.
* `dl_ae_threshold_v1.0.json`: JSON configuration holding the float value of the reconstruction threshold ($E_{thresh}$).
* `dl_scalers_v1.0.pkl`: Standard scaler fit on the training data.
* `dl_onehot_v1.0.pkl`: One-hot encoder fit on categorical attributes.

### 10.2 Layout of `service/models/artifacts`

```
service/models/artifacts/
├── final_encoders.pkl            # Legacy ML encoders
├── final_labels.pkl              # Legacy ML labels
├── final_xgb.pkl                 # Legacy Stage-2 XGBoost
├── final_rf.pkl                  # Legacy Stage-2 RF
├── final_lgbm.pkl                # Legacy Stage-2 LGBM
├── stage1_binary.pkl             # Legacy Stage-1 Binary XGBoost
│
├── dl_lstm_v1.0.onnx             # NEW: PyTorch LSTM compiled to ONNX
├── dl_ae_v1.0.onnx               # NEW: PyTorch Autoencoder compiled to ONNX
├── dl_ae_threshold_v1.0.json     # NEW: Anomaly reconstruction threshold configuration
├── dl_scalers_v1.0.pkl           # NEW: Scaler for Deep Learning features
├── dl_onehot_v1.0.pkl            # NEW: One-Hot encoder for categorical fields
└── dl_labels_v1.0.pkl            # NEW: Target label encoder mapping for DL
```

---

## 11. Step-by-Step Implementation Scripts

This section outlines the code structure and logic of the training scripts that will produce the above artifacts.

### 11.1 Sequential LSTM Training Script: `train_dl_lstm.py`

Constructs sequences and trains the LSTM model.

```python
# Draft Implementation of service/models/src/train_dl_lstm.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib
from pathlib import Path

class SequenceFlowDataset(Dataset):
    def __init__(self, X, y, seq_len=10):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.X) - self.seq_len + 1
        
    def __getitem__(self, idx):
        # Slice sequence window
        x_seq = self.X[idx : idx + self.seq_len]
        # Label is assigned to the last item in the sequence
        y_target = self.y[idx + self.seq_len - 1]
        return x_seq, y_target
```

### 11.2 Autoencoder Training Script: `train_dl_autoencoder.py`

Filters for normal traffic, trains the Autoencoder, and calculates the reconstruction error threshold.

```python
# Draft Implementation of service/models/src/train_dl_autoencoder.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Autoencoder trained on Normal class ONLY
# Validation normal traffic used to set reconstruction error threshold:
# E_thresh = percentile(errors, 99)
```

---
*Document Version: 3.0 — Proposed as part of the KodeMapper DL Expansion Plan.*
*Supersedes: dl_models_plan.md v2.0 (FNN architecture removed)*
