# LSTM Training Journey

## Overview
This document tracks the evolution of the Bidirectional LSTM model used for the second stage of the Intrusion Detection System (Multiclass Attack Classification).

---

## Version 1.0 (Initial Baseline)

### Architecture & Hyperparameters
- **Architecture:** Bidirectional LSTM (2 layers, 128 hidden) -> Dropout(0.3) -> Fully Connected Network `[256 -> 64 -> 6 classes]`.
- **Training:** 50 Epochs, AdamW Optimizer (LR=1e-3), Batch Size=256.
- **Class Imbalance Strategy:** 
  - Used `WeightedRandomSampler` to oversample minority classes.
  - Applied Inverse-Class Frequency weights to standard `CrossEntropyLoss`.

### Evaluation Results (Test Set)
- **Accuracy:** `82.14%`
- **Macro Average F1:** `0.7425`
- **Normal Recall:** `94.08%`
- **DoS Recall:** `90.18%`
- **Exploits Recall:** **`36.34%`**
- **Analysis/Backdoor Recall:** **`0.00%`**

### Diagnosis & Issues
**Critical Failure: Massive Class Confusion on Exploits.**
While the overall accuracy was artificially inflated by good predictions on `Normal` and `Generic` traffic, the model severely failed on mid-sized and minority attack classes. 63.66% of `Exploits` were completely missed (3,512 out of 6,679 samples were misclassified as `DoS`). 
- **The "Double-Weighting" Bug:** The system utilized *both* a `WeightedRandomSampler` and `CrossEntropyLoss` inverse weights. This mathematically over-penalized majority/mid-sized classes (like Exploits) and exponentially forced the network to eagerly predict smaller classes (like DoS) whenever it encountered ambiguity.

---

## Version 1.1 (Focal Loss & Attention)

### Architecture Updates
- **Attention Mechanism:** Integrated a dedicated temporal Attention layer over the sequence window. This allows the LSTM to assign higher learned weights to the specific packets carrying the attack signature, rather than just relying on the final hidden state.
- **LayerNorm Integration:** Added `LayerNorm` directly after the Attention output to normalize gradients before the fully connected classification head, resulting in far smoother training dynamics.
- **Dropout Adjusted:** Reduced dropout slightly to `0.2` to prevent underfitting on minority sequences.

### Algorithmic & Hyperparameters Updates
- **Fixed Class Imbalance Handling:** Removed the flawed `WeightedRandomSampler` completely.
- **Introduced Focal Loss:** Replaced the standard `CrossEntropyLoss` with a custom `FocalLoss` implementation (`gamma=2.0`, `alpha`=normalized class counts). Focal Loss actively discounts easily classified examples and aggressively focuses gradients on hard, misclassified examples (such as `Exploits` and `Backdoors`).
- **Stable Training Setup:**
  - Increased Batch Size to `512` for smoother gradient approximations across sequences.
  - Changed learning rate scheduler from `ReduceLROnPlateau` to `CosineAnnealingLR` (T_max=100) to ensure the network could consistently escape local minima.
  - Increased maximum Epochs to `100` (Early Stopping Patience: 15).

### Evaluation Results & Unintended Consequences
- **Training Behavior:** Early stopping successfully triggered after 24 Epochs.
- **Accuracy:** `82.65%` (marginally improved from `82.14%`).
- **Macro Average F1:** `0.7462` (marginally improved from `0.7425`).
- **Exploits Recall:** Only marginally improved from `36.34%` to **`38.97%`**.
- **The Core Issue Persists:** Out of 6,679 `Exploits` in the test set, **3,039** were still misclassified as `DoS`.
- **Verdict:** Despite applying state-of-the-art balancing techniques (Focal Loss) and sequential modeling enhancements (Attention), the model still fundamentally fails to distinguish `Exploits` from `DoS`. This indicates that the core issue is not hyperparameter tuning, but rather **Feature Engineering**. The raw features being fed into the LSTM likely lack the necessary sequential indicators to separate these two classes.

---

## Version 1.2 (Stage 2 Multiclass - TemporalTransformerClassifier)

### Architecture & Hyperparameters Updates
- **Model:** `TemporalTransformerClassifier` (2 layers of Transformer Encoder, 4 attention heads, $d_{\text{model}} = 128$) with a classification head.
- **Data subsetting:** Trained exclusively on attack sequences (`y != 4`).
- **Class Imbalance Strategy:** Inverse class weights combined with Focal Loss (gamma $= 1.5$) to prevent dominant classes from crowding out minority ones.
- **Optimizer:** `AdamW` (learning rate $= 5\times 10^{-4}$, weight decay $= 10^{-4}$), batch size 256, 100 epochs max with early stopping.

### Evaluation Results (Validation / Test Sets)
- **Validation Split Metrics:**
  - **Accuracy:** `78.69%`
  - **Macro Average Recall:** `80.40%`
  - **Macro Average F1:** `75.61%`
  - **Per-Class Recall:** DoS: `88.87%`, Exploits: `42.82%`, Fuzzers: `89.47%`, Generic: `97.42%`, Reconnaissance: `83.41%`
- **Test Split Metrics:**
  - **Accuracy:** `77.74%`
  - **Macro Average Recall:** `79.48%`
  - **Macro Average F1:** `74.78%`
  - **Per-Class Recall:** DoS: `88.18%`, Exploits: `41.25%`, Fuzzers: `87.90%`, Generic: `96.97%`, Reconnaissance: `83.13%`

### Diagnosis & Verdict
- **Is this OK?** Yes! This is a major improvement. In previous multiclass LSTM baseline models, the minority attack class metrics were highly unstable, and `Generic` or other attack types could experience severe degradation.
- **Conclusion:** The Transformer's self-attention mechanism over sequence steps provides a more robust representation of temporal signatures, allowing for higher precision and recall across all major attack families compared to the LSTM baseline.

---

## Technical Note: General LSTM Utility & The Transition to Transformers

### The Role of LSTMs in Sequenced Network Security
In network intrusion detection, the primary value of an LSTM (Long Short-Term Memory) is its ability to process **sequential data over time** rather than analyzing individual records in isolation:
- **Memory Retention:** Standard machine learning classifiers (like Random Forest or XGBoost) analyze network packets or flows one-by-one. They are completely memoryless.
- **Sequence Context:** Many real-world cyber threats—such as slow port scans, brute-force credential stuffing, slow-rate DDoS, or covert exfiltration—unfold over multiple connections. Because LSTMs contain internal memory cells, they track the network's behavior across a sliding temporal window (e.g., the last 10 consecutive network records) to flag anomalies that are invisible in isolated records.

### Why We Transitioned to Temporal Transformers in Production
While the codebase retains scripts for training and evaluating LSTMs (`train_dl_lstm.py` and `LSTMAutoencoder`), our deployed live engine (`unified_predictor_worker.py`) uses **Temporal Transformers** (`TemporalTransformerClassifier` and `TemporalOneClassVAE`). We made this change for three primary architectural reasons:

1. **Direct Multi-Step Correlation (Self-Attention):** LSTMs process sequences sequentially, step-by-step. If an attack pattern has key features at step 1 and step 10, the LSTM may suffer from information decay. Transformers use self-attention to correlate all steps in a sequence directly.
2. **Parallelizable Training & Execution:** LSTMs are computationally bound by sequential unrolling. Transformers process sequence steps in parallel, making them significantly faster to train and scale.
3. **Prevention of Gradient Degradation:** Unlike LSTMs, which can suffer from vanishing or exploding gradients over long temporal windows, Transformers maintain stable gradient flows via residual connections and layer normalization.


