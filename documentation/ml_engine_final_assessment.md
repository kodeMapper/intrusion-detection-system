# Machine Learning Engine Final Assessment

This document provides a comprehensive, honest, and technically rigorous assessment of the Intrusion Detection and Prevention System (IDPS) engine. It details what was expected in the project planning documents, what has been implemented, and an evaluation of the system's ability to detect both known and unknown (zero-day) attacks.

---

## 1. Expectations vs. Implementation

The following table summarizes the requirements outlined in [03_implementation_plan.md](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/documentation/03_implementation_plan.md) and [dl_models_plan.md](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/documentation/dl_models_plan.md) compared to the final built codebase:

| Component | Planned Specification | Deployed Implementation |
| :--- | :--- | :--- |
| **Feature Extraction & Preprocessing** | Fit Scalers and Categorical Encoders on traffic datasets. | Implemented in [dl_preprocessor.py](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/service/preproc/dl_preprocessor.py) using serialized `RobustScaler` and custom mapping encoders. |
| **ML Baseline Engine** | Classifiers to detect known static tabular signatures. | Implemented in [unified_predictor_worker.py](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/service/models/src/unified_predictor_worker.py#L58-L106) as a soft-voting ensemble averaging prediction probabilities of XGBoost, Random Forest, and LightGBM models. |
| **DL Stage 1: Binary Gate** | Sequence-aware model to filter Normal vs. Attack flows. | Implemented in [unified_predictor_worker.py](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/service/models/src/unified_predictor_worker.py#L112-L153) using a sequence-based [TemporalTransformerClassifier](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/service/models/src/dl_pipeline/lightning_modules.py#L101). |
| **DL Stage 2: Multiclass Classifier** | Sequence-aware model classifying attacks into specific families. | Deployed as a secondary [TemporalTransformerClassifier](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/service/models/src/dl_pipeline/lightning_modules.py#L101) loaded in the unified predictor. |
| **Zero-Day Canary** | Unsupervised anomaly model to detect novel attacks. | Deployed in [unified_predictor_worker.py](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/service/models/src/unified_predictor_worker.py#L235-L277) using a Transformer-based [TemporalOneClassVAE](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/service/models/src/dl_pipeline/lightning_modules.py#L215). |
| **Unified Pipeline** | Parallel execution, MongoDB storage, and UI alerts. | Orchestrated by [unified_predictor_worker.py](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/service/models/src/unified_predictor_worker.py#L347-L406) running all three engines concurrently and passing verdicts to the API server. |

---

## 2. Detection of Known Attacks

### Performance Metrics
Based on the test split evaluations in [dl_stage1_binary_report_v1.0.json](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/service/models/artifacts/dl_stage1_binary_report_v1.0.json) and [dl_stage2_multiclass_report_v1.0.json](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/service/models/artifacts/dl_stage2_multiclass_report_v1.0.json):
* **Stage 1 (Binary Gate) Accuracy:** **97.55%**
* **Stage 1 (Binary Gate) Attack Recall:** **99.97%** 
* **Stage 1 (Binary Gate) Attack F1-Score:** **98.09%**
* **Stage 2 (Multiclass) Recall rates:**
  * **Generic:** 96.97%
  * **Fuzzers:** 87.90%
  * **Reconnaissance:** 83.13%
  * **DoS:** 88.18%
  * **Exploits:** 41.25%

### Assessment
* **Ensemble Redundancy:** By running the tabular ML baseline (RF + XGB + LGBM) in parallel with the DL sequence models, the system ensures that known signatures are captured by the tree ensembles while sequence-oriented attacks are captured by the Transformers.
* **Honest Caveat - Exploits vs. DoS Multiclass Confusion:** The Stage 2 model has a low recall for `Exploits` (41.25%), frequently misclassifying them as `DoS` (due to feature overlap in temporal flow signatures). However, because **Stage 1 catches 99.97% of attacks**, these flows are still correctly identified as malicious. The system fails only at naming the exact attack family, not at detecting its presence.

---

## 3. Detection of Unknown (Zero-Day) Attacks

### Detection Mechanism
The unsupervised [TemporalOneClassVAE](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/service/models/src/dl_pipeline/lightning_modules.py#L215) is trained **exclusively on benign (normal) traffic**. 
* **Low Reconstruction Error:** When normal traffic is processed, the model compresses and reconstructs it with high fidelity.
* **High Reconstruction Error:** When a zero-day attack occurs, its statistical features and timing footprint are unfamiliar to the VAE. The reconstruction fails, resulting in a spike in reconstruction error.
* **Canary Trigger:** If the error exceeds the dynamic threshold, the system flags the flow with `zero_day_flag = true`.

### Honest Caveats
* **Behavioral Mimicry:** A highly sophisticated attacker who purposefully throttles their attack to look statistically identical to normal user behavior can bypass the Autoencoder.
* **Benign Anomalies (False Positives):** Legitimately rare network behaviors (e.g., system updates, backup synchronization) will trigger high reconstruction errors, causing false positive zero-day warnings.
* **Advisory Verdict:** To protect availability, the zero-day flag is advisory. It alerts the security operator on the dashboard but does not alter the core binary verdict if the supervised models are highly confident the flow is benign.

---

## 4. Why We Transitioned from LSTM to Transformers

While the codebase maintains scripts for training LSTMs ([train_dl_lstm.py](file:///d:/Sarang/Skills/Linux%20Shared%20Folder/Project%20Repo/service/models/src/train_dl_lstm.py)), we transitioned to **Temporal Transformers** in production due to key structural advantages:

1. **Direct Multi-Step Correlation:** LSTMs process flows step-by-step sequentially, which degrades information density over longer windows. Transformers use self-attention to correlate all events in a sliding window directly.
2. **Parallel Processing Speed:** LSTMs require sequential calculation, limiting throughput. Transformers compute sequence steps in parallel, maximizing packet processing speeds.
3. **Gradient Stability:** Transformers utilize Layer Normalization and residual paths, avoiding vanishing/exploding gradients common during sequence training in recurrent models.
