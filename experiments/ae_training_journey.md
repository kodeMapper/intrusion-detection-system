# Autoencoder Training Journey

## Overview
This document tracks the evolution of the Anomaly Autoencoder used for the first stage of the Intrusion Detection System (Binary Anomaly Detection).

---

## Version 1.0 (Initial Baseline)

### Architecture & Hyperparameters
- **Encoder Structure:** `[Input(198) -> Linear(64) -> ReLU -> Linear(32) -> ReLU -> Linear(16)]`
- **Decoder Structure:** `[Linear(16) -> ReLU -> Linear(32) -> ReLU -> Linear(64) -> Linear(198)]`
- **Training:** 30 Epochs, Adam Optimizer (LR=1e-3, weight_decay=1e-5), Batch Size=256
- **Thresholding Strategy:** Statically set to the 99th percentile of Mean Squared Error (MSE) computed **only on Normal training traffic**.

### Evaluation Results (Test Set)
- **Accuracy:** `63.66%`
- **Normal Traffic Recall:** `98.93%` (F1: `0.6686`)
- **Attack/Anomaly Recall:** `42.89%` (F1: `0.5977`)
- **Macro Average F1:** `0.6332`

### Diagnosis & Issues
**Critical Failure: High False Negatives.** The model completely missed 57.11% of attacks (13,533 out of 23,698 test attack samples were classified as Normal). 
1. **Underfitting:** The 16-dimensional latent space bottleneck was too constricting, causing a loss of critical variance. Normal and Attack distributions heavily overlapped.
2. **Suboptimal Thresholding:** Blindly choosing the 99th percentile of Normal traffic meant the system was structurally biased to allow 99% of normal traffic through, inadvertently allowing attacks that shared similar representation spaces to pass undetected.

---

## Version 1.1 (Improved Architecture & Thresholding)

### Architecture & Hyperparameters Updates
- **Expanded Capacity:** `[Input(198) -> Linear(128) -> Linear(64) -> Linear(32)]`
- **Regularization added:** Integrated `BatchNorm1d` to stabilize deep representation learning and prevent feature collapse. Swapped `ReLU` for `LeakyReLU(0.1)` to resolve dying neurons.
- **Robust Training:** 
  - Increased maximum Epochs to `100`.
  - Added `ReduceLROnPlateau` scheduler (Factor 0.5, Patience 5).
  - Implemented `Early Stopping` (Patience 15) tracking the Validation MSE.
  - Increased Weight Decay to `1e-4` to compensate for the larger capacity.

### Algorithmic Updates
- **Dynamic F1-Optimized Threshold:** The threshold extraction logic was entirely rewritten. Instead of looking solely at normal traffic, the system now evaluates MSE on the **Mixed Validation Set** (containing both normal and attack traffic). It actively scans the error distribution and selects the exact threshold that maximizes the overall **F1-Score**.

### Impact & Unintended Consequences (Evaluation Results)
- **The "Return True" Failure:** During evaluation, the model classified **100% of the test traffic** as `Anomaly/Attack` (False Positives jumped to 100%, False Negatives dropped to 0%).
- **Why it failed:** The dynamic F1-optimizer blindly optimized for the Attack class. Because the validation set contained more attacks than normal traffic (23,698 vs 13,950), the optimizer determined that setting the threshold to near-zero (`0.000089`) to classify *everything* as an attack yielded the highest raw Attack F1-score (`0.7726`). 
- **Verdict:** Version 1.1 is critically broken and unusable. The threshold logic must be reverted or redesigned to prioritize Normal traffic retention (e.g., maximizing Macro F1 or enforcing a strict False Positive upper bound).

---

## Version 1.2 (PyTorch Lightning Refactor & Bug Discovery)

### Architecture & Hyperparameters Updates
- **Framework:** Refactored training pipeline to **PyTorch Lightning** using `AnomalyAutoencoderLightning` for structural safety, better logging, and reproducibility.
- **Model Checkpointing:** Saved best weights monitoring Validation Loss.
- **Optimizer:** Standardized on `AdamW` (learning rate `1e-3`, weight decay `1e-4`) with a `ReduceLROnPlateau` scheduler (factor `0.5`, patience `5`).

### Evaluation Results (With Saved Threshold 3.3887e-05)
- **Normal Traffic Recall:** `0.00%` (F1: `0.0000`)
- **Attack/Anomaly Recall:** `100.00%` (F1: `0.7726`)
- **Macro Average F1:** `0.3863`
- **Verdict:** Unusable. The model still classifies 100% of traffic as attacks (100% False Positive Rate).

### Critical Diagnosis & Root Cause Analysis
1. **The Grid Search Outlier Bug:** The threshold selection code uses `np.linspace(min_err, max_err, 200)` to scan for the threshold maximizing Macro F1. Because the validation set has an extreme reconstruction error outlier (`max_err = 56.899`), the step size of the search is `~0.284`. The search checks the first threshold (`0.000034`), which predicts everything as an attack (Macro F1 = 0.3863), and then immediately jumps to `0.284`. Since `0.284` is higher than almost all errors, the model predicts everything as normal, yielding a worse Macro F1. As a result, the optimizer blindly selected the very first threshold value, causing the 100% False Positive rate.
2. **Architecture Mismatch:** There is a minor layer discrepancy where `AnomalyAutoencoderLightning` has no activation after the final encoder projection, but `AnomalyAutoencoder` does (`LeakyReLU(0.1)`). While this loads without errors (since LeakyReLU has no weights), it alters the forward pass representations.
3. **Severe Distribution Overlap:** Even if the grid-search bug is fixed (using a finer search space focusing on the lower density region, yielding an optimal threshold of `~0.1156`), the model achieves:
   - **Test Macro F1:** `0.4400`
   - **Test Normal Recall:** `87.96%`
   - **Test Attack Recall:** `21.37%` (misses 78.6% of attacks).
   
   This occurs because the reconstruction error distributions for Normal and Attack traffic overlap almost completely (Normal Median: `0.0477`, Attack Median: `0.1165`; Normal 90th percentile: `0.1625`, Attack 90th percentile: `0.1921`). Unsupervised reconstruction error magnitude alone is insufficient to distinguish attacks from normal traffic on this feature set.

---

## Version 1.3 (RobustScaler Outlier Bug & Evaluation)

### Architecture & Hyperparameters Updates
- **Preprocessing Shift**: Replaced `StandardScaler` with `RobustScaler` to handle extreme outliers in the UNSW-NB15 dataset.
- **Model Capacity Constraint**: Decreased `latent_dim` from `32` to `8` to enforce a stronger bottleneck, preventing the model from acting as an identity function. Added `L1` sparsity penalty to latent encodings.
- **Strict 95th Percentile Thresholding**: Scrapped F1-optimization logic. Threshold is strictly set to the 95th percentile of normal traffic reconstruction MSE.

### Evaluation Results (Validation Set)
- **Normal Traffic Recall:** `95.00%` (Fixed by definition)
- **Attack/Anomaly Recall:** `1.23%` (Missed 98.7% of attacks)
- **Macro Average F1:** `0.2736`
- **Optimal Anomaly Threshold:** `50386.2`

### Critical Diagnosis & Root Cause Analysis
1. **The IQR=0 Fallback Scaling Bug**: The extreme threshold (`50386.2`) and terrible performance were caused by `RobustScaler`. For highly skewed features consisting mostly of zeros (like `response_body_len`), the Interquartile Range (IQR) is `0.0`. `RobustScaler` falls back to dividing by `1.0` in this case. As a result, `response_body_len` remained completely unscaled, retaining raw values up to **1.09 million**.
2. **MSE Domination**: A prediction error on this single unscaled feature yielded a squared error of `~10^12`, dominating the MSE loss (pushing sample average MSE to `~3.9 million`). The network focused 100% of its capacity on reconstructing these massive unscaled features, completely ignoring the other 197 features. Consequently, any attack that didn't have an enormously long response payload had a tiny MSE (like `1.0`) which fell vastly below the `50386.2` threshold, classifying it as Normal traffic.

### Solution moving forward
Replace `RobustScaler` with `QuantileTransformer(output_distribution='normal')` to map all numerical features strictly to a standard normal Gaussian `N(0, 1)`, strictly clipping the impact of extreme outliers and ensuring no single feature dominates the reconstruction loss space. Update the evaluation script to strictly use `AnomalyAutoencoderLightning(latent_dim=8)`.

---

## Version 1.4 (QuantileTransformer & TF32 Optimization)

### Architecture & Hyperparameters Updates
- **Preprocessing Shift**: Swapped the problematic `RobustScaler` with `QuantileTransformer(output_distribution='normal')`. All features are now strictly Gaussian N(0, 1), bounded roughly between `[-5.19, 5.19]`. This completely neutralizes the extreme zero-skew in features like `response_body_len`.
- **Hardware Acceleration**: Enabled Tensor Cores (TF32 precision) using `torch.set_float32_matmul_precision('medium')`.
- **Training Optimization**: Increased batch size from 256 to 2048 to prevent massive CPU overhead, and added persistent dataloader workers.

### Evaluation Results (Test Set)
- **Normal Traffic Recall:** `94.95%` (F1: `0.5932`)
- **Attack/Anomaly Recall:** `26.33%` (F1: `0.4072`)
- **Macro Average F1:** `0.5002`
- **Optimal Anomaly Threshold:** `0.0499`

### Conclusion & Next Steps
The massive million-loss bug is completely resolved. The model is now perfectly scaling features and training stably (loss converging nicely to `0.05`). 
However, **Attack Recall is capped at 26.33%**. This proves a structural limitation: on the complex UNSW-NB15 dataset, standard Unsupervised Reconstruction Error (MSE) from a Dense Autoencoder is simply not discriminative enough to cleanly separate Attack traffic from Normal traffic, even with perfect scaling. 
To improve unsupervised performance, we must leverage temporal dependencies (switching to an LSTM Autoencoder) or switch to a Variational Autoencoder (VAE).

---

## Version 1.5 (Temporal LSTM Autoencoder Baseline)

### Architecture & Hyperparameters Updates
- **Framework & Model:** Implemented `TemporalLSTMAutoencoderLightning` mapping sequences of shape `(Batch, SeqLen, InputDim)` into a central bottleneck using an Encoder LSTM and Decoder LSTM.
- **Parameters:** `seq_len=10`, `latent_dim=64`, trained strictly on normal traffic.
- **Thresholding Strategy:** Set to the 95th percentile of validation MSE on normal traffic.

### Evaluation Results (Test Set)
- **Normal Traffic Recall:** `95.43%`
- **Attack/Anomaly Recall:** `8.71%` (F1: `0.1581`)
- **Macro Average F1:** `0.3831`
- **Optimal Anomaly Threshold:** `0.0499`

### Diagnosis & Issues
- **Severe Over-generalization:** The Attack Recall dropped significantly to `8.71%` compared to the Dense Autoencoder's peak of `26.33%`. Because the LSTM network with `latent_dim=64` has extremely high capacity, it successfully learns to reconstruct unseen attack sequences perfectly, keeping their reconstruction MSE low and rendering them indistinguishable from normal traffic.

---

## Version 1.6 (Choked Bottleneck LSTM Autoencoder)

### Architecture & Hyperparameters Updates
- **Information Bottleneck:** Drastically reduced `latent_dim` from `64` to `8` to limit the model's capacity and prevent it from learning identity/easy reconstructions.
- **Regularization:** Added `num_layers=2` for the LSTM layers and introduced a `dropout=0.2` layer to prevent the model from memorizing localized features.
- **Optimal Anomaly Threshold:** `1.658595`

### Evaluation Results (Test Set)
- **Normal Traffic Recall:** `94.70%` (F1: `0.1205`)
- **Attack/Anomaly Recall:** `4.00%` (F1: `0.0766`)
- **Macro Average F1:** `0.0986`

### Diagnosis & Issues
- **Bottleneck Strategy Failure:** Tightening the bottleneck and adding regularization had the opposite effect, dropping the Attack Recall further to `4.00%`. With a tighter bottleneck, the model likely learns to output a highly generalized average representation, resulting in a moderate reconstruction error for *both* normal and attack sequences, failing to create discriminative error boundaries.
- **Verdict:** Unsupervised sequence reconstruction error alone is insufficient here. We need professional deep learning guidelines (`skill.md` resources) on advanced sequence anomaly detection architectures (like VAEs, GANs, or customized temporal modeling).

---

## Version 1.7 (Denoising Sparse LSTM Autoencoder)

### Architecture & Hyperparameters Updates
- **Denoising Augmentation:** Added Gaussian Noise (`0.05`) and Random Masking (`0.15`) during the training step. The model receives corrupted sequences but computes MSE loss against the clean original sequences.
- **Sparsity Penalty:** Added an L1 penalty (`1e-5`) to the bottleneck representations to prevent the model from using all latent capacity.
- **Capacity:** Reverted `latent_dim` from `8` back to `32`, `hidden_size` to `64`.
- **Thresholding Strategy:** `find_optimal_threshold` scanning for balanced F1 with a minimum constraint of 90% Normal Recall.

### Evaluation Results (Validation Set)
- **Normal Traffic Recall:** `90.10%`
- **Attack/Anomaly Recall:** `14.50%` (F1: `0.246`)
- **Optimal Anomaly Threshold:** `0.721220`

### Diagnosis & Issues
- **Persistent Over-generalization:** While the Attack Recall slightly recovered from 4.00% to 14.50%, the network is still drastically over-generalizing. It perfectly reconstructs over 85% of anomalous attacks.
- **Why it failed:** Even with dropout, Gaussian noise, feature masking, and L1 sparsity, a standard continuous latent space inside an LSTM has enough interpolative capacity to map unseen attack sequences back to themselves. We need a fundamental architectural shift to break this interpolative capacity (e.g., discretizing the latent space via Memory-Augmented Autoencoders or using completely different modalities).

---

## Version 1.8 (Memory-Augmented LSTM Autoencoder - MemAE Baseline)

### Architecture & Hyperparameters Updates
- **Discrete Memory Latent Constraint**: Integrated a `MemoryModule` with a discrete matrix $M \in \mathbb{R}^{50 \times 32}$ representing 50 normal prototypes. This maps the continuous latent representation $z$ to a sparse combination of prototypes using hard shrinkage (`shrink_thres = 0.0025`).
- **Sparsity Regularization**: Added an entropy penalty ($\lambda_{\text{entropy}} = 0.0002$) to the loss function to encourage sparse prototype selections.
- **Denoising**: Kept Denoising Augmentation (Gaussian Noise `0.05` + Masking `0.15`) active.
- **Dataloader Worker Update**: Set `num_workers=0` to resolve the Windows paging memory error (Error 1455).

### Evaluation Results (Validation Set)
- **Normal Traffic Recall:** `90.00%`
- **Attack/Anomaly Recall:** `15.40%` (F1: `0.258`)
- **Optimal Anomaly Threshold:** `1.199324`

### Diagnosis & Evaluation
- **Is this OK?** No. The Attack Recall is only $15.40\%$, meaning the model still fails to detect $84.6\%$ of attacks. The improvement over the denoising baseline (Version 1.7 at $14.50\%$) is negligible.
- **Analysis (Why it failed):** 
  1. **Lack of L2-Normalization in Similarity**: The query $z$ and memory items $M$ were not L2-normalized. Since they are computed from raw dot products, the similarity values were extremely small in magnitude. When passed to Softmax, it produced a flat, near-uniform attention distribution ($\approx 0.02$ per prototype).
  2. **Failed Hard Shrinkage**: Because the attention distribution was near-uniform, almost all 50 weights were above the `shrink_thres = 0.0025`. Thus, no shrinkage/pruning occurred, and the decoder reconstructed sequences using a dense combination of all 50 prototypes, defeating the purpose of a memory-augmented bottleneck.
  3. **High Attention Entropy**: The entropy loss stabilized at a high value of `2.402` (expected entropy of a uniform distribution over 50 prototypes is $\ln(50) \approx 3.912$, meaning it was utilizing an average of $e^{2.402} \approx 11$ prototypes).
- **Corrective Actions**:
  - Implement L2-normalization of query and memory matrices (cosine similarity) and introduce a scaling multiplier ($\alpha = 15.0$) to make attention peaky.
  - Constrain model capacity further by reducing memory dimensions (`mem_dim = 30`), increasing shrinkage threshold (`shrink_thres = 0.005`), and scaling up entropy regularization ($\lambda_{\text{entropy}} = 0.001$).

---

## Version 1.9 (Temporal One-Class VAE - TemporalOneClassVAE)

### Architecture & Hyperparameters Updates
- **Transitioned to a Probabilistic One-Class VAE**: Replaced the Memory-Augmented LSTM Autoencoder with a new architecture: `TemporalOneClassVAE`.
- **Transformer Encoder**: Processes the $(10, 198)$ sequences using a Transformer Encoder layer (2 layers, 4 heads, $d_{\text{model}} = 128$) to project them into a $16$-dimensional latent space distribution ($\mu, \sigma$).
- **One-Class Latent Clustering**: Introduced a latent clustering loss (weight $= 0.2$) that pulls benign latent representation vectors $\mu$ toward a registered center (`latent_center`) initialized from the mean latent space of benign training traffic.
- **Probabilistic Bottleneck (KL Divergence)**: Added a KL Divergence loss (maximum weight $\beta = 0.02$, with a 20-epoch warmup) to enforce a smooth Gaussian latent prior.
- **MLP Decoder**: Reconstructs inputs using a simple 2-layer MLP (hidden dimension $= 64$) combined with static sequence position/time embeddings.
- **Composite Anomaly Score**: Computes the final anomaly score as a weighted sum of normalized components:
  $$\text{Score} = 0.45 \times \text{Reconstruction MSE} + 0.35 \times \text{Latent Center Distance} + 0.20 \times \text{KL Surprise}$$
- **Training Strategy**: Trained strictly on normal traffic for 100 epochs using `AdamW` (learning rate $= 5\times 10^{-4}$, weight decay $= 10^{-4}$).

### Evaluation Results (Validation Set)
- **Normal Traffic Recall:** `90.30%` (satisfies the $\ge 90\%$ constraint)
- **Attack/Anomaly Recall:** `40.90%` (F1: `0.577`)
- **Optimal Anomaly Threshold:** `0.671306`

### Diagnosis & Evaluation
- **Is this OK?** It is a **major step forward** (raising Attack Recall from $15.4\%$ to $40.9\%$), but it is **not yet fully satisfying**. We are still missing $59.1\%$ of attacks.
- **Root Cause Analysis (Why it is underperforming)**:
  1. **Severe Decoder Underfitting (High Reconstruction Loss)**: During training, both the train and validation reconstruction loss converged to $\approx 1.03-1.04$. Because the features are normalized using `QuantileTransformer` to $\mathcal{N}(0, 1)$ standard normals, an MSE of $\approx 1.0$ is the mathematical equivalent of predicting a constant mean. The decoder is completely failing to reconstruct any fine-grained temporal patterns.
  2. **Insufficient Decoder Capacity**: The decoder is a simple MLP with only $64$ hidden units trying to reconstruct $10 \times 198 = 1980$ dimensions from a tiny $16$-dimensional latent bottleneck. This is a severe information bottleneck.
  3. **Static Decoder Temporal Modeling**: The decoder merely repeats the latent code across timesteps and adds static positional embeddings. It has no recurrent/causal mechanisms to capture dynamic sequence transitions.
  4. **Why Attack Recall improved anyway**: The improvement is driven almost entirely by the **one-class latent distance** and **KL surprise** metrics (which penalize out-of-distribution patterns in latent space), but the reconstruction MSE component (which has a $45\%$ weight in the anomaly score) is acting as uninformative noise, diluting the detection signal.
- **Path Forward**:
  - **Option A (Boost Decoder Capacity)**: Set `decoder_hidden = 256` or `512` to give the decoder the required parameters to map the latent representation back to the 198 features.
  - **Option C (Adjust Score Weights)**: Reduce the reconstruction score weight from $0.45$ to $0.10$ or $0.15$, and increase the weight of the latent center distance and KL surprise components. Since the model detects anomalies primarily through the latent space, we should align the scoring function to weight the working components more heavily.

---

## Version 1.10 (Properly Trained TemporalOneClassVAE)

### Architecture & Hyperparameters Updates
- **Decoder Capacity**: Increased `decoder_hidden` from $64$ to $256$, and `decoder_layers` from $1$ to $2$.
- **Training Fix**: Decreased `batch_size` from $2048$ to $256$ and increased `epochs` to $200$. This fixed the severe gradient-starvation bug (which was giving the model only 6 updates per epoch).

### Evaluation Results (Validation Set)
- **Normal Traffic Recall:** `90.00%`
- **Attack/Anomaly Recall:** `19.40%` (F1: `0.323`)
- **Validation Recon Loss:** `0.531` (Down from 1.021)
- **Optimal Anomaly Threshold:** `0.603410`

### Retuning Attempt
We ran `retune_dl_autoencoder_threshold.py` to sweep candidate score weights:
- `latent_heavy (0.10, 0.60, 0.30)` -> Attack Recall: `19.1%`
- `balanced (0.15, 0.50, 0.35)` -> Attack Recall: `25.8%`
- `current (0.45, 0.35, 0.20)` -> Attack Recall: `27.9%`

### Diagnosis & Evaluation
- **Is this OK?** No. The model actually performed *worse* when it learned properly. Attack recall plummeted from 40.9% down to ~27.9% at best.
- **Root Cause Analysis (Why it failed)**:
  1. **The Over-Generalization Trap**: Now that the Transformer encoder and decoder have enough gradient steps and capacity, they successfully learned to compress and reconstruct the 198-dimensional sequences. However, because the normal traffic is highly diverse, the model learned a generalized manifold that *also perfectly reconstructs the attack sequences*.
  2. **Indistinguishable Distributions**: Our diagnostics show that the median reconstruction MSE for normal traffic is `0.532`, and for attack traffic, it is `0.588`. The latent center distance for normal is `0.0019` vs `0.0028` for attacks. The distributions are almost perfectly overlapping. 
- **Verdict & Next Steps**: We have formally hit the ceiling for purely unsupervised reconstruction-based anomaly detection on this feature set. Because attack features share the same subspace as normal features, an autoencoder cannot mathematically separate them without labels. 
  - **Do we need more optimization?** No more Autoencoder optimization. We need a fundamental shift in strategy.
  - **Proposed Shift**: Since we *have* binary labels (`y_binary`) in our training set, we should abandon the purely unsupervised Autoencoder for Stage 1, and instead train a **Supervised Sequence Binary Classifier** (e.g., a Transformer or LSTM that outputs a probability of anomaly). This will explicitly learn the boundary between Normal and Attack, easily exceeding 60% recall.

---

## Version 2.0 (Supervised Sequence Binary Classifier - Production Stage 1)

### Architecture & Hyperparameters Updates
- **Framework & Model:** `TemporalTransformerClassifier` mapping sequential windows of shape `(Batch, 10, 198)` to binary class logits.
- **Components:** 2 Transformer Encoder layers (4 attention heads, $d_{\text{model}} = 128$) followed by a classification head (Linear -> LayerNorm -> Dropout -> Output).
- **Optimizer:** `AdamW` (learning rate $= 5\times 10^{-4}$, weight decay $= 10^{-4}$), Focal Loss (gamma $= 1.5$) to handle class imbalance.
- **Training Strategy:** Supervised training using binary targets: `y_binary = (y != normal_label)` (where normal label is strictly checked as `4`).
- **Optimal Threshold Selection:** Swept anomaly probabilities on the validation split, selecting the threshold that maximizes attack recall while enforcing a strict constraint of $\ge 90\%$ normal recall.

### Evaluation Results (Validation / Test Sets)
- **Optimal Probability Threshold:** `0.207865`
- **Validation Split Metrics:**
  - **Normal Recall:** `92.86%` (Satisfies $\ge 90\%$ constraint)
  - **Attack Recall:** `99.89%`
  - **Accuracy:** `97.29%`
  - **F1 Score:** `0.9789`
- **Test Split Metrics:**
  - **Normal Recall:** `93.43%`
  - **Attack Recall:** `99.97%`
  - **Accuracy:** `97.55%`
  - **F1 Score:** `0.9809`

### Diagnosis & Evaluation
- **Is this OK?** Yes! This is a massive home run. The model meets and exceeds all stage 1 requirements (achieving near-perfect detection of attacks with minimal false positives).
- **Trade-off:** Purely supervised classification sacrifices the theoretical zero-day detection capabilities of unsupervised anomaly detection, but provides extremely high, production-ready accuracy on known attack vectors and mutations.

