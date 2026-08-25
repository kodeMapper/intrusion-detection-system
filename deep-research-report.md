# Improving Unsupervised Autoencoder Anomaly Detection

To improve the unsupervised autoencoder, address each failure mode systematically:

## 1. Ensure Architecture Consistency  
Verify that the **training** and **inference** autoencoder networks are identical. Even a non-parametric difference (like an extra LeakyReLU in the evaluation model) will change the internal representations and ruin consistency. In practice, load exactly the same encoder/decoder layers (with the same activations) for training and testing. Any mismatch means the model at inference does not match what was learned, causing unpredictable reconstruction errors.

## 2. Control Latent Capacity (Sparsity)  
An autoencoder must be constrained to learn only dominant normal patterns. If the latent “bottleneck” is too large, the network can **memorize the identity** ( $f(x)\approx x$) and reconstruct anomalies well. Conversely, too-small a latent space underfits. In practice, **reduce the latent dimension** (e.g. from 32 to 4–8 units) so the model cannot encode all details. Additionally add sparsity or regularization to limit capacity. For example:  
- **L1 or KL sparsity penalty:** Add a term in the loss that encourages most latent activations to be near zero. This forces the autoencoder to use only a few neurons for each normal input.  
- **Dropout or noise:** Apply dropout layers or input noise to prevent the network from relying on any one neuron. Combining sparsity with dropout yields more robust features.  

These capacity-control techniques force the autoencoder to capture only the strongest, common structures of normal data. In a sparse autoencoder, anomalies (which do not follow those exact patterns) typically yield much higher reconstruction errors. Notably, a recent analysis shows that **larger latent spaces tend to overfit**, while smaller ones underfit, so tuning this bottleneck is critical.

## 3. Use Robust Preprocessing  
Tabular traffic data often has extreme outliers (e.g. very large packet counts or durations). **Standard scaling (z-score)** will be dominated by these outliers, squeezing normal values into a narrow range. Instead, use **robust methods**:  
- **RobustScaler:** Centers on the median and scales by the interquartile range. This makes each feature’s scale insensitive to a few extreme values. After robust scaling, most normal data spreads evenly (e.g. into roughly [–2,3]) without being squashed by outliers.  
- **QuantileTransformer:** Applies a nonlinear transform to map each feature’s distribution to uniform (or Gaussian). This “spreads out” the inliers across the range [0,1] and effectively clips extreme values to the boundaries. As a result, the overlap between normal and attack features is reduced and outliers no longer dominate the feature space.  

In short, replace StandardScaler with robust scaling or quantile transforms so that normal-vs-attack differences become more apparent and the autoencoder sees a balanced input distribution.

## 4. Consider Deep One-Class (SVDD) Models  
Instead of a reconstruction objective, an alternative unsupervised approach is **Deep SVDD** (deep support vector data description). Here the network is trained to map *normal* data into a compact hypersphere in latent space. Formally, the loss minimizes the volume of the sphere enclosing the training set. At test time, any point that falls outside this learned hypersphere is flagged as anomalous. This directly enforces “compactness” of the normal data representation, so the model does not trivially reproduce every input. Deep SVDD avoids the identity-mapping problem of autoencoders: it does not try to reconstruct inputs, but only to enclose normal data tightly. (If even a few labeled anomalies are available, **DeepSAD** adds them by pushing known anomalies away from the center, but pure Deep SVDD requires no anomaly labels.)  

## 5. Percentile-Based Thresholding  
Finally, set the anomaly threshold using only the *normal* error distribution. For example, choose a high percentile (95th, 99th, etc.) of the normal reconstruction errors as the cutoff. This guarantees a controlled false-positive rate: e.g. the 95th percentile means at most 5% of normal examples exceed the threshold. In practice, this is done by running the model on a held-out normal-validation set and computing `threshold = percentile(errors_normal, p)`. Any sample with error above this value is labeled anomalous. This method does not use any attack labels or F1 optimization, so it remains fully unsupervised. It explicitly **fixes the acceptable false-positive rate**, leaving maximum flexibility to catch attacks. For instance, one deployment chose the 99.5th percentile of normal-frame errors to yield only ~0.5% false alarms in production.  

Setting the threshold this way avoids the “100% anomalies” bug and is robust to outliers by construction. (If the normal error distribution has heavy outliers, one can use an even lower percentile or trim those outliers first.) After thresholding, validate performance on a small holdout (if possible) to ensure the chosen percentile is reasonable, but do **not** use labeled anomalies to tune it if strict unsupervised operation is required.

By applying these steps – matching architectures, restricting capacity, scaling features robustly, optionally using a one-class loss, and thresholding by percentile – the autoencoder’s ability to discriminate normal vs anomalous data can improve substantially **without any supervised labels**. Each technique addresses a specific failure mode in the training journey. Together they provide a principled, unsupervised way to enhance anomaly detection performance.

**Sources:** Standard preprocessing reference; sparse autoencoder principles; deep one-class SVDD theory; and unsupervised thresholding examples.