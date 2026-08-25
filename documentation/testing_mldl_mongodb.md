# Unified ML & DL Integration Testing with MongoDB / CSV Fallback

This document explains the end-to-end integration and testing guide for the unified Intrusion Detection and Prevention System (IDPS) engine. It brings together our **ML Baseline Models** (XGBoost, Random Forest, LightGBM) and our **DL Advanced Models** (Stage 1 Binary Gate, Stage 2 Multiclass Classifier, and the Unsupervised Autoencoder Zero-Day Canary) into a single, real-time prediction pipeline.

---

## 1. What This Setup Does

This setup simulates live network traffic by feeding records sequentially (one every second) from a data source into the unified ML/DL predictor engine. The predictions, individual engine details, and zero-day alerts are then visualized in real-time on the **Sentinel React Dashboard**.

You can run this setup in two modes:
1. **MongoDB Mode:** The API reads sample records from a MongoDB collection, simulating a live database collector.
2. **CSV Fallback Mode:** The API reads sample records directly from a local CSV file sequentially, removing the requirement to run a MongoDB instance.

---

## 2. Key Architectural Changes

Previously, the testing setup was ML-only. Integrating Deep Learning and Autoencoders required significant structural upgrades:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     End-to-End Testing Pipeline                          │
│                                                                          │
│   MongoDB (live_test_dataset)  ──OR──  CSV Fallback (direct file read)   │
│       │                                                                  │
│       ▼ (1 sample/sec)                                                   │
│   Express API Server (server.js) ─── orchestrator                        │
│       │                                                                  │
│       ▼ JSON stdin                                                       │
│   ┌──────────────────────────────────────────────────────────────┐       │
│   │  unified_predictor_worker.py                    [NEW]       │       │
│   │                                                              │       │
│   │  ┌─────────────────┐ ┌──────────────┐ ┌────────────────┐    │       │
│   │  │ ML Baseline Eng. │ │ DL Adv Eng.  │ │ AE Anomaly Eng │    │       │
│   │  │ (XGB+RF+LGBM)   │ │ (S1+S2)      │ │ (Zero-Day Flg) │    │       │
│   │  │ single-flow pred │ │ seq window   │ │ anomaly score  │    │       │
│   │  └───────┬─────────┘ └──────┬───────┘ └───────┬────────┘    │       │
│   │          │                  │                 │             │       │
│   │          └────────┬─────────┘                 │             │       │
│   │                   ▼                           │             │       │
│   │          Combined Verdict Logic ◄─────────────┘             │       │
│   │          (ML + DL labels + AE anomaly score)                │       │
│   └──────────────────────┬───────────────────────────────────────┘       │
│                          │ JSON stdout                                   │
│                          ▼                                               │
│   Express API → /alerts, /health, /stats, /metrics                       │
│                          │                                               │
│                          ▼                                               │
│   Sentinel Dashboard v2 (React)                                          │
│   - ML vs DL verdict comparison                                         │
│   - Per-engine confidence meters                                         │
│   - Zero-Day Anomaly Indicator (AE anomaly score)                        │
│   - Attack type breakdown chart                                          │
│   - Model Evaluation Metrics Panel (Train/Val/Test)                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.1 The Unified Prediction Worker (`unified_predictor_worker.py`)
* **Old Way:** The previous prediction worker (`live_predictor_worker.py`) only loaded and ran the three ML baseline models (XGBoost, Random Forest, LightGBM) on a single, isolated network flow record.
* **New Way:** A single, unified script loads **all** models (3 ML models, 2 supervised DL models, and 1 unsupervised Autoencoder model) at startup. It handles preprocessing for both ML (label encoding + feature selection) and DL (scaling + one-hot encoding).
* **Impact:** Drastically reduces runtime overhead by loading all parameters into memory once, ensuring consistent preprocessing pipelines, and serving as a single source of truth for alerts.

### 2.2 Sequence Windowing Buffer
* **Old Way:** Every incoming packet flow was predicted independently, which meant the system was blind to time-series patterns.
* **New Way:** A sliding window buffer of size 10 is maintained inside the predictor worker. Incoming flows are transformed and pushed to this buffer. The DL models run inference over the tensor of shape `(batch, 10, 198)`.
* **Impact:** 
  * Enables the advanced deep learning models (Transformer-based) to capture sequence-based and temporal behaviors (e.g., slow port scans, distributed attacks).
  * **Warmup Phase:** During the first 9 samples, the DL engine is warming up its buffer, during which only the ML engine predicts. The system automatically shifts to active dual-engine prediction starting from flow 10.

### 2.3 Dual-Path Preprocessing
* **Old Way:** Simple encoder mappings.
* **New Way:** The unified worker splits the incoming JSON. The ML path goes through the tabular encoders, while the DL path is fed into `DLPreprocessor` (utilizing a `QuantileTransformer` scaler and custom one-hot encoders).
* **Impact:** Prevents feature leakage and ensures that each model receives data formatted exactly as it was during training.

### 2.4 Zero-Day Anomaly Canary
* **Old Way:** No zero-day detection; only known attack categories were classified.
* **New Way:** An unsupervised Variational Autoencoder (`TemporalOneClassVAE`) runs in parallel. It calculates a reconstruction error score. If the error exceeds a calibrated threshold, a `zero_day_flag` is marked true.
* **Impact:** Acts as an early warning system for novel, unseen anomalies (canaries) that might otherwise slide past the supervised classifiers.

### 2.5 CSV Fallback Mode
* **Old Way:** Required a running MongoDB instance to test.
* **New Way:** Set `USE_CSV_FALLBACK=true` in `.env` and the Express server reads samples directly from the CSV file sequentially. No MongoDB dependency.
* **Impact:** Makes testing trivially easy on any machine without infrastructure setup.

### 2.6 Extended API Endpoints
* **`GET /stats`:** Returns real-time engine statistics — ML-only detections, DL-only detections, both-agreed detections, attack breakdown, zero-day count, engine agreement rate.
* **`GET /metrics`:** Returns static model evaluation metrics (loaded from JSON report files at startup) for Stage 1 Binary, Stage 2 Multiclass, and AE Anomaly models.

### 2.7 Sentinel Dashboard v2
* **Engine Comparison Panel:** Shows ML, DL, and AE results side-by-side with confidence bars and threat-level coloring.
* **Model Metrics Tab:** Displays Train/Val/Test performance metrics for all models in a tabular format.
* **Enhanced Telemetry Feed:** Each alert row shows ML badge, DL badge, AE score, agreement indicator, verdict source, and zero-day warnings.

---

## 3. Ensemble Voting: Is it Necessary or Good?

Inside our baseline ML engine, we *already* use **soft-voting ensemble** (averaging the prediction probabilities of XGBoost, Random Forest, and LightGBM). However, a natural question is: **Should we use ensemble voting between the ML baseline engine and the DL advanced engine?**

### The Short Answer: No, it is not recommended.
A traditional voting ensemble (like majority voting or averaging probabilities) between ML and DL would actually **hurt** our detection accuracy. Here is why:

1. **Different Detection Philosophies (Stateless vs. Stateful):**
   * **ML Baseline** is stateless and looks at a *single* flow. It is highly sensitive to immediate, high-volume, structural characteristics of a packet.
   * **DL Advanced** is stateful and looks at a *sequence* of 10 flows. It is designed to capture slow, low-intensity, or multi-step behavior over time.
2. **The "Voted Down" Threat (Lower Recall):**
   * If a slow, stealthy attack occurs, the DL model will detect the sequence pattern and sound an alert. However, each individual flow in that sequence might look completely normal when viewed in isolation. Consequently, the stateless ML model will predict "Normal" with high confidence.
   * If we used a voting ensemble, the ML model's "Normal" vote would average out or override the DL model's "Attack" vote. The stealthy attack would bypass the system.

### Our Solution: Combined Verdict (OR-Gate / Union Logic)
Instead of voting, we use a **Combined Verdict Logic** that aggregates the strengths of both engines:

* **BENIGN Verdict:** We only classify a sample as `Normal` if **both** ML and DL engines agree it is Normal.
* **ATTACK Verdict:** If **either** engine detects an attack, we trigger an alert.
* **RESOLVING DISAGREEMENTS:**
  * If the ML engine detects an attack but the DL engine says "Normal" (or is warming up), we trust the ML engine (stateless detection).
  * If the DL engine detects an attack but the ML engine says "Normal", we trust the DL engine (temporal detection).
  * If **both** detect an attack, we report the DL engine's classification (as the Transformer has a richer sequence-level view of the attack family) and boost the alert's confidence.

---

## 4. How We Handle Zero-Day Attacks

Supervised models (both ML and DL) are excellent at recognizing known attack types (like DoS, Exploits, Fuzzers) and their close mutations. However, they struggle to classify a completely new attack style (a zero-day).

We handle zero-day scenarios using a two-pronged strategy:

1. **Behavioral Generalization (Primary Defense):**
   * Our Deep Learning model (Stage 1 Binary Gate) is trained on temporal behavior (flow counts, byte ratios, and timing sequences) rather than static, hardcoded signature rules. 
   * Most real-world "zero-day" attacks are variations of existing techniques (e.g., a new style of denial of service). Because their temporal footprint looks suspicious, they will still trigger the Stage 1 Binary Gate.
2. **Autoencoder Anomaly Canary (Supplementary Defense):**
   * The unsupervised `TemporalOneClassVAE` is trained **exclusively** on normal, benign traffic. It learns the "normal rhythm" of the network.
   * When traffic deviates significantly from this rhythm, the reconstruction error shoots up.
   * If the supervised models say "Normal" but the Autoencoder's reconstruction error is high, the system flags a **Zero-Day Warning** (`zero_day_flag = true`).
   * This mimics modern enterprise network sensors (like **Darktrace** or **CrowdStrike**), which run supervised detection for efficiency and known targets alongside unsupervised anomaly metrics for zero-day identification.

---

## 5. Files in the Pipeline

| File | Status | Purpose |
| :--- | :---: | :--- |
| `service/models/src/unified_predictor_worker.py` | **NEW** | Core unified prediction worker (ML + DL + AE engines) |
| `service/models/src/live_predictor_worker.py` | KEPT | Legacy ML-only worker (kept as fallback) |
| `service/api/src/server.js` | MODIFIED | Extended alert payload + `/stats` + `/metrics` + CSV fallback |
| `service/api/.env.example` | MODIFIED | New `USE_CSV_FALLBACK` variable + updated defaults |
| `service/dashboard/src/App.jsx` | MODIFIED | Fetches /stats + /metrics, tab bar for views |
| `service/dashboard/src/components/StatsWidget.jsx` | MODIFIED | ML vs DL engine stats + zero-day count |
| `service/dashboard/src/components/TelemetryFeed.jsx` | MODIFIED | Multi-engine alert rows with badges |
| `service/dashboard/src/components/EngineComparison.jsx` | **NEW** | Visual ML vs DL vs AE comparison panel |
| `service/dashboard/src/components/ModelMetrics.jsx` | **NEW** | Model evaluation metrics panel (Train/Val/Test) |
| `service/dashboard/src/index.css` | MODIFIED | Comprehensive new styles for all components |
| `service/preproc/dl_preprocessor.py` | KEPT | Used by unified worker for DL preprocessing |
| `service/collector/mongoDB_csvCreate.py` | KEPT | CSV generator already includes attack_cat + label |

---

## 6. Model Artifacts Used

| Artifact File | Engine | Purpose |
| :--- | :--- | :--- |
| `final_xgb.pkl` | ML Baseline | XGBoost classifier |
| `final_rf.pkl` | ML Baseline | Random Forest classifier |
| `final_lgbm.pkl` | ML Baseline | LightGBM classifier |
| `feature_selector.pkl` | ML Baseline | Feature selection transform |
| `final_encoders.pkl` | ML Baseline | Label encoders for categorical features |
| `final_labels.pkl` | ML Baseline | Label encoder for class names |
| `dl_stage1_binary_v1.0.pt` | DL Advanced | Stage 1 Binary Attack Gate weights |
| `dl_stage1_binary_threshold_v1.0.json` | DL Advanced | Stage 1 optimal probability threshold |
| `dl_stage1_binary_report_v1.0.json` | DL Advanced | Stage 1 evaluation metrics (dashboard display) |
| `dl_stage2_multiclass_v1.0.pt` | DL Advanced | Stage 2 Multiclass Classifier weights |
| `dl_stage2_multiclass_config_v1.0.json` | DL Advanced | Stage 2 label map + confidence policy |
| `dl_stage2_multiclass_report_v1.0.json` | DL Advanced | Stage 2 evaluation metrics (dashboard display) |
| `dl_ae_v1.0.pt` | AE Anomaly Sensor | Autoencoder weights (zero-day canary) |
| `dl_ae_threshold_v1.0.json` | AE Anomaly Sensor | AE calibration + anomaly threshold |
| `scaler_unsw_v1_20260612.pkl` | DL Preprocessor | QuantileTransformer scaler |
| `onehot_encoder_unsw_v1_20260612.pkl` | DL Preprocessor | OneHot encoder for categorical features |
| `preprocessing_config_unsw_v1_20260612.json` | DL Preprocessor | Config (dropped cols, label map, etc.) |
| `feature_list_unsw_v1_20260612.json` | DL Preprocessor | Ordered feature column list (198 features) |

---

## 7. End-to-End Data Flow

Here is how data flows through the system during a test run:

1. **Feeding Traffic:** The Express API server reads one network flow record per second, either from MongoDB or sequentially from `live_test_dataset.csv`.
2. **Streaming to Predictor:** The API pipes this record via `stdin` (JSON format) to the running Python `unified_predictor_worker.py`.
3. **ML Prediction Path:** 
   * The record is encoded and features are selected.
   * XGBoost, RF, and LightGBM make individual predictions.
   * Their probabilities are averaged (soft-voting ensemble) to produce `ml_prediction` and `ml_confidence`.
4. **DL & AE Prediction Path:**
   * The record is scaled and one-hot encoded using `DLPreprocessor`.
   * The flow is appended to the 10-flow sequence buffer.
   * If the buffer has fewer than 10 flows, DL returns `"warming_up"`.
   * If the buffer is full, the sequence tensor is fed into:
     * **DL Stage 1 (Binary Gate):** Determines if the sequence is an attack. If probability exceeds the tuned threshold (~0.218), it passes to **DL Stage 2**.
     * **DL Stage 2 (Classifier):** Determines the specific attack category and confidence.
     * **Autoencoder (VAE):** Measures the reconstruction error. If it exceeds the optimal anomaly threshold (~0.603), it sets `zero_day_flag = true`.
5. **Verdict Merger:** The worker combines the individual results using the **Combined Verdict Logic** (OR-gate) and outputs the result via `stdout` (JSON format) to the Express server.
6. **Dashboard Telemetry:** The Express server updates its `/alerts` feed, increments runtime `/stats`, and the Sentinel Dashboard UI reflects changes instantly.

---

## 8. How to Run and Test (Step-by-Step)

### Prerequisites

Make sure you have:
- Python 3.10+ with packages: `torch`, `joblib`, `numpy`, `pandas`, `scikit-learn`, `xgboost`, `lightgbm`, `lightning`
- Node.js 18+ with `npm`
- All model artifacts in `service/models/artifacts/` (see Section 6)
- Test CSV at `tests/live_test_dataset.csv` (generate with Section 8.1 if missing)

### 8.1 Generate Balanced Test CSV (if needed)

```powershell
cd "c:\Users\acer\Desktop\Linux Shared Folder\Project Repo"
python service\collector\mongoDB_csvCreate.py
```

This creates `tests/live_test_dataset.csv` with 1000 samples (80% Normal, 20% Attack).

### 8.2 Configure Environment

Create or edit `service/api/.env`:

```env
PORT=3001
MONGODB_URI=your_mongodb_connection_string
MONGODB_DB_NAME=IDPS
MONGODB_COLLECTION=live_test_dataset
CSV_PATH=
POLL_INTERVAL_MS=1000
PYTHON_CMD=python
PREDICTOR_SCRIPT=
RELOAD_CSV=false
USE_CSV_FALLBACK=true
```

**Key settings:**
- Set `USE_CSV_FALLBACK=true` to test locally without MongoDB.
- Leave `PREDICTOR_SCRIPT` blank — it defaults to `unified_predictor_worker.py`.
- Leave `CSV_PATH` blank — it defaults to `tests/live_test_dataset.csv`.

### 8.3 Install API Dependencies

```powershell
cd "c:\Users\acer\Desktop\Linux Shared Folder\Project Repo\service\api"
npm install
```

### 8.4 Start the API Server

```powershell
cd "c:\Users\acer\Desktop\Linux Shared Folder\Project Repo\service\api"
npm start
```

**What happens on startup:**
1. Model metrics are loaded from JSON report files.
2. CSV rows are loaded (if CSV fallback mode) or MongoDB is connected.
3. The unified Python predictor is spawned — it loads **all 6+ models** into memory. This takes **30-60 seconds** on first run. You will see `[predictor] [unified] Loading ML baseline models …` etc. in the console.
4. Once `[startup] Predictor ready.` appears, the system begins polling.
5. Alerts print to the console showing ML, DL, AE scores and zero-day flags.

### 8.5 Start the Sentinel Dashboard

In a **new terminal window**:

```powershell
cd "c:\Users\acer\Desktop\Linux Shared Folder\Project Repo\service\dashboard"
npm install
npm run dev
```

Open **http://localhost:5173** in your browser. You will see:
- Real-time alerts with ML/DL engine badges and agreement indicators.
- **Engine Comparison Panel** in the sidebar showing ML vs DL vs AE results for the latest detection.
- **Detection Source** stats showing how many alerts came from ML only, DL only, or both engines.
- **Zero-Day Flags** counter showing how many anomalous patterns the AE canary flagged.
- Click the **Model Metrics** tab to see Train/Val/Test evaluation metrics for all models.

### 8.6 Quick API Health Check

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:3001/health"
Invoke-RestMethod -Uri "http://127.0.0.1:3001/stats"
Invoke-RestMethod -Uri "http://127.0.0.1:3001/metrics"
Invoke-RestMethod -Uri "http://127.0.0.1:3001/alerts"
Invoke-RestMethod -Uri "http://127.0.0.1:3001/poll-once" -Method Post
```

---

## 9. API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| GET | `/health` | Server uptime, cursor position, sample count, mode (CSV/MongoDB), DL warmup status |
| GET | `/alerts` | Array of latest 200 alert objects with all engine fields |
| POST | `/poll-once` | Manually trigger one prediction cycle |
| GET | `/config` | Current server configuration |
| GET | `/stats` | Runtime statistics: detection sources, attack breakdown, zero-day count, agreement rate |
| GET | `/metrics` | Static model evaluation metrics from training reports |

### Sample Alert Object (from `/alerts`)

```json
{
  "sampleIndex": 42,
  "prediction": "DoS",
  "confidence": 0.94,
  "ml_prediction": "DoS",
  "ml_confidence": 0.91,
  "dl_prediction": "DoS",
  "dl_confidence": 0.97,
  "dl_stage1_attack_prob": 0.99,
  "ae_anomaly_score": 0.42,
  "zero_day_flag": false,
  "engine_agreement": true,
  "verdict_source": "both",
  "detectedAt": "2026-06-16T12:00:01.000Z"
}
```

---

## 10. Common Troubleshooting

### Server starts but predictor times out
- The unified worker loads ~500MB+ of ML models (RF alone is 400MB+). Allow up to 120 seconds.
- Check that all artifacts exist in `service/models/artifacts/`.

### DL says "warming_up" for all samples
- This is expected for the first 9 samples. The DL engine requires 10 flows in its sliding window.
- After sample 10, DL predictions become active.

### Too many alerts
- Verify CSV ratio: `tests/live_test_dataset.csv` should have 80% Normal, 20% Attack.
- Regenerate with: `python service\collector\mongoDB_csvCreate.py`

### MongoDB connection error
- Check `MONGODB_URI` in `.env`.
- Or set `USE_CSV_FALLBACK=true` to bypass MongoDB entirely.

### Python import errors
- Ensure your Python environment has: `torch`, `lightning`, `joblib`, `xgboost`, `lightgbm`, `scikit-learn`, `pandas`, `numpy`.
- The script expects to be run from the project root as working directory.

### Zero-day flags are too frequent
- The AE anomaly threshold was calibrated on training data. If many test samples trigger it, the AE model may need retraining or threshold adjustment.
- Zero-day flags are **advisory only** — they do not change the combined verdict.

---

## 11. Architecture Summary

```
Express server = orchestrator
  - Handles MongoDB/CSV polling and API routes
  - Spawns Python worker as a child process
  - Passes each sample via stdin → receives JSON via stdout

Python unified_predictor_worker.py = model executor
  - Loads all models at startup (ML + DL + AE)
  - ML path: tabular preprocessing → XGB+RF+LGBM ensemble
  - DL path: DLPreprocessor → sequence buffer → Stage 1 → Stage 2
  - AE path: uses same DL tensor → anomaly score + zero-day flag
  - Combined verdict: OR-gate logic merging ML + DL results

MongoDB / CSV = live stream source
  - Stores CSV rows as test traffic stream
  - _sampleIndex controls sequential playback

React Sentinel Dashboard v2 = User Interface
  - Connects to /health, /alerts, /stats, /metrics endpoints
  - Shows ML vs DL engine comparison in real-time
  - Displays model evaluation metrics (Train/Val/Test)
  - Highlights zero-day anomaly warnings
```
