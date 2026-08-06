"""
Unified Prediction Worker — ML Baseline + DL Advanced + AE Zero-Day Canary.

Reads JSON samples from stdin (one per line, sent by the Express API server),
runs all engines in parallel, merges verdicts, and writes JSON results to stdout.

Replaces live_predictor_worker.py as the PREDICTOR_SCRIPT in .env.
"""

import json
import sys
import warnings
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings(
    "ignore",
    message="X has feature names, but SelectFromModel was fitted without feature names",
)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)

# ── paths ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "service" / "models" / "artifacts"
PREPROC_DIR = ROOT / "data" / "artifacts"

# Columns that are metadata / labels — not features
ML_DROP_COLUMNS = [
    "_id", "_sampleIndex", "srcip", "dstip", "id",
    "Stime", "Ltime", "attack_cat", "label",
]

# ── helpers ─────────────────────────────────────────────────────────────

def write_message(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _log(msg: str) -> None:
    """Write debug info to stderr so it doesn't interfere with JSON protocol."""
    sys.stderr.write(f"[unified] {msg}\n")
    sys.stderr.flush()


# ========================================================================
#  ML Baseline Engine
# ========================================================================

class MLBaselineEngine:
    """Soft-voting ensemble of XGBoost + Random Forest + LightGBM."""

    def __init__(self):
        _log("Loading ML baseline models …")
        self.xgb = joblib.load(MODEL_DIR / "final_xgb.pkl")
        self.rf = joblib.load(MODEL_DIR / "final_rf.pkl")
        self.lgbm = joblib.load(MODEL_DIR / "final_lgbm.pkl")
        self.selector = joblib.load(MODEL_DIR / "feature_selector.pkl")
        self.encoders = joblib.load(MODEL_DIR / "final_encoders.pkl")
        self.label_encoder = joblib.load(MODEL_DIR / "final_labels.pkl")
        self.feature_columns = list(getattr(self.selector, "feature_names_in_", []))
        self.categorical_columns = set(self.encoders.keys())
        _log("ML baseline ready.")

    def _preprocess(self, sample: dict) -> np.ndarray:
        features = pd.DataFrame([sample]).drop(columns=ML_DROP_COLUMNS, errors="ignore")

        if self.feature_columns:
            for col in self.feature_columns:
                if col not in features.columns:
                    features[col] = 0
            features = features[self.feature_columns]

        for col, enc in self.encoders.items():
            if col in features.columns:
                class_map = {v: i for i, v in enumerate(enc.classes_)}
                features[col] = features[col].astype(str).map(
                    lambda v, cm=class_map: cm.get(v, 0)
                )

        for col in features.columns:
            if col not in self.categorical_columns:
                features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0)

        return self.selector.transform(features.to_numpy())

    def predict(self, sample: dict) -> tuple[str, float]:
        transformed = self._preprocess(sample)
        probs = (
            self.xgb.predict_proba(transformed)
            + self.rf.predict_proba(transformed)
            + self.lgbm.predict_proba(transformed)
        ) / 3
        idx = int(np.argmax(probs, axis=1)[0])
        conf = float(np.max(probs, axis=1)[0])
        label = self.label_encoder.inverse_transform([idx])[0]
        return label, conf


# ========================================================================
#  DL Advanced Engine  (Stage 1 Binary Gate  +  Stage 2 Multiclass)
# ========================================================================

class DLAdvancedEngine:
    """Temporal Transformer — Stage 1 filters attacks, Stage 2 classifies type."""

    SEQ_LEN = 10

    def __init__(self, device: torch.device):
        self.device = device
        self.buffer: deque[np.ndarray] = deque(maxlen=self.SEQ_LEN)

        # ── preprocessor ────────────────────────────────────────────
        _log("Loading DL preprocessor …")
        sys.path.insert(0, str(ROOT / "service"))
        from preproc import DLPreprocessor

        preproc_artifacts = PREPROC_DIR
        if not (preproc_artifacts / "scaler_unsw_v1_20260612.pkl").exists():
            preproc_artifacts = MODEL_DIR  # fallback
        self.preprocessor = DLPreprocessor(preproc_artifacts)

        # ── Stage 1: Binary Gate ────────────────────────────────────
        _log("Loading DL Stage 1 binary model …")
        with open(MODEL_DIR / "dl_stage1_binary_threshold_v1.0.json") as f:
            s1_cfg = json.load(f)
        hp = s1_cfg["model_hparams"]
        from models.src.dl_pipeline import TemporalTransformerClassifier

        self.stage1 = TemporalTransformerClassifier(
            input_size=hp["input_size"],
            seq_len=hp["seq_len"],
            num_classes=hp["num_classes"],
            d_model=hp["d_model"],
            n_heads=hp["n_heads"],
            num_layers=hp["num_layers"],
            dim_feedforward=hp["dim_feedforward"],
            classifier_hidden=hp["classifier_hidden"],
            dropout=hp["dropout"],
        )
        s1_state = torch.load(MODEL_DIR / "dl_stage1_binary_v1.0.pt", map_location="cpu", weights_only=True)
        self.stage1.load_state_dict(s1_state)
        self.stage1.to(device).eval()
        self.s1_threshold = float(s1_cfg["attack_probability_threshold"])

        # ── Stage 2: Multiclass Classifier ──────────────────────────
        _log("Loading DL Stage 2 multiclass model …")
        with open(MODEL_DIR / "dl_stage2_multiclass_config_v1.0.json") as f:
            s2_cfg = json.load(f)
        hp2 = s2_cfg["model_hparams"]
        self.stage2 = TemporalTransformerClassifier(
            input_size=hp2["input_size"],
            seq_len=hp2["seq_len"],
            num_classes=hp2["num_classes"],
            d_model=hp2["d_model"],
            n_heads=hp2["n_heads"],
            num_layers=hp2["num_layers"],
            dim_feedforward=hp2["dim_feedforward"],
            classifier_hidden=hp2["classifier_hidden"],
            dropout=hp2["dropout"],
        )
        s2_state = torch.load(MODEL_DIR / "dl_stage2_multiclass_v1.0.pt", map_location="cpu", weights_only=True)
        self.stage2.load_state_dict(s2_state)
        self.stage2.to(device).eval()
        self.attack_classes = s2_cfg["attack_classes"]  # ["DoS", "Exploits", …]

        _log("DL advanced engine ready.")

    def push_flow(self, sample: dict) -> None:
        """Preprocess a raw sample and append it to the sliding window buffer."""
        df = pd.DataFrame([sample])
        vec = self.preprocessor.transform(df)  # shape (1, 198)
        self.buffer.append(vec[0])

    @property
    def is_warm(self) -> bool:
        return len(self.buffer) >= self.SEQ_LEN

    @torch.no_grad()
    def predict(self) -> dict:
        """Run Stage 1 → conditionally Stage 2 on the current window."""
        if not self.is_warm:
            return {
                "dl_prediction": "warming_up",
                "dl_confidence": 0.0,
                "dl_stage1_attack_prob": 0.0,
            }

        seq = np.stack(list(self.buffer), axis=0)  # (10, 198)
        tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1,10,198)

        # Stage 1 — binary
        s1_probs = self.stage1.predict_proba(tensor)  # (1,2)
        attack_prob = float(s1_probs[0, 1].item())

        if attack_prob < self.s1_threshold:
            return {
                "dl_prediction": "Normal",
                "dl_confidence": float(1.0 - attack_prob),
                "dl_stage1_attack_prob": attack_prob,
            }

        # Stage 2 — multiclass
        s2_probs = self.stage2.predict_proba(tensor)  # (1,5)
        cls_idx = int(torch.argmax(s2_probs, dim=1).item())
        cls_conf = float(s2_probs[0, cls_idx].item())
        cls_label = self.attack_classes[cls_idx]

        return {
            "dl_prediction": cls_label,
            "dl_confidence": cls_conf,
            "dl_stage1_attack_prob": attack_prob,
        }

    def get_tensor(self) -> torch.Tensor | None:
        """Return current window as tensor for AE engine reuse."""
        if not self.is_warm:
            return None
        seq = np.stack(list(self.buffer), axis=0)
        return torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)


# ========================================================================
#  AE Anomaly Canary Engine  (Zero-Day Sensor)
# ========================================================================

class AEAnomalyEngine:
    """TemporalOneClassVAE — unsupervised anomaly canary."""

    def __init__(self, device: torch.device):
        self.device = device

        _log("Loading AE anomaly model …")
        with open(MODEL_DIR / "dl_ae_threshold_v1.0.json") as f:
            ae_cfg = json.load(f)
        hp = ae_cfg["model_hparams"]

        from models.src.dl_pipeline import TemporalOneClassVAE

        self.model = TemporalOneClassVAE(
            input_size=hp["input_size"],
            seq_len=hp["seq_len"],
            d_model=hp["d_model"],
            latent_dim=hp["latent_dim"],
            n_heads=hp["n_heads"],
            num_layers=hp["num_layers"],
            dim_feedforward=hp["dim_feedforward"],
            decoder_hidden=hp["decoder_hidden"],
            decoder_layers=hp["decoder_layers"],
            dropout=hp["dropout"],
            score_weights=tuple(hp.get("score_weights", (0.1, 0.6, 0.3))),
        )
        state = torch.load(MODEL_DIR / "dl_ae_v1.0.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(device).eval()

        # Restore score calibration from threshold JSON
        cal = ae_cfg.get("score_calibration", {})
        if cal:
            self.model.score_median.copy_(torch.tensor(cal["median"], dtype=torch.float32))
            self.model.score_iqr.copy_(torch.tensor(cal["iqr"], dtype=torch.float32))
            self.model.is_score_calibrated.copy_(torch.tensor(True, dtype=torch.bool))

        # Restore score weights from threshold JSON
        sw = ae_cfg.get("score_weights")
        if sw:
            self.model.set_score_weights(sw)

        self.threshold = float(ae_cfg["E_thresh"])
        _log(f"AE anomaly engine ready (threshold={self.threshold:.4f}).")

    @torch.no_grad()
    def score(self, tensor: torch.Tensor) -> tuple[float, bool]:
        """Return (anomaly_score, zero_day_flag)."""
        sc = float(self.model.anomaly_score(tensor).item())
        return sc, sc >= self.threshold


# ========================================================================
#  Combined Verdict
# ========================================================================

def combined_verdict(
    ml_pred: str,
    ml_conf: float,
    dl_result: dict,
) -> dict:
    """
    Merge ML + DL predictions using OR-gate logic:
    - Both Normal → Normal
    - Either Attack → Attack (prefer DL label when both agree on Attack)
    """
    dl_pred = dl_result["dl_prediction"]
    dl_conf = dl_result["dl_confidence"]

    ml_is_normal = (ml_pred.upper() == "NORMAL")
    dl_is_normal = (dl_pred.upper() == "NORMAL") or (dl_pred == "warming_up")
    dl_warming = (dl_pred == "warming_up")

    if ml_is_normal and dl_is_normal:
        return {
            "prediction": "Normal",
            "confidence": ml_conf if dl_warming else max(ml_conf, dl_conf),
            "verdict_source": "ml" if dl_warming else "both",
            "engine_agreement": True,
        }

    if not ml_is_normal and (dl_is_normal or dl_warming):
        # ML sees attack, DL doesn't (or warming up)
        return {
            "prediction": ml_pred,
            "confidence": ml_conf,
            "verdict_source": "ml",
            "engine_agreement": dl_warming,  # agreement unknown during warmup
        }

    if ml_is_normal and not dl_is_normal:
        # DL sees attack, ML doesn't — trust DL (temporal detection)
        return {
            "prediction": dl_pred,
            "confidence": dl_conf,
            "verdict_source": "dl",
            "engine_agreement": False,
        }

    # Both see attack — use DL classification with boosted confidence
    return {
        "prediction": dl_pred,
        "confidence": min(1.0, (ml_conf + dl_conf) / 2 + 0.05),
        "verdict_source": "both",
        "engine_agreement": True,
    }


# ========================================================================
#  Main loop
# ========================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"Device: {device}")

    ml_engine = MLBaselineEngine()
    dl_engine = DLAdvancedEngine(device)
    ae_engine = AEAnomalyEngine(device)

    write_message({"type": "ready"})

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("requestId")
            sample = request.get("sample", {})

            # ── ML Engine ───────────────────────────────────────────
            ml_pred, ml_conf = ml_engine.predict(sample)

            # ── DL Engine (push flow + predict if warm) ─────────────
            dl_engine.push_flow(sample)
            dl_result = dl_engine.predict()

            # ── AE Engine (reuse DL tensor) ─────────────────────────
            ae_score = 0.0
            zero_day_flag = False
            dl_tensor = dl_engine.get_tensor()
            if dl_tensor is not None:
                ae_score, zero_day_flag = ae_engine.score(dl_tensor)

            # ── Verdict ─────────────────────────────────────────────
            verdict = combined_verdict(ml_pred, ml_conf, dl_result)

            write_message({
                "requestId": request_id,
                "prediction": verdict["prediction"],
                "confidence": round(verdict["confidence"], 6),
                "ml_prediction": ml_pred,
                "ml_confidence": round(ml_conf, 6),
                "dl_prediction": dl_result["dl_prediction"],
                "dl_confidence": round(dl_result["dl_confidence"], 6),
                "dl_stage1_attack_prob": round(dl_result["dl_stage1_attack_prob"], 6),
                "ae_anomaly_score": round(ae_score, 6),
                "zero_day_flag": zero_day_flag,
                "engine_agreement": verdict["engine_agreement"],
                "verdict_source": verdict["verdict_source"],
            })

        except Exception as exc:
            write_message({"requestId": request_id, "error": str(exc)})


if __name__ == "__main__":
    main()
