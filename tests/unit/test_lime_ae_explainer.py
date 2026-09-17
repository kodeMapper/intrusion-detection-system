"""Tests for LimeSequenceExplainer against the real TemporalOneClassVAE
checkpoint — explains anomaly_score() (reconstruction-based), not a
classification probability. Skips cleanly when artifacts are absent."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from models.src.explain import ExplanationMethod, LimeSequenceExplainer

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "service" / "models" / "artifacts"


@pytest.fixture(scope="module")
def ae_model():
    sys.path.insert(0, str(ROOT / "service"))
    from models.src.dl_pipeline import TemporalOneClassVAE

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = json.loads((MODEL_DIR / "dl_ae_threshold_v1.0.json").read_text())
        hp = cfg["model_hparams"]
        model = TemporalOneClassVAE(
            input_size=hp["input_size"], seq_len=hp["seq_len"], d_model=hp["d_model"],
            latent_dim=hp["latent_dim"], n_heads=hp["n_heads"], num_layers=hp["num_layers"],
            dim_feedforward=hp["dim_feedforward"], decoder_hidden=hp["decoder_hidden"],
            decoder_layers=hp["decoder_layers"], dropout=hp["dropout"],
            score_weights=tuple(hp.get("score_weights", (0.1, 0.6, 0.3))),
        )
        state = torch.load(MODEL_DIR / "dl_ae_v1.0.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        cal = cfg.get("score_calibration", {})
        if cal:
            model.score_median.copy_(torch.tensor(cal["median"], dtype=torch.float32))
            model.score_iqr.copy_(torch.tensor(cal["iqr"], dtype=torch.float32))
            model.is_score_calibrated.copy_(torch.tensor(True, dtype=torch.bool))
        return model


@pytest.fixture(scope="module")
def feature_names() -> list[str]:
    data = json.loads((ROOT / "data" / "artifacts" / "feature_list_unsw_v1_20260612.json").read_text())
    return data if isinstance(data, list) else data["feature_columns"]


@pytest.mark.requires_artifacts
class TestLimeSequenceExplainerOnAe:
    def test_method_is_lime(self, ae_model, feature_names) -> None:
        n = len(feature_names)
        prefix = np.random.randn(9, n).astype(np.float32)
        last_step = np.random.randn(n).astype(np.float32)
        background = np.random.randn(20, n).astype(np.float32)

        def predict_fn(x2d: np.ndarray) -> np.ndarray:
            k = x2d.shape[0]
            window = np.repeat(prefix[None, :, :], k, axis=0)
            window = np.concatenate([window, x2d[:, None, :]], axis=1)
            with torch.no_grad():
                scores = ae_model.anomaly_score(torch.from_numpy(window).float())
            return scores.numpy()

        explainer = LimeSequenceExplainer(predict_fn=predict_fn, feature_names=feature_names, background=background, num_samples=100)
        result = explainer.explain(last_step=last_step, raw_sample={n: 0.0 for n in feature_names})

        assert result.method == ExplanationMethod.LIME
        assert result.elapsed_ms < 10000

    def test_predicted_value_matches_real_anomaly_score(self, ae_model, feature_names) -> None:
        n = len(feature_names)
        prefix = np.random.randn(9, n).astype(np.float32)
        last_step = np.random.randn(n).astype(np.float32)
        background = np.random.randn(20, n).astype(np.float32)

        def predict_fn(x2d: np.ndarray) -> np.ndarray:
            k = x2d.shape[0]
            window = np.repeat(prefix[None, :, :], k, axis=0)
            window = np.concatenate([window, x2d[:, None, :]], axis=1)
            with torch.no_grad():
                scores = ae_model.anomaly_score(torch.from_numpy(window).float())
            return scores.numpy()

        full_window = np.concatenate([prefix, last_step[None, :]], axis=0)
        with torch.no_grad():
            expected_score = float(ae_model.anomaly_score(torch.from_numpy(full_window[None]).float()).item())

        explainer = LimeSequenceExplainer(predict_fn=predict_fn, feature_names=feature_names, background=background, num_samples=100)
        result = explainer.explain(last_step=last_step, raw_sample={n: 0.0 for n in feature_names})

        assert result.predicted_value == pytest.approx(expected_score, abs=1e-4)

    def test_feature_names_length_matches_grouped_count(self, ae_model, feature_names) -> None:
        # LimeSequenceExplainer.explain() groups the 156 one-hot
        # proto_*/service_*/state_* columns into 3 logical features (see
        # explain.group_one_hot), so the result has 45 entries, not 198.
        n = len(feature_names)
        grouped_count = 45
        prefix = np.random.randn(9, n).astype(np.float32)
        last_step = np.random.randn(n).astype(np.float32)
        background = np.random.randn(20, n).astype(np.float32)

        def predict_fn(x2d: np.ndarray) -> np.ndarray:
            k = x2d.shape[0]
            window = np.repeat(prefix[None, :, :], k, axis=0)
            window = np.concatenate([window, x2d[:, None, :]], axis=1)
            with torch.no_grad():
                scores = ae_model.anomaly_score(torch.from_numpy(window).float())
            return scores.numpy()

        explainer = LimeSequenceExplainer(predict_fn=predict_fn, feature_names=feature_names, background=background, num_samples=100)
        result = explainer.explain(last_step=last_step, raw_sample={n: 0.0 for n in feature_names})

        assert len(result.feature_names) == grouped_count
        assert len(result.all_shap_values) == grouped_count

    def test_top_feature_value_is_not_the_shap_value(self, ae_model, feature_names) -> None:
        # Regression: same TopFeature.value bug as DeepSequenceExplainer —
        # passthrough numeric features must report their real (raw_sample)
        # value, not silently duplicate shap_value.
        n = len(feature_names)
        prefix = np.random.randn(9, n).astype(np.float32)
        last_step = np.random.randn(n).astype(np.float32)
        background = np.random.randn(20, n).astype(np.float32)
        raw_sample = {name: 0.0 for name in feature_names}

        def predict_fn(x2d: np.ndarray) -> np.ndarray:
            k = x2d.shape[0]
            window = np.repeat(prefix[None, :, :], k, axis=0)
            window = np.concatenate([window, x2d[:, None, :]], axis=1)
            with torch.no_grad():
                scores = ae_model.anomaly_score(torch.from_numpy(window).float())
            return scores.numpy()

        explainer = LimeSequenceExplainer(predict_fn=predict_fn, feature_names=feature_names, background=background, num_samples=100)
        result = explainer.explain(last_step=last_step, raw_sample=raw_sample)

        grouped_categorical = {"proto", "service", "state"}
        numeric_features = [
            f for f in result.top_features if f.feature in raw_sample and f.feature not in grouped_categorical
        ]
        assert numeric_features, "expected at least one passthrough numeric feature in top_features"
        for f in numeric_features:
            assert f.value == pytest.approx(0.0)
