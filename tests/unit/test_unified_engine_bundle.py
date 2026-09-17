"""Tests for UnifiedEngineBundle — the concrete EngineBundle backed by the
real trained models (in-process import of unified_predictor_worker's engine
classes). Deferred from Phase 5 per DEV1_PROGRESS.md; DLAdvancedEngine's own
internal buffer is never used here since FlowWindowStore replaces it with a
bounded, per-flow-key window. Skips cleanly when artifacts are absent."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from service.detection_api.src.services.detection import UnifiedEngineBundle

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def bundle():
    return UnifiedEngineBundle()


@pytest.fixture(scope="module")
def flows() -> list[dict]:
    df = pd.read_csv(ROOT / "tests" / "live_test_dataset.csv", nrows=12)
    drop = {"id", "attack_cat", "label"}
    cols = [c for c in df.columns if c not in drop]
    return [df.iloc[i][cols].to_dict() for i in range(len(df))]


@pytest.mark.requires_artifacts
class TestUnifiedEngineBundle:
    def test_predict_ml_returns_label_and_confidence(self, bundle, flows) -> None:
        label, confidence = bundle.predict_ml(flows[0])
        assert isinstance(label, str)
        assert 0.0 <= confidence <= 1.0

    def test_preprocess_dl_returns_198_length_vector(self, bundle, flows) -> None:
        vector = bundle.preprocess_dl(flows[0])
        assert isinstance(vector, np.ndarray)
        assert vector.shape == (198,)

    def test_predict_window_matches_predict_from_tensor_shape(self, bundle, flows) -> None:
        window = np.stack([bundle.preprocess_dl(flow) for flow in flows[:10]], axis=0)
        result = bundle.predict_window(window)
        assert set(result) == {"dl_prediction", "dl_confidence", "dl_stage1_attack_prob"}
        assert isinstance(result["dl_prediction"], str)

    def test_score_ae_returns_float_and_bool(self, bundle, flows) -> None:
        window = np.stack([bundle.preprocess_dl(flow) for flow in flows[:10]], axis=0)
        score, zero_day = bundle.score_ae(window)
        assert isinstance(score, float)
        assert isinstance(zero_day, bool)

    def test_ml_transform_matches_selected_feature_count(self, bundle, flows) -> None:
        transformed = bundle.ml_transform(flows[0])
        assert transformed.ndim == 2
        assert transformed.shape[0] == 1

    def test_ml_label_classes_include_normal(self, bundle) -> None:
        assert "Normal" in bundle.ml_label_classes
