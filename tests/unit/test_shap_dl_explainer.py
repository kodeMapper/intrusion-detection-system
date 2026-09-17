"""Tests for DeepSequenceExplainer against the real Stage1 Transformer
checkpoint, using real preprocessed flow windows (not random noise) so the
additivity check is meaningful — expected gradients is only a good
approximation near the actual input distribution. Tolerant of the
documented "Transformer SHAP incompatibility" risk: passes whether the
method ends up shap_deep, lime (fallback), or (only if truly nothing works)
unavailable — falling back is the specified behavior, not a test failure.
Skips cleanly when artifacts are absent.

Note: DeepSequenceExplainer's explain() groups the 156 one-hot
proto_*/service_*/state_* columns into 3 logical features before ranking
(see explain.group_one_hot), so the returned feature_names has 45 entries
(42 numeric/derived + 3 grouped), not the raw 198."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from models.src.explain import DeepSequenceExplainer, ExplanationMethod

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "service" / "models" / "artifacts"
GROUPED_FEATURE_COUNT = 45  # 42 numeric/derived + proto/service/state grouped


@pytest.fixture(scope="module")
def stage1_model():
    sys.path.insert(0, str(ROOT / "service"))
    from models.src.dl_pipeline import TemporalTransformerClassifier

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = json.loads((MODEL_DIR / "dl_stage1_binary_threshold_v1.0.json").read_text())
        hp = cfg["model_hparams"]
        model = TemporalTransformerClassifier(
            input_size=hp["input_size"], seq_len=hp["seq_len"], num_classes=hp["num_classes"],
            d_model=hp["d_model"], n_heads=hp["n_heads"], num_layers=hp["num_layers"],
            dim_feedforward=hp["dim_feedforward"], classifier_hidden=hp["classifier_hidden"], dropout=hp["dropout"],
        )
        state = torch.load(MODEL_DIR / "dl_stage1_binary_v1.0.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model


@pytest.fixture(scope="module")
def dl_feature_names() -> list[str]:
    data = json.loads((ROOT / "data" / "artifacts" / "feature_list_unsw_v1_20260612.json").read_text())
    return data if isinstance(data, list) else data["feature_columns"]


@pytest.fixture(scope="module")
def dl_preprocessor():
    sys.path.insert(0, str(ROOT / "service"))
    from preproc import DLPreprocessor

    preproc_dir = ROOT / "data" / "artifacts"
    return DLPreprocessor(preproc_dir)


@pytest.fixture(scope="module")
def real_windows(dl_preprocessor):
    """Real preprocessed vectors from live_test_dataset.csv: a (10,10,198)
    background of overlapping windows and one held-out (1,10,198) window."""
    df = pd.read_csv(ROOT / "tests" / "live_test_dataset.csv", nrows=40)
    drop = {"id", "attack_cat", "label"}
    cols = [c for c in df.columns if c not in drop]
    vectors = np.stack(
        [dl_preprocessor.transform(pd.DataFrame([df.iloc[i][cols].to_dict()]))[0] for i in range(40)]
    )
    background = np.stack([vectors[i : i + 10] for i in range(0, 20, 2)])  # (10, 10, 198)
    window = vectors[20:30][None, :, :]  # (1, 10, 198)
    raw_sample = df.iloc[29][cols].to_dict()
    return (
        torch.tensor(background, dtype=torch.float32),
        torch.tensor(window, dtype=torch.float32),
        raw_sample,
    )


@pytest.mark.requires_artifacts
class TestDeepSequenceExplainer:
    def test_explain_returns_a_supported_method(self, stage1_model, dl_feature_names, real_windows) -> None:
        background, window, raw_sample = real_windows
        explainer = DeepSequenceExplainer(model=stage1_model, feature_names=dl_feature_names, background=background)
        result = explainer.explain(window=window, class_index=1, raw_sample=raw_sample, predicted_is_attack=True)

        assert result.method in (ExplanationMethod.SHAP_DEEP, ExplanationMethod.LIME, ExplanationMethod.UNAVAILABLE)
        assert result.elapsed_ms < 5000

    def test_feature_dimensionality_matches_grouped_count(self, stage1_model, dl_feature_names, real_windows) -> None:
        background, window, raw_sample = real_windows
        explainer = DeepSequenceExplainer(model=stage1_model, feature_names=dl_feature_names, background=background)
        result = explainer.explain(window=window, class_index=1, raw_sample=raw_sample, predicted_is_attack=True)

        if result.method != ExplanationMethod.UNAVAILABLE:
            assert len(result.feature_names) == GROUPED_FEATURE_COUNT
            assert len(result.all_shap_values) == GROUPED_FEATURE_COUNT
            assert len(result.top_features) <= 10

    def test_top_feature_value_is_not_the_shap_value(self, stage1_model, dl_feature_names, real_windows) -> None:
        # Regression: passthrough numeric features (everything but the 3
        # grouped proto/service/state) must report their real transformed
        # value, not silently fall back to duplicating shap_value (a real
        # bug caught in review — group_one_hot's value_map only ever
        # populated the 3 grouped categorical names).
        background, window, raw_sample = real_windows
        explainer = DeepSequenceExplainer(model=stage1_model, feature_names=dl_feature_names, background=background)
        result = explainer.explain(window=window, class_index=1, raw_sample=raw_sample, predicted_is_attack=True)

        if result.method == ExplanationMethod.SHAP_DEEP:
            grouped_categorical = {"proto", "service", "state"}
            numeric_features = [
                f for f in result.top_features if f.feature in raw_sample and f.feature not in grouped_categorical
            ]
            assert numeric_features, "expected at least one passthrough numeric feature in top_features"
            for f in numeric_features:
                assert f.value == pytest.approx(float(raw_sample[f.feature]))

    def test_additivity_only_asserted_when_shap_deep(self, stage1_model, dl_feature_names, real_windows) -> None:
        # Seeded for determinism: GradientExplainer's expected-gradients
        # estimate is a Monte Carlo approximation (random interpolation
        # points between background and input), so the exact sum varies
        # run to run without a fixed seed. base_value/predicted_value are
        # logits (see explain.py's DeepSequenceExplainer docstring) — a
        # ~0.3 residual at nsamples=100 is the approximation's real,
        # expected error, not a bug; tightened once nsamples was fixed to
        # actually control this in explain.py (was a genuine bug: comparing
        # against predict_proba() instead of raw logits, ~1.0-1.4 error).
        torch.manual_seed(0)
        background, window, raw_sample = real_windows
        explainer = DeepSequenceExplainer(model=stage1_model, feature_names=dl_feature_names, background=background)
        result = explainer.explain(window=window, class_index=1, raw_sample=raw_sample, predicted_is_attack=True)

        if result.method == ExplanationMethod.SHAP_DEEP:
            reconstructed = result.base_value + sum(result.all_shap_values)
            assert reconstructed == pytest.approx(result.predicted_value, abs=0.5)
