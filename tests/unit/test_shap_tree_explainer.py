"""Tests for TreeEnsembleExplainer against the real XGBoost/RF/LightGBM
artifacts. Skips cleanly when final_rf.pkl (gitignored) is absent."""
from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from models.src.explain import ExplanationMethod, TreeEnsembleExplainer

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "service" / "models" / "artifacts"
ML_DROP_COLUMNS = {"_id", "_sampleIndex", "srcip", "dstip", "id", "Stime", "Ltime", "attack_cat", "label"}

# Per DEV1_PROGRESS.md: Stage 2 recall for "Exploits" is only 41.25% (frequent
# confusion with DoS) — explanations for that class are lower-trust and
# should not be asserted against a specific expected top feature.
LOW_TRUST_CLASSES = {"Exploits"}


@pytest.fixture(scope="module")
def ml_artifacts():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "xgb": joblib.load(MODEL_DIR / "final_xgb.pkl"),
            "rf": joblib.load(MODEL_DIR / "final_rf.pkl"),
            "lgbm": joblib.load(MODEL_DIR / "final_lgbm.pkl"),
            "selector": joblib.load(MODEL_DIR / "feature_selector.pkl"),
            "encoders": joblib.load(MODEL_DIR / "final_encoders.pkl"),
            "labels": joblib.load(MODEL_DIR / "final_labels.pkl"),
        }


def _preprocess(df: pd.DataFrame, artifacts: dict) -> np.ndarray:
    d = df.drop(columns=[c for c in ML_DROP_COLUMNS if c in df.columns], errors="ignore").copy()
    for col, enc in artifacts["encoders"].items():
        if col in d.columns:
            class_map = {v: i for i, v in enumerate(enc.classes_)}
            d[col] = d[col].astype(str).map(lambda v, cm=class_map: cm.get(v, 0))
    for col in d.columns:
        if col not in artifacts["encoders"]:
            d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0)
    return artifacts["selector"].transform(d.to_numpy())


@pytest.fixture(scope="module")
def selected_feature_names(ml_artifacts) -> list[str]:
    from tests.conftest import CANONICAL_FEATURE_COLUMNS

    support = ml_artifacts["selector"].get_support()
    return [name for name, keep in zip(CANONICAL_FEATURE_COLUMNS, support) if keep]


@pytest.mark.requires_artifacts
class TestTreeEnsembleExplainer:
    def test_returns_shap_tree_method(self, ml_artifacts, selected_feature_names, live_dataset) -> None:
        row_df = live_dataset.iloc[[8]]  # first attack row (label==1)
        transformed = _preprocess(row_df, ml_artifacts)
        raw_sample = row_df.iloc[0].to_dict()

        explainer = TreeEnsembleExplainer(
            xgb=ml_artifacts["xgb"], rf=ml_artifacts["rf"], lgbm=ml_artifacts["lgbm"],
            feature_names=selected_feature_names,
        )
        pred_label = live_dataset.iloc[8]["attack_cat"]
        class_index = list(ml_artifacts["labels"].classes_).index(pred_label)

        result = explainer.explain(
            transformed_row=transformed,
            raw_sample=raw_sample,
            class_index=class_index,
            predicted_is_attack=pred_label != "Normal",
        )

        assert result.method == ExplanationMethod.SHAP_TREE
        assert len(result.top_features) <= 10
        assert all(f.feature in selected_feature_names for f in result.top_features)
        assert result.elapsed_ms < 5000

    def test_additivity_holds_per_model_output_space(self, ml_artifacts, selected_feature_names, live_dataset) -> None:
        row_df = live_dataset.iloc[[8]]
        transformed = _preprocess(row_df, ml_artifacts)
        raw_sample = row_df.iloc[0].to_dict()
        pred_label = live_dataset.iloc[8]["attack_cat"]
        class_index = list(ml_artifacts["labels"].classes_).index(pred_label)

        explainer = TreeEnsembleExplainer(
            xgb=ml_artifacts["xgb"], rf=ml_artifacts["rf"], lgbm=ml_artifacts["lgbm"],
            feature_names=selected_feature_names,
        )
        result = explainer.explain(
            transformed_row=transformed, raw_sample=raw_sample, class_index=class_index, predicted_is_attack=True
        )
        # base_value + sum(all_shap_values) should reconstruct predicted_value
        # (additivity is exact per-model in its own raw output space; the
        # averaged ensemble value is what we assert here).
        assert result.base_value + sum(result.all_shap_values) == pytest.approx(result.predicted_value, abs=1e-2)

    def test_values_are_plain_python_floats_not_numpy(self, ml_artifacts, selected_feature_names, live_dataset) -> None:
        row_df = live_dataset.iloc[[0]]
        transformed = _preprocess(row_df, ml_artifacts)
        raw_sample = row_df.iloc[0].to_dict()
        explainer = TreeEnsembleExplainer(
            xgb=ml_artifacts["xgb"], rf=ml_artifacts["rf"], lgbm=ml_artifacts["lgbm"],
            feature_names=selected_feature_names,
        )
        result = explainer.explain(transformed_row=transformed, raw_sample=raw_sample, class_index=4, predicted_is_attack=False)
        import json

        json.dumps(
            {
                "base_value": result.base_value,
                "predicted_value": result.predicted_value,
                "all_shap_values": list(result.all_shap_values),
            }
        )

    def test_no_high_confidence_assertion_for_low_trust_classes(self) -> None:
        # Documents the known Stage2 weakness (41.25% recall on Exploits per
        # ml_engine_final_assessment.md) — intentionally no assertion here
        # beyond confirming the class is flagged as low-trust for review.
        assert "Exploits" in LOW_TRUST_CLASSES
