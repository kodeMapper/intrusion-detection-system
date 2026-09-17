"""Snapshot tests guarding the frozen PredictionResponse/ShapExplanation/TopFeature
contract (dev_assignment_plan.md, additive-only versioning), plus behavioral
tests for FlowInput.to_sample()'s canonical column ordering (the Finding-0
regression guard: MLBaselineEngine._preprocess has no way to realign columns
on its own, so the request adapter must emit them in training order)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from service.detection_api.src.models.schemas import (
    CANONICAL_FEATURE_COLUMNS,
    DetectRequest,
    FlowInput,
    PredictionResponse,
    ShapExplanation,
    TopFeature,
)


class TestFrozenContractShape:
    def test_top_feature_fields_unchanged(self) -> None:
        assert set(TopFeature.model_fields) == {"feature", "value", "shap_value", "direction"}

    def test_shap_explanation_fields_unchanged(self) -> None:
        assert set(ShapExplanation.model_fields) == {
            "base_value",
            "predicted_value",
            "top_features",
            "all_shap_values",
            "feature_names",
            "explanation_method",
        }

    def test_prediction_response_fields_unchanged(self) -> None:
        assert set(PredictionResponse.model_fields) == {
            "prediction",
            "confidence",
            "attack_probability",
            "model_name",
            "anomaly_score",
            "zero_day_flag",
            "shap_explanation",
        }

    def test_shap_explanation_is_optional_on_prediction_response(self) -> None:
        response = PredictionResponse(
            prediction="BENIGN",
            confidence=0.9,
            attack_probability=0.1,
            model_name="ml_ensemble_xgb_rf_lgbm",
            anomaly_score=None,
            zero_day_flag=False,
            shap_explanation=None,
        )
        assert response.shap_explanation is None


class TestFlowInputToSample:
    def test_to_sample_emits_42_canonical_keys_in_order(self, benign_flow: dict) -> None:
        flow = FlowInput(**benign_flow)
        sample = flow.to_sample()
        assert list(sample.keys()) == list(CANONICAL_FEATURE_COLUMNS)
        assert len(sample) == 42

    def test_to_sample_excludes_optional_metadata_fields(self, benign_flow: dict) -> None:
        flow = FlowInput(**benign_flow, src_ip="10.0.0.1", dst_ip="10.0.0.2", src_port=1234, dst_port=80)
        sample = flow.to_sample()
        assert "src_ip" not in sample
        assert "dst_ip" not in sample
        assert "src_port" not in sample
        assert "dst_port" not in sample

    def test_rejects_unknown_extra_field(self, benign_flow: dict) -> None:
        with pytest.raises(ValidationError):
            FlowInput(**benign_flow, made_up_field=123)

    def test_rejects_negative_numeric_field(self, benign_flow: dict) -> None:
        bad = dict(benign_flow, sbytes=-5)
        with pytest.raises(ValidationError):
            FlowInput(**bad)

    def test_rejects_overlong_proto_string(self, benign_flow: dict) -> None:
        bad = dict(benign_flow, proto="x" * 64)
        with pytest.raises(ValidationError):
            FlowInput(**bad)

    def test_rejects_invalid_ip(self, benign_flow: dict) -> None:
        with pytest.raises(ValidationError):
            FlowInput(**benign_flow, src_ip="not-an-ip")

    def test_accepts_valid_ip(self, benign_flow: dict) -> None:
        flow = FlowInput(**benign_flow, src_ip="192.168.1.10")
        assert str(flow.src_ip) == "192.168.1.10"


class TestDetectRequest:
    def test_requires_at_least_one_flow(self) -> None:
        with pytest.raises(ValidationError):
            DetectRequest(flows=[])

    def test_rejects_more_than_100_flows(self, benign_flow: dict) -> None:
        with pytest.raises(ValidationError):
            DetectRequest(flows=[FlowInput(**benign_flow) for _ in range(101)])

    def test_explain_defaults_to_true(self, benign_flow: dict) -> None:
        request = DetectRequest(flows=[FlowInput(**benign_flow)])
        assert request.explain is True
