"""API tests for POST /api/v1/detect, using dependency_overrides so no real
model ever loads. Covers 422 validation, 200 happy/degraded paths, 503 on
engine/repository unavailability, and the Finding-0 regression guard
(FlowInput.to_sample() must emit the 42 canonical keys in training order)."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from models.src.explain import ExplanationMethod, ExplanationResult
from service.detection_api.src.models.schemas import PredictionResponse
from service.detection_api.src.routes.detect import get_detection_service, router
from service.detection_api.src.services.detection import DetectionUnavailableError


class FakeDetectionService:
    def __init__(self, *, response=None, raise_error: Exception | None = None):
        self._response = response or PredictionResponse(
            prediction="BENIGN",
            confidence=0.95,
            attack_probability=0.05,
            model_name="ml_ensemble_xgb_rf_lgbm",
            anomaly_score=None,
            zero_day_flag=False,
            shap_explanation=None,
        )
        self._raise_error = raise_error
        self.received_flow_keys: list[str] = []

    async def detect_flow(self, flow, *, flow_key, explain=True):
        self.received_flow_keys.append(flow_key)
        if self._raise_error is not None:
            raise self._raise_error
        return self._response


def _build_app(service: FakeDetectionService) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_detection_service] = lambda: service
    return app


def _valid_flow(benign_flow: dict) -> dict:
    return {**benign_flow, "src_ip": "10.0.0.1", "dst_ip": "10.0.0.2", "src_port": 1234, "dst_port": 80}


class TestValidationErrors:
    def test_missing_required_field_returns_422(self, benign_flow: dict) -> None:
        flow = _valid_flow(benign_flow)
        del flow["dur"]
        client = TestClient(_build_app(FakeDetectionService()))
        response = client.post("/api/v1/detect", json={"flows": [flow]})
        assert response.status_code == 422

    def test_unknown_extra_field_returns_422(self, benign_flow: dict) -> None:
        flow = {**_valid_flow(benign_flow), "made_up": 1}
        client = TestClient(_build_app(FakeDetectionService()))
        response = client.post("/api/v1/detect", json={"flows": [flow]})
        assert response.status_code == 422

    def test_empty_flows_list_returns_422(self) -> None:
        client = TestClient(_build_app(FakeDetectionService()))
        response = client.post("/api/v1/detect", json={"flows": []})
        assert response.status_code == 422

    def test_over_100_flows_returns_422(self, benign_flow: dict) -> None:
        flow = _valid_flow(benign_flow)
        client = TestClient(_build_app(FakeDetectionService()))
        response = client.post("/api/v1/detect", json={"flows": [flow] * 101})
        assert response.status_code == 422

    def test_negative_numeric_field_returns_422(self, benign_flow: dict) -> None:
        flow = {**_valid_flow(benign_flow), "sbytes": -1}
        client = TestClient(_build_app(FakeDetectionService()))
        response = client.post("/api/v1/detect", json={"flows": [flow]})
        assert response.status_code == 422

    def test_overlong_categorical_field_returns_422(self, benign_flow: dict) -> None:
        flow = {**_valid_flow(benign_flow), "proto": "x" * 64}
        client = TestClient(_build_app(FakeDetectionService()))
        response = client.post("/api/v1/detect", json={"flows": [flow]})
        assert response.status_code == 422

    def test_invalid_ip_returns_422(self, benign_flow: dict) -> None:
        flow = {**_valid_flow(benign_flow), "src_ip": "not-an-ip"}
        client = TestClient(_build_app(FakeDetectionService()))
        response = client.post("/api/v1/detect", json={"flows": [flow]})
        assert response.status_code == 422


class TestHappyPath:
    def test_valid_flow_returns_200_with_matching_result_count(self, benign_flow: dict) -> None:
        flow = _valid_flow(benign_flow)
        client = TestClient(_build_app(FakeDetectionService()))
        response = client.post("/api/v1/detect", json={"flows": [flow, flow]})
        assert response.status_code == 200
        body = response.json()
        assert len(body["results"]) == 2

    def test_response_has_expected_top_level_keys(self, benign_flow: dict) -> None:
        flow = _valid_flow(benign_flow)
        client = TestClient(_build_app(FakeDetectionService()))
        response = client.post("/api/v1/detect", json={"flows": [flow]})
        body = response.json()
        assert set(body.keys()) == {"results", "request_id", "elapsed_ms"}
        assert set(body["results"][0].keys()) == {
            "prediction",
            "confidence",
            "attack_probability",
            "model_name",
            "anomaly_score",
            "zero_day_flag",
            "shap_explanation",
        }


class TestExplanationUnavailablePath:
    def test_explainer_failure_still_returns_200(self, benign_flow: dict) -> None:
        degraded = PredictionResponse(
            prediction="DoS",
            confidence=0.9,
            attack_probability=0.9,
            model_name="ml_ensemble_xgb_rf_lgbm",
            anomaly_score=None,
            zero_day_flag=False,
            shap_explanation=None,
        )
        flow = _valid_flow(benign_flow)
        client = TestClient(_build_app(FakeDetectionService(response=degraded)))
        response = client.post("/api/v1/detect", json={"flows": [flow]})
        assert response.status_code == 200
        assert response.json()["results"][0]["shap_explanation"] is None


class TestEngineUnavailable:
    def test_detection_unavailable_error_returns_503(self, benign_flow: dict) -> None:
        flow = _valid_flow(benign_flow)
        service = FakeDetectionService(raise_error=DetectionUnavailableError("db down"))
        client = TestClient(_build_app(service))
        response = client.post("/api/v1/detect", json={"flows": [flow]})
        assert response.status_code == 503

    def test_missing_service_dependency_returns_503(self, benign_flow: dict) -> None:
        from fastapi import FastAPI as _FastAPI

        app = _FastAPI()
        app.include_router(router)
        # No dependency_overrides: request.app.state.detection_service is unset.
        client = TestClient(app)
        response = client.post("/api/v1/detect", json={"flows": [_valid_flow(benign_flow)]})
        assert response.status_code == 503


class TestFlowKeyDerivation:
    def test_flow_key_derived_from_ips_when_not_explicit(self, benign_flow: dict) -> None:
        flow = _valid_flow(benign_flow)
        service = FakeDetectionService()
        client = TestClient(_build_app(service))
        client.post("/api/v1/detect", json={"flows": [flow]})
        assert service.received_flow_keys == ["10.0.0.1->10.0.0.2"]

    def test_explicit_flow_key_overrides_derived_key(self, benign_flow: dict) -> None:
        flow = _valid_flow(benign_flow)
        service = FakeDetectionService()
        client = TestClient(_build_app(service))
        client.post("/api/v1/detect", json={"flows": [flow], "flow_key": "custom-key"})
        assert service.received_flow_keys == ["custom-key"]


class TestFindingZeroRegressionGuard:
    def test_to_sample_emits_canonical_key_order(self, benign_flow: dict) -> None:
        from service.detection_api.src.models.schemas import CANONICAL_FEATURE_COLUMNS, FlowInput

        flow = FlowInput(**_valid_flow(benign_flow))
        assert list(flow.to_sample().keys()) == list(CANONICAL_FEATURE_COLUMNS)
