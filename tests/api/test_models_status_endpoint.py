"""API tests for GET /api/v1/models/status, using dependency_overrides so no
real model ever loads."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from service.detection_api.src.models.schemas import (
    InferenceStats,
    ModelStatus,
    ModelsStatusResponse,
    WindowState,
)
from service.detection_api.src.routes.models import get_models_status_provider, router


class FakeProvider:
    def status(self) -> ModelsStatusResponse:
        return ModelsStatusResponse(
            models=[
                ModelStatus(
                    name="xgboost",
                    kind="ml_ensemble",
                    version="v1.0",
                    loaded=True,
                    artifact="final_xgb.pkl",
                    explainer="shap_tree",
                )
            ],
            inference_stats=InferenceStats(total_predictions=0, avg_latency_ms=0.0, p99_latency_ms=0.0),
            window_state=WindowState(seq_len=10, tracked_keys=0, warm_keys=0),
        )


def _build_app(provider: FakeProvider) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_models_status_provider] = lambda: provider
    return app


class TestModelsStatusEndpoint:
    def test_returns_200_with_model_list(self) -> None:
        client = TestClient(_build_app(FakeProvider()))
        response = client.get("/api/v1/models/status")
        assert response.status_code == 200
        body = response.json()
        assert body["models"][0]["name"] == "xgboost"
        assert body["models"][0]["artifact"] == "final_xgb.pkl"

    def test_artifact_field_never_contains_a_path_separator(self) -> None:
        client = TestClient(_build_app(FakeProvider()))
        response = client.get("/api/v1/models/status")
        for model in response.json()["models"]:
            assert "/" not in model["artifact"]
            assert "\\" not in model["artifact"]

    def test_missing_provider_returns_503(self) -> None:
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        response = client.get("/api/v1/models/status")
        assert response.status_code == 503
