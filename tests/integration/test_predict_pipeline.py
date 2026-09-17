"""Phase 8 integration test: 12 consecutive flows from live_test_dataset.csv
through the real DetectionService — real UnifiedEngineBundle (all 5 trained
models), real TreeEnsembleExplainer/DeepSequenceExplainer/LimeSequenceExplainer
via ExplanationService, and a real (in-memory) AlertRepository. No FastAPI
layer involved (that's Dev 2's app assembly); this proves the predict ->
explain -> persist wiring itself is correct end-to-end.

Skips cleanly when artifacts are absent (session-scoped model loading is slow
-- several seconds -- hence one shared bundle/service for the whole module)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from models.src.explain import DeepSequenceExplainer, ExplanationService, LimeSequenceExplainer, TreeEnsembleExplainer
from service.detection_api.src.models.schemas import AlertFilter, FlowInput
from service.detection_api.src.services.alert_repository import InMemoryAlertRepository
from service.detection_api.src.services.alerting import NullNotifier
from service.detection_api.src.services.detection import DetectionConfig, DetectionService, UnifiedEngineBundle

ROOT = Path(__file__).resolve().parents[2]
NUM_FLOWS = 12
SEQ_LEN = 10


@pytest.fixture(scope="module")
def bundle() -> UnifiedEngineBundle:
    return UnifiedEngineBundle()


@pytest.fixture(scope="module")
def dl_background(bundle: UnifiedEngineBundle) -> torch.Tensor:
    df = pd.read_csv(ROOT / "tests" / "live_test_dataset.csv", nrows=40)
    drop = {"id", "attack_cat", "label"}
    cols = [c for c in df.columns if c not in drop]
    vectors = np.stack([bundle.preprocess_dl(df.iloc[i][cols].to_dict()) for i in range(40)])
    windows = np.stack([vectors[i : i + SEQ_LEN] for i in range(0, 20, 2)])  # (10, 10, 198)
    return torch.tensor(windows, dtype=torch.float32)


def _make_lime_factory(model_call, feature_names: list[str], background: np.ndarray):
    """Builds a factory that, given a (10, n_features) window, returns a
    LimeSequenceExplainer whose predict_fn freezes the window's first 9 steps
    and perturbs only the last -- required because LIME's predict_fn closes
    over that specific prefix (see ExplanationService's docstring)."""

    def factory(window: np.ndarray) -> LimeSequenceExplainer:
        prefix = window[:-1]

        def predict_fn(x2d: np.ndarray) -> np.ndarray:
            k = x2d.shape[0]
            batch = np.repeat(prefix[None, :, :], k, axis=0)
            batch = np.concatenate([batch, x2d[:, None, :]], axis=1)
            with torch.no_grad():
                return model_call(torch.from_numpy(batch).float()).numpy()

        return LimeSequenceExplainer(
            predict_fn=predict_fn, feature_names=feature_names, background=background, num_samples=100
        )

    return factory


@pytest.fixture(scope="module")
def explanation_service(bundle: UnifiedEngineBundle, dl_background: torch.Tensor) -> ExplanationService:
    xgb, rf, lgbm = bundle.ml_models
    tree_explainer = TreeEnsembleExplainer(xgb=xgb, rf=rf, lgbm=lgbm, feature_names=bundle.selected_ml_feature_names)
    deep_explainer = DeepSequenceExplainer(
        model=bundle.stage1_model, feature_names=bundle.dl_feature_names, background=dl_background
    )

    dl_feature_background = np.random.randn(20, len(bundle.dl_feature_names)).astype(np.float32)
    lime_dl_factory = _make_lime_factory(
        lambda t: bundle.stage1_model.predict_proba(t)[:, 1], bundle.dl_feature_names, dl_feature_background
    )
    ae_lime_factory = _make_lime_factory(
        lambda t: bundle.ae_model.anomaly_score(t), bundle.dl_feature_names, dl_feature_background
    )

    return ExplanationService(
        tree_explainer=tree_explainer,
        ml_transform=bundle.ml_transform,
        ml_label_classes=bundle.ml_label_classes,
        deep_explainer=deep_explainer,
        lime_dl_factory=lime_dl_factory,
        ae_lime_factory=ae_lime_factory,
    )


@pytest.fixture(scope="module")
def flows() -> list[FlowInput]:
    df = pd.read_csv(ROOT / "tests" / "live_test_dataset.csv", nrows=NUM_FLOWS)
    drop = {"id", "attack_cat", "label"}
    cols = [c for c in df.columns if c not in drop]
    return [FlowInput(**df.iloc[i][cols].to_dict()) for i in range(NUM_FLOWS)]


@pytest.mark.requires_artifacts
class TestPredictPipeline:
    @pytest.mark.asyncio
    async def test_full_pipeline_end_to_end(
        self, bundle: UnifiedEngineBundle, explanation_service: ExplanationService, flows: list[FlowInput]
    ) -> None:
        repo = InMemoryAlertRepository()
        service = DetectionService(
            engines=bundle,
            repo=repo,
            notifier=NullNotifier(),
            explain_fn=explanation_service.explain,
            config=DetectionConfig(),
        )

        responses = []
        for flow in flows:
            response = await service.detect_flow(flow, flow_key="integration-test", explain=True)
            responses.append(response)

        # json.dumps must succeed for every response -- guards against a
        # numpy scalar (float32/int64) leaking into the Pydantic response,
        # which works fine in-memory but breaks JSON/JSONB persistence.
        for response in responses:
            json.dumps(response.model_dump(mode="json"))

        # FlowWindowStore.DEFAULT_SEQ_LEN is 10: the window is cold for the
        # first 9 pushes and warm from the 10th push onward, deterministic
        # regardless of what the models actually predict.
        for response in responses[: SEQ_LEN - 1]:
            assert response.anomaly_score is None
        for response in responses[SEQ_LEN - 1 :]:
            assert response.anomaly_score is not None

        # Every response gets a shap_explanation (never None) when explain=True,
        # per the Definition of Done -- even if the method ends up "unavailable".
        assert all(r.shap_explanation is not None for r in responses)

        # At least one explanation actually succeeded via a real explainer
        # (not every one degraded to "unavailable") -- proves the real
        # TreeEnsembleExplainer/DeepSequenceExplainer/LIME wiring works, not
        # just its fallback path.
        methods = {r.shap_explanation.explanation_method for r in responses}
        assert methods & {"shap_tree", "shap_deep", "lime"}

        # Persistence policy: exactly the non-BENIGN (or zero-day) responses
        # were persisted -- verifies the predict -> explain -> persist wiring
        # itself, without hardcoding which specific rows the real models
        # classify as attacks.
        expected_persisted = sum(1 for r in responses if r.prediction != "BENIGN" or r.zero_day_flag)
        alert_page = await repo.list(filters=AlertFilter(), page=1, per_page=100)
        assert alert_page.total == expected_persisted
