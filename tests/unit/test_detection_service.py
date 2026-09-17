"""Unit tests for DetectionService — fully mocked engines/explainer/repo/notifier,
no real models. Asserts the adapter mapping table (worker dict -> frozen
PredictionResponse), cold-window None handling, persist-only-for-attacks,
persist-before-notify ordering, and 503-worthy repository failure surfacing."""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pytest

from models.src.explain import ExplanationMethod, ExplanationResult
from service.detection_api.src.models.schemas import Alert, AlertCreate, FlowInput
from service.detection_api.src.services.alerting import NotificationResult
from service.detection_api.src.services.detection import (
    DetectionConfig,
    DetectionUnavailableError,
    DetectionService,
    FlowWindowStore,
)


class FakeEngineBundle:
    def __init__(self, *, ml_pred="Normal", ml_conf=0.9, dl_result=None, ae_score=0.1, zero_day=False):
        self.ml_pred = ml_pred
        self.ml_conf = ml_conf
        self.dl_result = dl_result or {"dl_prediction": "Normal", "dl_confidence": 0.9, "dl_stage1_attack_prob": 0.05}
        self.ae_score = ae_score
        self.zero_day = zero_day
        self.dl_vector_calls = 0

    def predict_ml(self, sample):
        return self.ml_pred, self.ml_conf

    def preprocess_dl(self, sample):
        self.dl_vector_calls += 1
        return np.zeros(198, dtype=np.float32)

    def predict_window(self, window):
        return self.dl_result

    def score_ae(self, window):
        return self.ae_score, self.zero_day


class FakeRepository:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.created: list[AlertCreate] = []

    async def create(self, alert: AlertCreate) -> Alert:
        if self.fail:
            raise RuntimeError("db unreachable")
        self.created.append(alert)
        return Alert(
            **alert.model_dump(),
            id=uuid4(),
            status="open",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

    async def get_by_id(self, id):
        raise NotImplementedError

    async def list(self, filters, page, per_page):
        raise NotImplementedError

    async def update_status(self, id, status, by):
        raise NotImplementedError


class SpyNotifier:
    def __init__(self):
        self.sent = []

    async def send(self, notification):
        self.sent.append(notification)
        return NotificationResult(success=True, channel="spy")


class RaisingNotifier:
    async def send(self, notification):
        raise RuntimeError("smtp exploded")


class FakeBackgroundTasks:
    """Matches fastapi.BackgroundTasks' add_task interface -- captures the
    call instead of running it, so tests can assert scheduling happened
    without needing a real FastAPI app/event loop shutdown to run it."""

    def __init__(self):
        self.tasks: list[tuple] = []

    def add_task(self, func, *args, **kwargs):
        self.tasks.append((func, args, kwargs))


def _flow(benign_flow: dict) -> FlowInput:
    return FlowInput(**benign_flow, src_ip="10.0.0.1", dst_ip="10.0.0.2", src_port=1234, dst_port=80)


def _make_service(engines, repo=None, notifier=None, explain_fn=None, config=None) -> DetectionService:
    return DetectionService(
        engines=engines,
        repo=repo or FakeRepository(),
        notifier=notifier or SpyNotifier(),
        explain_fn=explain_fn,
        config=config or DetectionConfig(),
    )


class TestColdWindowHandling:
    @pytest.mark.asyncio
    async def test_anomaly_score_is_none_when_window_cold(self, benign_flow: dict) -> None:
        engines = FakeEngineBundle()
        service = _make_service(engines)
        response = await service.detect_flow(_flow(benign_flow), flow_key="10.0.0.1->10.0.0.2", explain=False)
        assert response.anomaly_score is None

    @pytest.mark.asyncio
    async def test_dl_warming_up_maps_to_benign_prediction_source_ml(self, benign_flow: dict) -> None:
        engines = FakeEngineBundle(ml_pred="Normal", ml_conf=0.95)
        service = _make_service(engines)
        response = await service.detect_flow(_flow(benign_flow), flow_key="k1", explain=False)
        assert response.prediction == "BENIGN"


class TestWarmWindowHandling:
    @pytest.mark.asyncio
    async def test_anomaly_score_present_once_window_is_warm(self, benign_flow: dict) -> None:
        engines = FakeEngineBundle(ae_score=0.42)
        service = _make_service(engines)
        flow_key = "10.0.0.1->10.0.0.2"
        for _ in range(FlowWindowStore.DEFAULT_SEQ_LEN):
            response = await service.detect_flow(_flow(benign_flow), flow_key=flow_key, explain=False)
        assert response.anomaly_score == pytest.approx(0.42)

    @pytest.mark.asyncio
    async def test_zero_day_flag_passes_through(self, benign_flow: dict) -> None:
        engines = FakeEngineBundle(zero_day=True)
        service = _make_service(engines)
        flow_key = "warm-key"
        for _ in range(FlowWindowStore.DEFAULT_SEQ_LEN):
            response = await service.detect_flow(_flow(benign_flow), flow_key=flow_key, explain=False)
        assert response.zero_day_flag is True


class TestModelNameMapping:
    @pytest.mark.asyncio
    async def test_ml_only_verdict_uses_ml_ensemble_name(self, benign_flow: dict) -> None:
        engines = FakeEngineBundle(ml_pred="Normal", ml_conf=0.9)
        service = _make_service(engines)
        response = await service.detect_flow(_flow(benign_flow), flow_key="k", explain=False)
        assert "ml" in response.model_name.lower()


class TestExplanationHandling:
    @pytest.mark.asyncio
    async def test_explain_false_skips_explanation(self, benign_flow: dict) -> None:
        engines = FakeEngineBundle()
        called = {"count": 0}

        def explain_fn(ctx):
            called["count"] += 1
            return ExplanationResult.unavailable("should not be called")

        service = _make_service(engines, explain_fn=explain_fn)
        response = await service.detect_flow(_flow(benign_flow), flow_key="k", explain=False)
        assert response.shap_explanation is None
        assert called["count"] == 0

    @pytest.mark.asyncio
    async def test_explain_true_attaches_shap_explanation(self, benign_flow: dict) -> None:
        engines = FakeEngineBundle()

        def explain_fn(ctx):
            return ExplanationResult(
                base_value=0.1,
                predicted_value=0.9,
                top_features=(),
                all_shap_values=(0.1,),
                feature_names=("sttl",),
                method=ExplanationMethod.SHAP_TREE,
                elapsed_ms=10.0,
            )

        service = _make_service(engines, explain_fn=explain_fn)
        response = await service.detect_flow(_flow(benign_flow), flow_key="k", explain=True)
        assert response.shap_explanation is not None
        assert response.shap_explanation.explanation_method == "shap_tree"

    @pytest.mark.asyncio
    async def test_explainer_failure_still_returns_200_worthy_response(self, benign_flow: dict) -> None:
        engines = FakeEngineBundle()

        def explain_fn(ctx):
            raise ValueError("shap blew up")

        service = _make_service(engines, explain_fn=explain_fn)
        response = await service.detect_flow(_flow(benign_flow), flow_key="k", explain=True)
        assert response.shap_explanation is not None
        assert response.shap_explanation.explanation_method == "unavailable"


class TestPersistencePolicy:
    @pytest.mark.asyncio
    async def test_benign_flow_is_not_persisted(self, benign_flow: dict) -> None:
        engines = FakeEngineBundle(ml_pred="Normal", ml_conf=0.95)
        repo = FakeRepository()
        service = _make_service(engines, repo=repo)
        await service.detect_flow(_flow(benign_flow), flow_key="k", explain=False)
        assert len(repo.created) == 0

    @pytest.mark.asyncio
    async def test_attack_flow_is_persisted(self, attack_flow: dict) -> None:
        engines = FakeEngineBundle(ml_pred="DoS", ml_conf=0.95)
        repo = FakeRepository()
        service = _make_service(engines, repo=repo)
        await service.detect_flow(_flow(attack_flow), flow_key="k", explain=False)
        assert len(repo.created) == 1
        assert repo.created[0].attack_type == "DoS"

    @pytest.mark.asyncio
    async def test_zero_day_benign_is_persisted(self, benign_flow: dict) -> None:
        engines = FakeEngineBundle(ml_pred="Normal", ml_conf=0.95, zero_day=True)
        repo = FakeRepository()
        service = _make_service(engines, repo=repo)
        flow_key = "warm-zd"
        for _ in range(FlowWindowStore.DEFAULT_SEQ_LEN):
            await service.detect_flow(_flow(benign_flow), flow_key=flow_key, explain=False)
        assert len(repo.created) == 1

    @pytest.mark.asyncio
    async def test_repository_failure_raises_detection_unavailable(self, attack_flow: dict) -> None:
        engines = FakeEngineBundle(ml_pred="DoS", ml_conf=0.95)
        repo = FakeRepository(fail=True)
        service = _make_service(engines, repo=repo)
        with pytest.raises(DetectionUnavailableError):
            await service.detect_flow(_flow(attack_flow), flow_key="k", explain=False)


class TestNotificationOrdering:
    @pytest.mark.asyncio
    async def test_notification_fires_after_persist_with_real_alert_id(self, attack_flow: dict) -> None:
        engines = FakeEngineBundle(ml_pred="DoS", ml_conf=0.95)
        repo = FakeRepository()
        notifier = SpyNotifier()
        service = _make_service(engines, repo=repo, notifier=notifier)
        await service.detect_flow(_flow(attack_flow), flow_key="k", explain=False)
        assert len(notifier.sent) == 1
        assert notifier.sent[0].alert_id is not None

    @pytest.mark.asyncio
    async def test_benign_low_severity_does_not_notify(self, benign_flow: dict) -> None:
        engines = FakeEngineBundle(ml_pred="Normal", ml_conf=0.95)
        notifier = SpyNotifier()
        service = _make_service(engines, notifier=notifier)
        await service.detect_flow(_flow(benign_flow), flow_key="k", explain=False)
        assert len(notifier.sent) == 0

    @pytest.mark.asyncio
    async def test_notifier_exception_does_not_fail_the_request(self, attack_flow: dict) -> None:
        # A notifier bug must not turn an already-persisted detection into a
        # 500 for the caller (see DEV1_PROGRESS.md review findings).
        engines = FakeEngineBundle(ml_pred="DoS", ml_conf=0.95)
        repo = FakeRepository()
        service = _make_service(engines, repo=repo, notifier=RaisingNotifier())
        response = await service.detect_flow(_flow(attack_flow), flow_key="k", explain=False)
        assert response.prediction == "DoS"
        assert len(repo.created) == 1

    @pytest.mark.asyncio
    async def test_background_tasks_schedules_notify_instead_of_awaiting_inline(self, attack_flow: dict) -> None:
        # Per the original design (Phase 4) and the Phase 9 review finding:
        # SMTP/Slack latency must never ride on the /detect response. When a
        # BackgroundTasks-like object is passed, notify() must be scheduled
        # via add_task, not awaited inline -- so it hasn't run by the time
        # detect_flow returns.
        engines = FakeEngineBundle(ml_pred="DoS", ml_conf=0.95)
        repo = FakeRepository()
        notifier = SpyNotifier()
        service = _make_service(engines, repo=repo, notifier=notifier)
        background_tasks = FakeBackgroundTasks()

        response = await service.detect_flow(
            _flow(attack_flow), flow_key="k", explain=False, background_tasks=background_tasks
        )

        assert response.prediction == "DoS"
        assert len(repo.created) == 1
        assert len(notifier.sent) == 0  # not yet run -- only scheduled
        assert len(background_tasks.tasks) == 1
        func, args, kwargs = background_tasks.tasks[0]
        await func(*args, **kwargs)  # simulate FastAPI running it after the response
        assert len(notifier.sent) == 1

    @pytest.mark.asyncio
    async def test_no_background_tasks_still_notifies_inline(self, attack_flow: dict) -> None:
        # Backward-compatible default for non-HTTP callers (or tests) that
        # don't pass background_tasks at all.
        engines = FakeEngineBundle(ml_pred="DoS", ml_conf=0.95)
        notifier = SpyNotifier()
        service = _make_service(engines, notifier=notifier)
        await service.detect_flow(_flow(attack_flow), flow_key="k", explain=False, background_tasks=None)
        assert len(notifier.sent) == 1


class TestAttackProbabilitySource:
    @pytest.mark.asyncio
    async def test_uses_ml_confidence_when_ml_drives_a_disagreeing_verdict(self, attack_flow: dict) -> None:
        # ML sees an attack, DL's warm window disagrees and says Normal with
        # a low attack prob -> verdict_source is "ml", so attack_probability
        # must be derived from ml_conf, not DL's disagreeing low probability
        # (a real bug caught in review: it was previously keyed off window
        # warmth instead of verdict_source, so it silently returned DL's
        # number even when ML's verdict won).
        engines = FakeEngineBundle(
            ml_pred="DoS",
            ml_conf=0.95,
            dl_result={"dl_prediction": "Normal", "dl_confidence": 0.9, "dl_stage1_attack_prob": 0.05},
        )
        service = _make_service(engines)
        flow_key = "ml-drives-warm"
        for _ in range(FlowWindowStore.DEFAULT_SEQ_LEN):
            response = await service.detect_flow(_flow(attack_flow), flow_key=flow_key, explain=False)
        assert response.prediction == "DoS"
        assert response.attack_probability == pytest.approx(0.95)

    @pytest.mark.asyncio
    async def test_uses_dl_attack_prob_when_dl_drives_the_verdict(self, benign_flow: dict) -> None:
        engines = FakeEngineBundle(
            ml_pred="Normal",
            ml_conf=0.9,
            dl_result={"dl_prediction": "DoS", "dl_confidence": 0.8, "dl_stage1_attack_prob": 0.77},
        )
        service = _make_service(engines)
        flow_key = "dl-drives-warm"
        for _ in range(FlowWindowStore.DEFAULT_SEQ_LEN):
            response = await service.detect_flow(_flow(benign_flow), flow_key=flow_key, explain=False)
        assert response.attack_probability == pytest.approx(0.77)
