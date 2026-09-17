"""Orchestrates predict -> explain -> persist -> notify for a single flow.

Integration mode: in-process import of the worker's engine classes (not
subprocess) — SHAP/LIME need live access to model gradients/tree internals,
which a stdin/stdout JSON pipe can't carry. combined_verdict() is imported
and reused verbatim from unified_predictor_worker.py since it's pure and
forking that OR-gate logic would risk drift from the real decision path.
"""
from __future__ import annotations

import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

import anyio
import numpy as np

from models.src.explain import ExplanationResult, safe_explain
from service.detection_api.src.models.schemas import (
    Alert,
    AlertCreate,
    AlertRepository,
    FlowInput,
    PredictionResponse,
    SEVERITY_LEVELS,
    ShapExplanation,
    TopFeature,
    derive_severity,
)
from service.detection_api.src.services.alerting import AlertNotification, Notifier

MODEL_NAMES: dict[str, str] = {
    "ml": "ml_ensemble_xgb_rf_lgbm",
    "dl": "dl_transformer_stage2",
    "both": "unified_ensemble",
}

_PROTOCOL_NUMBERS: dict[str, int] = {"tcp": 6, "udp": 17, "icmp": 1}


class DetectionUnavailableError(RuntimeError):
    """A downstream dependency (repository, engines) is unavailable.

    Routes must map this to HTTP 503 — never silently swallow it, since a
    dropped security alert on a storage failure is a real failure.
    """


class EngineBundle(Protocol):
    def predict_ml(self, sample: dict[str, Any]) -> tuple[str, float]: ...
    def preprocess_dl(self, sample: dict[str, Any]) -> np.ndarray: ...
    def predict_window(self, window: np.ndarray) -> dict: ...
    def score_ae(self, window: np.ndarray) -> tuple[float, bool]: ...


class BackgroundTaskRunner(Protocol):
    """Matches fastapi.BackgroundTasks' interface without importing FastAPI
    here — detection.py stays framework-agnostic and independently testable."""

    def add_task(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None: ...


class UnifiedEngineBundle:
    """Concrete EngineBundle backed by the real trained models.

    In-process import of unified_predictor_worker's engine classes (see
    module docstring for why subprocess integration was rejected).
    DLAdvancedEngine's own internal sliding-window buffer is never used here
    — FlowWindowStore replaces it with a bounded, per-flow-key window, so
    only DLAdvancedEngine's preprocessor/stage1/stage2 are reused via
    predict_from_tensor(), never its own predict()/push_flow().

    ml_transform() and ml_label_classes are exposed beyond the EngineBundle
    Protocol specifically for ExplanationService, which needs the same
    ML-side transformed row and label space that MLBaselineEngine.predict()
    used internally to produce ml_pred.
    """

    def __init__(self, *, device=None) -> None:
        import torch

        from models.src.unified_predictor_worker import AEAnomalyEngine, DLAdvancedEngine, MLBaselineEngine

        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._ml = MLBaselineEngine()
        self._dl = DLAdvancedEngine(self._device)
        self._ae = AEAnomalyEngine(self._device)

    def predict_ml(self, sample: dict[str, Any]) -> tuple[str, float]:
        return self._ml.predict(sample)

    def preprocess_dl(self, sample: dict[str, Any]) -> np.ndarray:
        import pandas as pd

        return self._dl.preprocessor.transform(pd.DataFrame([sample]))[0]

    def predict_window(self, window: np.ndarray) -> dict:
        return self._dl.predict_from_tensor(self._to_tensor(window))

    def score_ae(self, window: np.ndarray) -> tuple[float, bool]:
        return self._ae.score(self._to_tensor(window))

    def ml_transform(self, sample: dict[str, Any]) -> np.ndarray:
        return self._ml._preprocess(sample)

    @property
    def ml_label_classes(self) -> list[str]:
        return list(self._ml.label_encoder.classes_)

    @property
    def dl_feature_names(self) -> list[str]:
        import json

        from models.src.unified_predictor_worker import PREPROC_DIR

        data = json.loads((PREPROC_DIR / "feature_list_unsw_v1_20260612.json").read_text())
        return data if isinstance(data, list) else data["feature_columns"]

    @property
    def selected_ml_feature_names(self) -> list[str]:
        from service.detection_api.src.models.schemas import CANONICAL_FEATURE_COLUMNS

        support = self._ml.selector.get_support()
        return [name for name, keep in zip(CANONICAL_FEATURE_COLUMNS, support, strict=True) if keep]

    @property
    def ml_models(self) -> tuple[Any, Any, Any]:
        """(xgb, rf, lgbm) — for constructing a TreeEnsembleExplainer."""
        return self._ml.xgb, self._ml.rf, self._ml.lgbm

    @property
    def stage1_model(self) -> Any:
        """For constructing a DeepSequenceExplainer / LIME fallback over Stage 1's binary gate."""
        return self._dl.stage1

    @property
    def ae_model(self) -> Any:
        """For constructing a LIME explainer over the AE's anomaly_score()."""
        return self._ae.model

    def _to_tensor(self, window: np.ndarray):
        import torch

        return torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self._device)


@dataclass(frozen=True)
class DetectionConfig:
    explanation_budget_s: float = 8.0
    min_notify_severity: str = "high"


class FlowWindowStore:
    """Bounded (LRU + TTL) per-flow-key sliding window.

    Replaces DLAdvancedEngine's single global deque, which is wrong under
    concurrent HTTP clients (interleaved requests would corrupt each
    other's temporal context) and has unbounded lifetime. The size/TTL
    bound is a security requirement: an unbounded dict keyed on
    attacker-supplied IPs is a memory-exhaustion vector.

    Concurrency invariant: this class has no internal lock and must not
    have one — it relies on always being called from the event-loop thread,
    never from inside an anyio.to_thread.run_sync() call, with no `await`
    between a push() and the paired window() in DetectionService.detect_flow
    (asyncio coroutines only yield control at `await`, so that pairing is
    atomic w.r.t. other coroutines on the same loop). The genuinely
    CPU-expensive engine/explainer calls are offloaded to threads instead;
    this store stays cheap, in-process, and single-threaded by design.
    """

    DEFAULT_SEQ_LEN = 10
    DEFAULT_MAX_KEYS = 1000
    DEFAULT_TTL_S = 300.0

    def __init__(
        self,
        *,
        seq_len: int = DEFAULT_SEQ_LEN,
        max_keys: int = DEFAULT_MAX_KEYS,
        ttl_s: float = DEFAULT_TTL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._seq_len = seq_len
        self._max_keys = max_keys
        self._ttl_s = ttl_s
        self._clock = clock
        self._buffers: OrderedDict[str, deque[np.ndarray]] = OrderedDict()
        self._last_seen: dict[str, float] = {}

    def _evict_expired(self) -> None:
        now = self._clock()
        expired = [key for key, seen in self._last_seen.items() if now - seen > self._ttl_s]
        for key in expired:
            self._buffers.pop(key, None)
            self._last_seen.pop(key, None)

    def push(self, key: str, vector: np.ndarray) -> None:
        self._evict_expired()
        if key not in self._buffers and len(self._buffers) >= self._max_keys:
            oldest_key, _ = self._buffers.popitem(last=False)
            self._last_seen.pop(oldest_key, None)
        if key not in self._buffers:
            self._buffers[key] = deque(maxlen=self._seq_len)
        self._buffers.move_to_end(key)
        self._buffers[key].append(vector)
        self._last_seen[key] = self._clock()

    def window(self, key: str) -> np.ndarray | None:
        self._evict_expired()
        buffer = self._buffers.get(key)
        if buffer is None or len(buffer) < self._seq_len:
            return None
        self._buffers.move_to_end(key)
        self._last_seen[key] = self._clock()
        return np.stack(list(buffer), axis=0)

    def stats(self) -> dict[str, int]:
        return {
            "tracked_keys": len(self._buffers),
            "warm_keys": sum(1 for buf in self._buffers.values() if len(buf) >= self._seq_len),
        }


def _severity_rank(severity: str) -> int:
    return SEVERITY_LEVELS.index(severity) if severity in SEVERITY_LEVELS else -1


def _to_shap_explanation(result: ExplanationResult) -> ShapExplanation:
    return ShapExplanation(
        base_value=result.base_value,
        predicted_value=result.predicted_value,
        top_features=[
            TopFeature(feature=f.feature, value=f.value, shap_value=f.shap_value, direction=f.direction)
            for f in result.top_features
        ],
        all_shap_values=list(result.all_shap_values) if result.all_shap_values is not None else None,
        feature_names=list(result.feature_names) if result.feature_names is not None else None,
        explanation_method=result.method.value,
    )


def _protocol_number(proto: str) -> int:
    return _PROTOCOL_NUMBERS.get(proto.lower(), 0)


class DetectionService:
    def __init__(
        self,
        *,
        engines: EngineBundle,
        repo: AlertRepository,
        notifier: Notifier,
        explain_fn: Callable[[dict[str, Any]], ExplanationResult] | None = None,
        config: DetectionConfig | None = None,
        window_store: FlowWindowStore | None = None,
    ) -> None:
        self._engines = engines
        self._repo = repo
        self._notifier = notifier
        self._explain_fn = explain_fn
        self._config = config or DetectionConfig()
        self._windows = window_store or FlowWindowStore()

    async def detect_flow(
        self,
        flow: FlowInput,
        *,
        flow_key: str,
        explain: bool = True,
        background_tasks: BackgroundTaskRunner | None = None,
    ) -> PredictionResponse:
        from models.src.unified_predictor_worker import combined_verdict

        sample = flow.to_sample()
        # Engine calls are genuinely CPU-expensive (PyTorch/XGBoost/sklearn)
        # once EngineBundle is a real UnifiedEngineBundle rather than a fast
        # test fake — offloaded to a thread so one request's inference can't
        # stall the event loop for every other concurrent caller.
        ml_pred, ml_conf = await anyio.to_thread.run_sync(self._engines.predict_ml, sample)

        dl_vector = await anyio.to_thread.run_sync(self._engines.preprocess_dl, sample)
        # FlowWindowStore itself always stays on this event-loop thread (never
        # inside a to_thread.run_sync call) with no await between push() and
        # window() — that's what keeps concurrent requests from interleaving
        # on the same flow_key without needing a lock; see FlowWindowStore's
        # docstring.
        self._windows.push(flow_key, dl_vector)
        window = self._windows.window(flow_key)

        if window is not None:
            dl_result = await anyio.to_thread.run_sync(self._engines.predict_window, window)
            ae_score, zero_day = await anyio.to_thread.run_sync(self._engines.score_ae, window)
        else:
            dl_result = {"dl_prediction": "warming_up", "dl_confidence": 0.0, "dl_stage1_attack_prob": 0.0}
            ae_score, zero_day = None, False

        verdict = combined_verdict(ml_pred, ml_conf, dl_result)
        prediction = "BENIGN" if verdict["prediction"] == "Normal" else verdict["prediction"]
        model_name = MODEL_NAMES.get(verdict["verdict_source"], verdict["verdict_source"])
        # Source attack_probability from whichever engine actually produced the
        # verdict (verdict_source), not merely from window warmth — otherwise a
        # disagreement (e.g. ML drives an attack verdict while DL's window is
        # warm but says Normal) returns a probability contradicting confidence.
        attack_probability = (
            dl_result["dl_stage1_attack_prob"]
            if verdict["verdict_source"] in ("dl", "both")
            else (ml_conf if prediction != "BENIGN" else 1.0 - ml_conf)
        )

        explanation: ExplanationResult | None = None
        if explain and self._explain_fn is not None:
            ctx = {
                "sample": sample,
                "prediction": prediction,
                "verdict_source": verdict["verdict_source"],
                "window": window,
                "ml_prediction": ml_pred,
                "zero_day_flag": zero_day,
            }

            def _run_explain() -> ExplanationResult:
                return safe_explain(
                    lambda: self._explain_fn(ctx),
                    label="detect_flow",
                    logger=_logger(),
                    budget_s=self._config.explanation_budget_s,
                )

            # SHAP/LIME are the most CPU-expensive part of this method (up
            # to the 5s/10s budgets in the Definition of Done) — offloaded
            # for the same reason as the engine calls above.
            explanation = await anyio.to_thread.run_sync(_run_explain)

        response = PredictionResponse(
            prediction=prediction,
            confidence=verdict["confidence"],
            attack_probability=attack_probability,
            model_name=model_name,
            anomaly_score=ae_score,
            zero_day_flag=zero_day,
            shap_explanation=_to_shap_explanation(explanation) if explanation is not None else None,
        )

        if prediction != "BENIGN" or zero_day:
            alert = await self._persist(flow, response)
            if background_tasks is not None:
                # Matches the original design intent (Phase 4): SMTP/Slack
                # latency must never ride on the /detect response. Callers
                # that don't pass background_tasks (existing unit tests, or
                # any non-HTTP caller) get the old inline-await behavior.
                background_tasks.add_task(self._notify, alert)
            else:
                await self._notify(alert)

        return response

    async def _persist(self, flow: FlowInput, response: PredictionResponse) -> Alert:
        severity = derive_severity(response.prediction, response.confidence, response.zero_day_flag)
        alert_create = AlertCreate(
            flow_id=flow.flow_id or f"flow-{int(time.time() * 1000)}",
            src_ip=str(flow.src_ip) if flow.src_ip else "0.0.0.0",
            dst_ip=str(flow.dst_ip) if flow.dst_ip else "0.0.0.0",
            src_port=flow.src_port or 0,
            dst_port=flow.dst_port or 0,
            protocol=_protocol_number(flow.proto),
            attack_type=response.prediction,
            severity=severity,
            confidence=response.confidence,
            model_name=response.model_name,
            shap_explanation=response.shap_explanation.model_dump() if response.shap_explanation else None,
        )

        try:
            return await self._repo.create(alert_create)
        except Exception as exc:
            # Log the real cause server-side only — a repository exception
            # (e.g. a DB driver error) can embed connection strings, hosts,
            # or query fragments that must never reach the HTTP caller, who
            # only sees DetectionUnavailableError's generic message via the
            # route's 503 mapping.
            _logger().exception("failed to persist alert")
            raise DetectionUnavailableError("detection service temporarily unavailable") from exc

    async def _notify(self, alert: Alert) -> None:
        if _severity_rank(alert.severity) < _severity_rank(self._config.min_notify_severity):
            return

        # top_features' construction is inside this try too: when notify()
        # runs as a FastAPI background task (see detect_flow), anything
        # outside this block would raise past Starlette's background-task
        # runner and be logged by the ASGI server's default handler instead
        # of this file's sanitized logger — keep every _notify failure mode
        # on the same logging path regardless of how it's invoked.
        try:
            top_features = ()
            if alert.shap_explanation:
                top_features = tuple(
                    (f["feature"], f["shap_value"]) for f in alert.shap_explanation.get("top_features", [])
                )
            await self._notifier.send(
                AlertNotification(
                    alert_id=str(alert.id),
                    flow_id=alert.flow_id,
                    severity=alert.severity,
                    attack_type=alert.attack_type,
                    confidence=alert.confidence,
                    src_ip=alert.src_ip,
                    dst_ip=alert.dst_ip,
                    detected_at=datetime.now(timezone.utc),
                    top_features=top_features,
                )
            )
        except Exception:
            # A notifier bug must not turn an already-persisted detection
            # into a 500 for the caller (or a swallowed background-task
            # exception into an unhandled crash) — the alert is saved either
            # way.
            _logger().exception("notification failed after successful persist")


def _logger():
    import logging

    return logging.getLogger(__name__)
