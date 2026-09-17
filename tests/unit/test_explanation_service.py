"""Tests for ExplanationService — dispatches to the right explainer by
verdict_source, with fully faked sub-explainers (no real models, no shap/lime
calls). Deferred from Phase 7 per DEV1_PROGRESS.md; this is the missing piece
that lets DetectionService's explain_fn produce a real explanation for
whichever engine actually drove the verdict, instead of always using one
explainer regardless of source."""
from __future__ import annotations

import numpy as np
import pytest

from models.src.explain import (
    ExplanationMethod,
    ExplanationResult,
    ExplanationService,
)


class FakeTreeExplainer:
    def __init__(self):
        self.calls: list[tuple[int, bool]] = []

    def explain(self, *, transformed_row, raw_sample, class_index, predicted_is_attack):
        self.calls.append((class_index, predicted_is_attack))
        return ExplanationResult(
            base_value=0.1, predicted_value=0.9, top_features=(), all_shap_values=(0.1,),
            feature_names=("f",), method=ExplanationMethod.SHAP_TREE, elapsed_ms=1.0,
        )


class FakeDeepExplainer:
    def __init__(self, *, raise_error: bool = False):
        self.raise_error = raise_error
        self.calls: list[int] = []

    def explain(self, *, window, class_index, raw_sample, predicted_is_attack):
        if self.raise_error:
            raise RuntimeError("gradient explainer blew up")
        self.calls.append(class_index)
        return ExplanationResult(
            base_value=0.0, predicted_value=1.0, top_features=(), all_shap_values=(1.0,),
            feature_names=("g",), method=ExplanationMethod.SHAP_DEEP, elapsed_ms=1.0,
        )


class FakeLimeExplainer:
    def __init__(self):
        self.calls = 0

    def explain(self, *, last_step, raw_sample):
        self.calls += 1
        return ExplanationResult(
            base_value=0.0, predicted_value=0.5, top_features=(), all_shap_values=(0.5,),
            feature_names=("l",), method=ExplanationMethod.LIME, elapsed_ms=1.0,
        )


def _ctx(**overrides) -> dict:
    base = {
        "sample": {"sttl": 64},
        "prediction": "DoS",
        "verdict_source": "ml",
        "window": None,
        "ml_prediction": "DoS",
        "zero_day_flag": False,
    }
    base.update(overrides)
    return base


def _make_service(**explainers) -> tuple[ExplanationService, dict]:
    tree = explainers.get("tree_explainer") or FakeTreeExplainer()
    lime_dl = explainers.get("lime_dl_explainer")
    ae_lime = explainers.get("ae_lime_explainer")
    service = ExplanationService(
        tree_explainer=tree,
        ml_transform=lambda sample: np.zeros((1, 5)),
        ml_label_classes=["Normal", "DoS", "Exploits"],
        deep_explainer=explainers.get("deep_explainer"),
        # LIME (unlike tree/deep) perturbs only the last timestep with the
        # rest of the window frozen via a closure baked in at construction,
        # so it can't be a long-lived shared instance across different
        # windows -- callers supply a factory that builds a fresh explainer
        # scoped to *this* window, exactly like the real wiring must.
        lime_dl_factory=(lambda window: lime_dl) if lime_dl is not None else None,
        ae_lime_factory=(lambda window: ae_lime) if ae_lime is not None else None,
    )
    return service, {"tree": tree}


class TestMlVerdictSource:
    def test_uses_tree_explainer_with_ml_class_index(self) -> None:
        service, made = _make_service()
        result = service.explain(_ctx(verdict_source="ml", ml_prediction="DoS", window=None))
        assert result.method == ExplanationMethod.SHAP_TREE
        assert made["tree"].calls == [(1, True)]  # "DoS" is index 1 in ml_label_classes

    def test_normal_ml_prediction_is_not_flagged_as_attack(self) -> None:
        service, made = _make_service()
        result = service.explain(
            _ctx(verdict_source="ml", ml_prediction="Normal", prediction="BENIGN", window=None)
        )
        assert result.method == ExplanationMethod.SHAP_TREE
        assert made["tree"].calls == [(0, False)]


class TestColdWindow:
    def test_cold_window_uses_tree_even_with_deep_explainer_configured(self) -> None:
        deep = FakeDeepExplainer()
        service, made = _make_service(deep_explainer=deep)
        result = service.explain(_ctx(verdict_source="ml", window=None))
        assert result.method == ExplanationMethod.SHAP_TREE
        assert deep.calls == []


class TestBothVerdictSource:
    def test_both_uses_tree_not_deep(self) -> None:
        # Per design: "both" is cheaper/exact via tree, deep is reserved for
        # verdict_source == "dl" specifically.
        deep = FakeDeepExplainer()
        service, made = _make_service(deep_explainer=deep)
        result = service.explain(
            _ctx(verdict_source="both", window=np.zeros((10, 198), dtype=np.float32))
        )
        assert result.method == ExplanationMethod.SHAP_TREE
        assert deep.calls == []


class TestDlVerdictSource:
    def test_warm_window_uses_deep_explainer(self) -> None:
        deep = FakeDeepExplainer()
        service, _ = _make_service(deep_explainer=deep)
        window = np.zeros((10, 198), dtype=np.float32)
        result = service.explain(_ctx(verdict_source="dl", window=window))
        assert result.method == ExplanationMethod.SHAP_DEEP
        assert deep.calls == [1]

    def test_falls_back_to_lime_when_deep_explainer_raises(self) -> None:
        deep = FakeDeepExplainer(raise_error=True)
        lime = FakeLimeExplainer()
        service, _ = _make_service(deep_explainer=deep, lime_dl_explainer=lime)
        window = np.zeros((10, 198), dtype=np.float32)
        result = service.explain(_ctx(verdict_source="dl", window=window))
        assert result.method == ExplanationMethod.LIME
        assert lime.calls == 1

    def test_propagates_when_deep_fails_and_no_lime_fallback_configured(self) -> None:
        deep = FakeDeepExplainer(raise_error=True)
        service, _ = _make_service(deep_explainer=deep)
        window = np.zeros((10, 198), dtype=np.float32)
        with pytest.raises(RuntimeError):
            service.explain(_ctx(verdict_source="dl", window=window))


class TestZeroDayBenign:
    def test_benign_zero_day_uses_ae_lime_explainer(self) -> None:
        ae_lime = FakeLimeExplainer()
        deep = FakeDeepExplainer()
        service, made = _make_service(ae_lime_explainer=ae_lime, deep_explainer=deep)
        window = np.zeros((10, 198), dtype=np.float32)
        result = service.explain(
            _ctx(verdict_source="both", prediction="BENIGN", zero_day_flag=True, window=window)
        )
        assert result.method == ExplanationMethod.LIME
        assert ae_lime.calls == 1
        assert deep.calls == []
        assert made["tree"].calls == []

    def test_attack_zero_day_does_not_use_ae_lime(self) -> None:
        # zero_day_flag alone isn't enough -- only benign+zero_day (the AE
        # canary firing independently of a supervised attack label) routes
        # to the AE's own LIME explanation.
        ae_lime = FakeLimeExplainer()
        service, made = _make_service(ae_lime_explainer=ae_lime)
        window = np.zeros((10, 198), dtype=np.float32)
        result = service.explain(
            _ctx(verdict_source="ml", prediction="DoS", zero_day_flag=True, window=window)
        )
        assert ae_lime.calls == 0
        assert result.method == ExplanationMethod.SHAP_TREE
