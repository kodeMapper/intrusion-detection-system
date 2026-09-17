"""Unit tests for explain.safe_explain — the wrapper that guarantees an
explainer failure never crashes the /detect request (HTTP 200, not 500)."""
from __future__ import annotations

import logging

import pytest

from models.src.explain import ExplanationMethod, ExplanationResult, safe_explain


def _make_logger() -> logging.Logger:
    logger = logging.getLogger("test.explain.fallback")
    logger.setLevel(logging.DEBUG)
    return logger


class TestSafeExplainCatchesFailures:
    def test_raising_factory_returns_unavailable_result(self) -> None:
        def factory() -> ExplanationResult:
            raise ValueError("model exploded")

        result = safe_explain(factory, label="tree_ensemble", logger=_make_logger())

        assert result.method == ExplanationMethod.UNAVAILABLE
        assert result.top_features == ()
        assert result.all_shap_values is None
        assert result.feature_names is None

    def test_failure_reason_contains_exception_type_and_message(self) -> None:
        def factory() -> ExplanationResult:
            raise RuntimeError("boom")

        result = safe_explain(factory, label="tree_ensemble", logger=_make_logger())

        assert result.failure_reason is not None
        assert "RuntimeError" in result.failure_reason
        assert "boom" in result.failure_reason

    def test_logs_the_exception(self, caplog: pytest.LogCaptureFixture) -> None:
        def factory() -> ExplanationResult:
            raise ValueError("logged failure")

        with caplog.at_level(logging.ERROR, logger="test.explain.fallback"):
            safe_explain(factory, label="tree_ensemble", logger=_make_logger())

        assert any("logged failure" in record.getMessage() or "tree_ensemble" in record.getMessage() for record in caplog.records) or any(
            record.exc_info for record in caplog.records
        )

    def test_keyboard_interrupt_propagates_not_swallowed(self) -> None:
        def factory() -> ExplanationResult:
            raise KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            safe_explain(factory, label="tree_ensemble", logger=_make_logger())

    def test_system_exit_propagates_not_swallowed(self) -> None:
        def factory() -> ExplanationResult:
            raise SystemExit(1)

        with pytest.raises(SystemExit):
            safe_explain(factory, label="tree_ensemble", logger=_make_logger())


class TestSafeExplainPassesThroughSuccess:
    def test_successful_result_is_returned_unmodified(self) -> None:
        expected = ExplanationResult(
            base_value=0.1,
            predicted_value=0.9,
            top_features=(),
            all_shap_values=(0.1, 0.2),
            feature_names=("a", "b"),
            method=ExplanationMethod.SHAP_TREE,
            elapsed_ms=5.0,
        )

        result = safe_explain(lambda: expected, label="tree_ensemble", logger=_make_logger())

        assert result is expected


class TestSafeExplainBudget:
    def test_over_budget_call_still_returns_result_but_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        import time

        def factory() -> ExplanationResult:
            time.sleep(0.05)
            return ExplanationResult(
                base_value=0.0,
                predicted_value=1.0,
                top_features=(),
                all_shap_values=None,
                feature_names=None,
                method=ExplanationMethod.LIME,
                elapsed_ms=50.0,
            )

        with caplog.at_level(logging.WARNING, logger="test.explain.fallback"):
            result = safe_explain(factory, label="lime_ae", logger=_make_logger(), budget_s=0.01)

        assert result.method == ExplanationMethod.LIME
        assert any(record.levelno == logging.WARNING for record in caplog.records)
