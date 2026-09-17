"""SHAP/LIME explainability for the unified predictor (ML ensemble + DL Transformer + VAE).

Pure value types and ranking/aggregation helpers live here with no model or
SHAP/LIME dependency, so they're testable without any artifacts. Explainer
classes that actually call shap/lime against the trained models are added
alongside their own tests (see DEV1_PROGRESS.md Phase 7).

Dependency direction: this module must not import from service/detection_api.
It returns plain, frozen dataclasses; the API layer adapts them into the
frozen PredictionResponse/ShapExplanation Pydantic contract.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

import numpy as np

AggregationStrategy = Literal["sum_time", "mean_abs", "last_step"]

_DEFAULT_TOP_K = 10


class ExplanationMethod(str, Enum):
    SHAP_TREE = "shap_tree"
    SHAP_DEEP = "shap_deep"
    LIME = "lime"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class FeatureAttribution:
    feature: str
    value: float | int | str
    shap_value: float
    direction: Literal["attack", "benign"]


@dataclass(frozen=True, slots=True)
class ExplanationResult:
    base_value: float
    predicted_value: float
    top_features: tuple[FeatureAttribution, ...]
    all_shap_values: tuple[float, ...] | None
    feature_names: tuple[str, ...] | None
    method: ExplanationMethod
    elapsed_ms: float
    failure_reason: str | None = None

    @classmethod
    def unavailable(cls, reason: str, elapsed_ms: float = 0.0) -> "ExplanationResult":
        return cls(
            base_value=0.0,
            predicted_value=0.0,
            top_features=(),
            all_shap_values=None,
            feature_names=None,
            method=ExplanationMethod.UNAVAILABLE,
            elapsed_ms=elapsed_ms,
            failure_reason=reason,
        )


def direction_of(shap_value: float, *, predicted_is_attack: bool) -> Literal["attack", "benign"]:
    """Whether this feature's contribution pushes toward maliciousness.

    Always answers "did this feature push toward attack", independent of
    which class was actually predicted — this is the sign convention the
    dashboard's SHAP waterfall depends on.
    """
    signed_toward_attack = shap_value if predicted_is_attack else -shap_value
    return "attack" if signed_toward_attack > 0 else "benign"


def rank_attributions(
    values: Sequence[float],
    names: Sequence[str],
    raw_values: Mapping[str, Any],
    *,
    predicted_is_attack: bool,
    top_k: int = 10,
    transformed_values: Sequence[Any] | None = None,
) -> tuple[FeatureAttribution, ...]:
    """Rank features by |shap value| and attach a human-readable value.

    `value` prefers the raw, pre-encoding sample value (e.g. proto="tcp")
    so the dashboard doesn't render an encoded integer. Falls back to the
    transformed numeric value, then to the shap value itself, when the raw
    key isn't available (e.g. a derived feature with no raw counterpart).
    """
    names = list(names)
    indexed = sorted(zip(names, values), key=lambda pair: abs(pair[1]), reverse=True)
    top = indexed[:top_k]

    attributions = []
    for name, shap_value in top:
        if name in raw_values:
            feature_value: float | int | str = raw_values[name]
        elif transformed_values is not None:
            feature_value = transformed_values[names.index(name)]
        else:
            feature_value = float(shap_value)
        attributions.append(
            FeatureAttribution(
                feature=name,
                value=feature_value,
                shap_value=float(shap_value),
                direction=direction_of(shap_value, predicted_is_attack=predicted_is_attack),
            )
        )
    return tuple(attributions)


def aggregate_sequence(shap_values: np.ndarray, *, strategy: AggregationStrategy = "sum_time") -> np.ndarray:
    """Collapse a (seq_len, n_features) SHAP matrix into one (n_features,) vector.

    "sum_time" preserves SHAP additivity (base + sum(aggregated) ~= predicted),
    which is why it's the default — it gives tests a numeric oracle instead of
    just a shape check, and reads as "total contribution across the window".
    """
    if strategy == "sum_time":
        return shap_values.sum(axis=0)
    if strategy == "mean_abs":
        return np.abs(shap_values).mean(axis=0)
    if strategy == "last_step":
        return shap_values[-1]
    raise ValueError(f"unknown aggregation strategy: {strategy!r}")


def group_one_hot(
    values: np.ndarray,
    names: Sequence[str],
    raw_sample: Mapping[str, Any],
    prefixes: Sequence[str] = ("proto_", "service_", "state_"),
) -> tuple[np.ndarray, list[str], dict[str, str]]:
    """Collapse one-hot columns (e.g. proto_tcp, proto_udp, ...) into one logical feature.

    Far more readable in a SHAP waterfall than ~150 near-zero one-hot
    attributions. The collapsed attribution is the sum of its constituent
    one-hot shap values (additivity-preserving), and its displayed `value`
    is the active category from the raw sample.
    """
    values = np.asarray(values, dtype=float)
    new_values: list[float] = []
    new_names: list[str] = []
    value_map: dict[str, str] = {}
    seen_logical: dict[str, int] = {}

    for i, name in enumerate(names):
        matched_prefix = next((p for p in prefixes if name.startswith(p)), None)
        if matched_prefix is None:
            new_names.append(name)
            new_values.append(float(values[i]))
            continue

        logical = matched_prefix.rstrip("_")
        category = name[len(matched_prefix) :]
        if logical in seen_logical:
            new_values[seen_logical[logical]] += float(values[i])
        else:
            seen_logical[logical] = len(new_names)
            new_names.append(logical)
            new_values.append(float(values[i]))
            value_map[logical] = str(raw_sample.get(logical, category))

    return np.array(new_values), new_names, value_map


def safe_explain(
    factory: Callable[[], ExplanationResult],
    *,
    label: str,
    logger: logging.Logger,
    budget_s: float | None = None,
) -> ExplanationResult:
    """Run an explainer factory, guaranteeing a result instead of a raised exception.

    Never catches KeyboardInterrupt/SystemExit — only genuine explainer
    failures (bad input, unsupported op, numerical error, etc.) are turned
    into ExplanationResult.unavailable(...). This is what lets /detect
    return HTTP 200 with explanation_method="unavailable" instead of 500
    when SHAP/LIME itself fails.
    """
    start = time.perf_counter()
    try:
        result = factory()
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.exception("explanation failed for %s", label)
        return ExplanationResult.unavailable(f"{type(exc).__name__}: {exc}", elapsed_ms=elapsed_ms)

    elapsed_ms = (time.perf_counter() - start) * 1000
    if budget_s is not None and elapsed_ms / 1000 > budget_s:
        logger.warning(
            "explanation for %s exceeded budget: %.1fms > %.1fms", label, elapsed_ms, budget_s * 1000
        )
    return result


def _class_slice(shap_values: np.ndarray, class_index: int) -> np.ndarray:
    """Normalize a shap TreeExplainer/GradientExplainer array to one class's values.

    shap's array layout for multiclass output is (n_samples, ..., n_classes)
    once a class axis is present; for a single sample this collapses to
    just the per-feature (or per-timestep-per-feature) values for that class.
    """
    if shap_values.ndim >= 2 and shap_values.shape[-1] > 1:
        return shap_values[0, ..., class_index]
    return shap_values[0, ...]


class TreeEnsembleExplainer:
    """SHAP TreeExplainer over the XGBoost+RF+LightGBM soft-voting ensemble.

    Uses model_output="raw" uniformly for all three models (see
    DEV1_PROGRESS.md Phase 0: shap 0.51 has no objective mapping for
    XGBoost's multiclass multi:softprob, so probability mode fails for that
    model specifically; RF's raw output already equals exact probability,
    LightGBM/XGBoost raw is log-odds/margin space). The averaged attribution
    is used for ranking/display; additivity is exact per-model in that
    model's own output space, not as one unified cross-model formula.
    """

    def __init__(self, *, xgb, rf, lgbm, feature_names: Sequence[str], top_k: int = _DEFAULT_TOP_K):
        import shap

        self._explainers = {
            "xgb": shap.TreeExplainer(xgb, model_output="raw"),
            "rf": shap.TreeExplainer(rf, model_output="raw"),
            "lgbm": shap.TreeExplainer(lgbm, model_output="raw"),
        }
        self._feature_names = list(feature_names)
        self._top_k = top_k

    def explain(
        self,
        *,
        transformed_row: np.ndarray,
        raw_sample: Mapping[str, Any],
        class_index: int,
        predicted_is_attack: bool,
    ) -> ExplanationResult:
        start = time.perf_counter()

        per_model_values = []
        per_model_base = []
        for explainer in self._explainers.values():
            shap_values = np.asarray(explainer.shap_values(transformed_row))
            per_model_values.append(_class_slice(shap_values, class_index))
            expected = np.asarray(explainer.expected_value)
            per_model_base.append(float(expected[class_index]) if expected.ndim else float(expected))

        avg_values = np.mean(per_model_values, axis=0)
        avg_base = float(np.mean(per_model_base))
        predicted_value = avg_base + float(avg_values.sum())

        top_features = rank_attributions(
            avg_values.tolist(),
            self._feature_names,
            raw_sample,
            predicted_is_attack=predicted_is_attack,
            top_k=self._top_k,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        return ExplanationResult(
            base_value=avg_base,
            predicted_value=predicted_value,
            top_features=top_features,
            all_shap_values=tuple(float(v) for v in avg_values),
            feature_names=tuple(self._feature_names),
            method=ExplanationMethod.SHAP_TREE,
            elapsed_ms=elapsed_ms,
        )


class DeepSequenceExplainer:
    """SHAP GradientExplainer over the Transformer Stage1/Stage2 classifiers.

    Uses GradientExplainer, never DeepExplainer — DeepExplainer's PyTorch
    backend has no reliable handler for nn.TransformerEncoder's fused
    attention/LayerNorm internals under torch 2.x (confirmed failing in the
    Phase 0 spike: "SHAP explanations do not sum up to the model's output").
    base_value approximates shap's expected-gradients baseline as the mean
    predicted probability over the background set.
    """

    def __init__(
        self,
        *,
        model,
        feature_names: Sequence[str],
        background,
        nsamples: int = 100,
        top_k: int = _DEFAULT_TOP_K,
        aggregation: AggregationStrategy = "sum_time",
    ):
        import shap

        self._model = model
        self._feature_names = list(feature_names)
        self._background = background
        self._nsamples = nsamples
        self._top_k = top_k
        self._aggregation = aggregation
        self._explainer = shap.GradientExplainer(model, background)

    def explain(
        self,
        *,
        window,
        class_index: int,
        raw_sample: Mapping[str, Any],
        predicted_is_attack: bool,
    ) -> ExplanationResult:
        import torch

        start = time.perf_counter()
        shap_values = self._explainer.shap_values(window, nsamples=self._nsamples)
        per_step = _class_slice(np.asarray(shap_values), class_index)  # (seq_len, n_features)
        aggregated = aggregate_sequence(per_step, strategy=self._aggregation)
        grouped_values, grouped_names, value_map = group_one_hot(aggregated, self._feature_names, raw_sample)
        # group_one_hot's value_map only covers the 3 collapsed categorical
        # names (proto/service/state); merge in raw_sample so the ~42
        # passthrough numeric features also get their real value instead of
        # falling through to transformed_values (which duplicates shap_value —
        # a real bug caught in review, see DEV1_PROGRESS.md).
        raw_values = {**raw_sample, **value_map}

        # GradientExplainer computes gradients of the model's raw forward()
        # output (logits), not predict_proba() — base_value/predicted_value
        # must be in the same (logit) space as the shap values for additivity
        # to hold at all. Comparing against predict_proba here was a real
        # bug caught during implementation (see DEV1_PROGRESS.md Phase 7):
        # softmax probabilities and pre-softmax logits differ by orders of
        # magnitude, which showed up as a ~1.0-1.4 additivity error.
        with torch.no_grad():
            predicted_value = float(self._model(window)[0, class_index].item())
            base_value = float(self._model(self._background)[:, class_index].mean().item())

        top_features = rank_attributions(
            grouped_values.tolist(),
            grouped_names,
            raw_values,
            predicted_is_attack=predicted_is_attack,
            top_k=self._top_k,
            transformed_values=grouped_values.tolist(),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        return ExplanationResult(
            base_value=base_value,
            predicted_value=predicted_value,
            top_features=top_features,
            all_shap_values=tuple(float(v) for v in grouped_values),
            feature_names=tuple(grouped_names),
            method=ExplanationMethod.SHAP_DEEP,
            elapsed_ms=elapsed_ms,
        )


class LimeSequenceExplainer:
    """LIME over a (10, n_features) sequence window.

    Used for the VAE AE (always — model-agnostic, explains reconstruction
    error rather than a classification probability) and as the DL fallback
    when GradientExplainer fails. LIME can't represent a sequence natively,
    so this perturbs only the most recent timestep, freezing the preceding
    9 — "given recent history, what about *this* flow drives the score,"
    the right question for an IDS, and it bounds perturbation to n_features
    dims instead of seq_len * n_features.
    """

    def __init__(
        self,
        *,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        feature_names: Sequence[str],
        background: np.ndarray,
        num_samples: int = 200,
        top_k: int = _DEFAULT_TOP_K,
        seed: int = 1337,
    ):
        from lime.lime_tabular import LimeTabularExplainer

        self._predict_fn = predict_fn
        self._feature_names = list(feature_names)
        self._num_samples = num_samples
        self._top_k = top_k
        self._explainer = LimeTabularExplainer(
            training_data=background, mode="regression", discretize_continuous=False, random_state=seed
        )

    def explain(self, *, last_step: np.ndarray, raw_sample: Mapping[str, Any]) -> ExplanationResult:
        start = time.perf_counter()
        explanation = self._explainer.explain_instance(
            last_step, self._predict_fn, num_features=len(self._feature_names), num_samples=self._num_samples
        )
        # Regression mode conventionally keys local_exp/intercept under label 1
        # (matches LimeTabularExplainer.as_list()'s sign convention).
        local_exp = dict(explanation.local_exp[1])
        base_value = float(explanation.intercept[1])
        raw_values = np.array([local_exp.get(i, 0.0) for i in range(len(self._feature_names))])

        grouped_values, grouped_names, value_map = group_one_hot(raw_values, self._feature_names, raw_sample)
        predicted_value = float(self._predict_fn(last_step.reshape(1, -1))[0])

        # See DeepSequenceExplainer.explain(): value_map alone only covers
        # the 3 collapsed categorical names, so merge raw_sample in for the
        # passthrough numeric features' real (not shap-value-duplicated) value.
        merged_raw_values = {**raw_sample, **value_map}
        top_features = rank_attributions(
            grouped_values.tolist(),
            grouped_names,
            merged_raw_values,
            predicted_is_attack=True,
            top_k=self._top_k,
            transformed_values=grouped_values.tolist(),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        return ExplanationResult(
            base_value=base_value,
            predicted_value=predicted_value,
            top_features=top_features,
            all_shap_values=tuple(float(v) for v in grouped_values),
            feature_names=tuple(grouped_names),
            method=ExplanationMethod.LIME,
            elapsed_ms=elapsed_ms,
        )
