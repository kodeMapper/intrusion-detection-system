"""Unit tests for the pure, model-free helpers in service/models/src/explain.py.

No SHAP/LIME calls and no model artifacts here — these test the ranking,
direction, and sequence-aggregation logic that every explainer type shares.
"""
from __future__ import annotations

import numpy as np
import pytest

from models.src.explain import (
    ExplanationMethod,
    ExplanationResult,
    FeatureAttribution,
    aggregate_sequence,
    direction_of,
    group_one_hot,
    rank_attributions,
)


class TestDirectionOf:
    def test_positive_shap_toward_attack_when_predicted_attack(self) -> None:
        assert direction_of(0.5, predicted_is_attack=True) == "attack"

    def test_negative_shap_toward_benign_when_predicted_attack(self) -> None:
        assert direction_of(-0.5, predicted_is_attack=True) == "benign"

    def test_positive_shap_toward_benign_when_predicted_benign(self) -> None:
        # A feature pushing the (benign) prediction higher is pushing away from attack.
        assert direction_of(0.5, predicted_is_attack=False) == "benign"

    def test_negative_shap_toward_attack_when_predicted_benign(self) -> None:
        assert direction_of(-0.5, predicted_is_attack=False) == "attack"

    def test_zero_shap_value_is_benign_by_convention(self) -> None:
        assert direction_of(0.0, predicted_is_attack=True) == "benign"


class TestRankAttributions:
    def test_orders_by_absolute_shap_value_descending(self) -> None:
        values = [0.1, -0.9, 0.5]
        names = ["a", "b", "c"]
        raw = {"a": 1, "b": 2, "c": 3}
        ranked = rank_attributions(values, names, raw, predicted_is_attack=True, top_k=10)
        assert [f.feature for f in ranked] == ["b", "c", "a"]

    def test_truncates_to_top_k(self) -> None:
        values = list(np.linspace(0.01, 1.0, 20))
        names = [f"f{i}" for i in range(20)]
        raw = {n: i for i, n in enumerate(names)}
        ranked = rank_attributions(values, names, raw, predicted_is_attack=True, top_k=10)
        assert len(ranked) == 10
        # top_k=10 should keep the 10 largest-magnitude values (last 10 of the ascending list)
        assert ranked[0].feature == "f19"

    def test_value_comes_from_raw_sample_not_encoded_number(self) -> None:
        values = [0.9]
        names = ["proto"]
        raw = {"proto": "tcp"}
        ranked = rank_attributions(values, names, raw, predicted_is_attack=True, top_k=10)
        assert ranked[0].value == "tcp"

    def test_falls_back_to_transformed_value_when_raw_key_missing(self) -> None:
        values = [0.9]
        names = ["derived_feature"]
        ranked = rank_attributions(values, names, {}, predicted_is_attack=True, top_k=10, transformed_values=[42.0])
        assert ranked[0].value == 42.0

    def test_each_attribution_is_a_feature_attribution(self) -> None:
        ranked = rank_attributions([0.3], ["x"], {"x": 1}, predicted_is_attack=False, top_k=10)
        assert isinstance(ranked[0], FeatureAttribution)
        assert ranked[0].direction in ("attack", "benign")


class TestAggregateSequence:
    def test_sum_time_preserves_additivity_across_timesteps(self) -> None:
        # (seq_len=3, features=2)
        per_step = np.array([[1.0, -1.0], [2.0, 0.5], [-0.5, 1.5]])
        aggregated = aggregate_sequence(per_step, strategy="sum_time")
        assert aggregated.shape == (2,)
        np.testing.assert_allclose(aggregated, per_step.sum(axis=0))

    def test_mean_abs_strategy_is_nonnegative(self) -> None:
        per_step = np.array([[1.0, -1.0], [2.0, 0.5], [-0.5, 1.5]])
        aggregated = aggregate_sequence(per_step, strategy="mean_abs")
        assert aggregated.shape == (2,)
        assert np.all(aggregated >= 0)

    def test_last_step_returns_final_timestep_only(self) -> None:
        per_step = np.array([[1.0, -1.0], [2.0, 0.5], [-0.5, 1.5]])
        aggregated = aggregate_sequence(per_step, strategy="last_step")
        np.testing.assert_allclose(aggregated, per_step[-1])

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(ValueError):
            aggregate_sequence(np.zeros((2, 2)), strategy="bogus")  # type: ignore[arg-type]


class TestGroupOneHot:
    def test_collapses_one_hot_columns_into_single_logical_feature(self) -> None:
        names = ["dur", "proto_tcp", "proto_udp", "proto_icmp"]
        values = np.array([0.1, 0.4, 0.0, 0.0])
        raw_sample = {"proto": "tcp", "dur": 0.1}
        new_values, new_names, value_map = group_one_hot(values, names, raw_sample, prefixes=("proto_",))
        assert "proto" in new_names
        assert "proto_tcp" not in new_names
        assert "dur" in new_names
        proto_idx = new_names.index("proto")
        # collapsed attribution should equal the sum of the constituent one-hot shap values
        assert new_values[proto_idx] == pytest.approx(0.4)
        assert value_map["proto"] == "tcp"

    def test_no_prefixes_present_leaves_input_unchanged(self) -> None:
        names = ["dur", "sbytes"]
        values = np.array([0.1, 0.2])
        new_values, new_names, _ = group_one_hot(values, names, {}, prefixes=("proto_",))
        assert new_names == names
        np.testing.assert_allclose(new_values, values)


class TestExplanationResult:
    def test_unavailable_factory_has_no_features_and_records_reason(self) -> None:
        result = ExplanationResult.unavailable("boom")
        assert result.method == ExplanationMethod.UNAVAILABLE
        assert result.top_features == ()
        assert result.all_shap_values is None
        assert result.feature_names is None
        assert result.failure_reason == "boom"

    def test_is_frozen(self) -> None:
        result = ExplanationResult.unavailable("boom")
        with pytest.raises(Exception):
            result.base_value = 1.0  # type: ignore[misc]
