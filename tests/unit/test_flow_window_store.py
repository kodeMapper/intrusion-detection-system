"""Unit tests for FlowWindowStore — the bounded (LRU + TTL) per-flow-key
sliding window that replaces DLAdvancedEngine's single global deque for the
stateless HTTP API. The size/TTL bound is a security requirement: an
unbounded dict keyed on attacker-supplied IPs is a memory-exhaustion vector."""
from __future__ import annotations

import numpy as np

from service.detection_api.src.services.detection import FlowWindowStore


def _vec(value: float) -> np.ndarray:
    return np.full(198, value, dtype=np.float32)


class TestColdAndWarmTransition:
    def test_window_is_none_below_seq_len(self) -> None:
        store = FlowWindowStore(seq_len=10)
        for i in range(9):
            store.push("k", _vec(i))
        assert store.window("k") is None

    def test_window_available_once_seq_len_reached(self) -> None:
        store = FlowWindowStore(seq_len=10)
        for i in range(10):
            store.push("k", _vec(i))
        window = store.window("k")
        assert window is not None
        assert window.shape == (10, 198)

    def test_window_unknown_key_is_none(self) -> None:
        store = FlowWindowStore(seq_len=10)
        assert store.window("never-seen") is None

    def test_window_slides_dropping_oldest(self) -> None:
        store = FlowWindowStore(seq_len=3)
        for i in range(5):
            store.push("k", _vec(i))
        window = store.window("k")
        np.testing.assert_allclose(window[:, 0], [2.0, 3.0, 4.0])


class TestKeyIsolation:
    def test_different_keys_have_independent_windows(self) -> None:
        store = FlowWindowStore(seq_len=2)
        store.push("a", _vec(1))
        store.push("a", _vec(2))
        store.push("b", _vec(9))
        assert store.window("b") is None
        assert store.window("a") is not None


class TestLruEviction:
    def test_oldest_key_evicted_when_capacity_exceeded(self) -> None:
        store = FlowWindowStore(seq_len=1, max_keys=2)
        store.push("a", _vec(1))
        store.push("b", _vec(2))
        store.push("c", _vec(3))
        assert store.window("a") is None
        assert store.window("b") is not None
        assert store.window("c") is not None

    def test_recently_used_key_is_not_evicted(self) -> None:
        store = FlowWindowStore(seq_len=1, max_keys=2)
        store.push("a", _vec(1))
        store.push("b", _vec(2))
        store.window("a")  # touch "a" so it's no longer least-recently-used
        store.push("c", _vec(3))
        assert store.window("a") is not None
        assert store.window("b") is None


class TestTtlExpiry:
    def test_expired_key_is_dropped(self) -> None:
        clock = {"t": 0.0}
        store = FlowWindowStore(seq_len=1, ttl_s=10.0, clock=lambda: clock["t"])
        store.push("a", _vec(1))
        clock["t"] = 11.0
        assert store.window("a") is None

    def test_unexpired_key_survives(self) -> None:
        clock = {"t": 0.0}
        store = FlowWindowStore(seq_len=1, ttl_s=10.0, clock=lambda: clock["t"])
        store.push("a", _vec(1))
        clock["t"] = 5.0
        assert store.window("a") is not None


class TestStats:
    def test_stats_report_tracked_and_warm_keys(self) -> None:
        store = FlowWindowStore(seq_len=2)
        store.push("a", _vec(1))
        store.push("a", _vec(2))
        store.push("b", _vec(1))
        stats = store.stats()
        assert stats["tracked_keys"] == 2
        assert stats["warm_keys"] == 1
