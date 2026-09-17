"""Shared pytest fixtures for Dev 1's explainability + detection API test suite.

Kept intentionally small: fixtures that depend on modules not yet written
(fake engines, FastAPI test client, in-memory repo, etc.) are added in later
phases alongside the modules they exercise, so an early collection error in
one area never blocks unrelated tests from running.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
LIVE_DATASET_CSV = ROOT / "tests" / "live_test_dataset.csv"

# Mirrors service/models/src/unified_predictor_worker.py's sys.path bootstrap so
# tests can `from models.src.explain import ...` exactly like production code does.
_SERVICE_DIR = str(ROOT / "service")
if _SERVICE_DIR not in sys.path:
    sys.path.insert(0, _SERVICE_DIR)

# Canonical 42-column UNSW-NB15 feature order the ML baseline was trained on
# (verified in Phase 0 against tests/live_test_dataset.csv's header, minus
# id/attack_cat/label). See DEV1_PROGRESS.md "Finding 0" for why this order
# matters: feature_selector.pkl cannot realign columns on its own.
CANONICAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes", "rate",
    "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit",
    "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean",
    "dmean", "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl",
    "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst",
    "is_sm_ips_ports",
)


def _row_as_sample(row: "pd.Series") -> dict[str, Any]:
    """Convert a pandas row into plain Python scalars (int/float/str).

    pandas/numpy dtypes (numpy.int64, numpy.float64, ...) aren't JSON
    serializable, which matters for tests that POST fixtures through
    FastAPI's TestClient.
    """
    result: dict[str, Any] = {}
    for col in CANONICAL_FEATURE_COLUMNS:
        value = row[col]
        result[col] = value.item() if hasattr(value, "item") else value
    return result


@pytest.fixture(scope="session")
def live_dataset() -> pd.DataFrame:
    return pd.read_csv(LIVE_DATASET_CSV)


@pytest.fixture()
def benign_flow(live_dataset: pd.DataFrame) -> dict[str, Any]:
    row = live_dataset[live_dataset["label"] == 0].iloc[0]
    return _row_as_sample(row)


@pytest.fixture()
def attack_flow(live_dataset: pd.DataFrame) -> dict[str, Any]:
    row = live_dataset[live_dataset["label"] == 1].iloc[0]
    return _row_as_sample(row)


@pytest.fixture()
def flow_row(benign_flow: dict[str, Any]) -> dict[str, Any]:
    """Default single-flow sample for tests that don't care about attack vs benign."""
    return benign_flow


@pytest.fixture(scope="session")
def artifacts_available() -> bool:
    """True when the gitignored ML ensemble artifacts are present on disk.

    final_rf.pkl (and *.ckpt files) are excluded from git, so a fresh clone
    cannot load the ML ensemble. Tests marked @pytest.mark.requires_artifacts
    should skip cleanly rather than fail when this is False.
    """
    return (ROOT / "service" / "models" / "artifacts" / "final_rf.pkl").exists()


@pytest.fixture(autouse=True)
def _skip_if_artifacts_required_but_missing(request: pytest.FixtureRequest, artifacts_available: bool) -> None:
    if request.node.get_closest_marker("requires_artifacts") and not artifacts_available:
        pytest.skip("model artifacts not present on disk (gitignored) — skipping artifact-dependent test")
