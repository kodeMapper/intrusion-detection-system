"""Unit tests for InMemoryAlertRepository — a provisional stand-in for Dev 2's
Postgres-backed AlertRepository (see dev_assignment_plan.md Contract 3).
Implements the frozen AlertRepository Protocol exactly so it's a drop-in
replacement later; these tests exercise that Protocol's contract directly."""
from __future__ import annotations

from uuid import UUID

import pytest

from service.detection_api.src.models.schemas import AlertCreate, AlertFilter
from service.detection_api.src.services.alert_repository import InMemoryAlertRepository


def _make_alert_create(**overrides: object) -> AlertCreate:
    defaults: dict[str, object] = dict(
        flow_id="flow-1",
        src_ip="10.0.0.1",
        dst_ip="10.0.0.2",
        src_port=1234,
        dst_port=80,
        protocol=6,
        attack_type="DoS",
        severity="high",
        confidence=0.9,
        model_name="ml_ensemble_xgb_rf_lgbm",
        shap_explanation=None,
    )
    defaults.update(overrides)
    return AlertCreate(**defaults)


class TestCreateAndGet:
    @pytest.mark.asyncio
    async def test_create_returns_alert_with_generated_id(self) -> None:
        repo = InMemoryAlertRepository()
        alert = await repo.create(_make_alert_create())
        assert isinstance(alert.id, UUID)
        assert alert.status == "open"
        assert alert.attack_type == "DoS"

    @pytest.mark.asyncio
    async def test_get_by_id_returns_created_alert(self) -> None:
        repo = InMemoryAlertRepository()
        created = await repo.create(_make_alert_create())
        fetched = await repo.get_by_id(created.id)
        assert fetched is not None
        assert fetched.id == created.id

    @pytest.mark.asyncio
    async def test_get_by_id_returns_none_when_missing(self) -> None:
        repo = InMemoryAlertRepository()
        from uuid import uuid4

        assert await repo.get_by_id(uuid4()) is None


class TestList:
    @pytest.mark.asyncio
    async def test_list_paginates(self) -> None:
        repo = InMemoryAlertRepository()
        for i in range(5):
            await repo.create(_make_alert_create(flow_id=f"flow-{i}"))
        page = await repo.list(AlertFilter(), page=1, per_page=2)
        assert len(page.items) == 2
        assert page.total == 5

    @pytest.mark.asyncio
    async def test_list_filters_by_severity(self) -> None:
        repo = InMemoryAlertRepository()
        await repo.create(_make_alert_create(severity="low"))
        await repo.create(_make_alert_create(severity="critical"))
        page = await repo.list(AlertFilter(severity="critical"), page=1, per_page=10)
        assert page.total == 1
        assert page.items[0].severity == "critical"


class TestUpdateStatus:
    @pytest.mark.asyncio
    async def test_update_status_changes_status(self) -> None:
        repo = InMemoryAlertRepository()
        created = await repo.create(_make_alert_create())
        updated = await repo.update_status(created.id, "resolved", by="analyst@example.com")
        assert updated.status == "resolved"

    @pytest.mark.asyncio
    async def test_update_status_missing_alert_raises_key_error(self) -> None:
        repo = InMemoryAlertRepository()
        from uuid import uuid4

        with pytest.raises(KeyError):
            await repo.update_status(uuid4(), "resolved", by="analyst@example.com")
