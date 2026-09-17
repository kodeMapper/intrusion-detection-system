"""Provisional AlertRepository implementation.

Per dev_assignment_plan.md Contract 3, the real Postgres-backed
AlertRepository is Dev 2's deliverable. Since no separate Dev 2 collaborator
exists yet in this working copy, this in-memory version unblocks Dev 1's own
progress now — it implements the frozen AlertRepository Protocol exactly, so
swapping in the real implementation later requires no call-site changes.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import UUID, uuid4

from service.detection_api.src.models.schemas import Alert, AlertCreate, AlertFilter, AlertPage


class InMemoryAlertRepository:
    def __init__(self) -> None:
        self._alerts: dict[UUID, Alert] = {}
        self._lock = asyncio.Lock()

    async def create(self, alert: AlertCreate) -> Alert:
        async with self._lock:
            now = datetime.now(timezone.utc)
            record = Alert(**alert.model_dump(), id=uuid4(), status="open", created_at=now, updated_at=now)
            self._alerts[record.id] = record
            return record

    async def get_by_id(self, id: UUID) -> Alert | None:
        async with self._lock:
            return self._alerts.get(id)

    async def list(self, filters: AlertFilter, page: int, per_page: int) -> AlertPage:
        async with self._lock:
            items = list(self._alerts.values())

        if filters.severity is not None:
            items = [a for a in items if a.severity == filters.severity]
        if filters.status is not None:
            items = [a for a in items if a.status == filters.status]
        if filters.attack_type is not None:
            items = [a for a in items if a.attack_type == filters.attack_type]

        items.sort(key=lambda a: a.created_at, reverse=True)
        total = len(items)
        start = (page - 1) * per_page
        page_items = items[start : start + per_page]
        return AlertPage(items=page_items, total=total, page=page, per_page=per_page)

    async def update_status(self, id: UUID, status: str, by: str) -> Alert:
        async with self._lock:
            existing = self._alerts.get(id)
            if existing is None:
                raise KeyError(f"alert {id} not found")
            updated = existing.model_copy(update={"status": status, "updated_at": datetime.now(timezone.utc)})
            self._alerts[id] = updated
            return updated
