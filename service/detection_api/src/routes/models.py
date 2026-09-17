"""GET /api/v1/models/status.

A superset of the documented single-model shape: this system serves 5 models
(ML ensemble + DL Stage1/Stage2 + VAE AE) in parallel, not one "active"
model. Artifact paths are reported as basenames only — returning absolute
filesystem paths from an unauthenticated endpoint is information disclosure.
No auth in Week 1 (Dev 2 adds JWT); combined with basename-only paths that
keeps the unauthenticated exposure to non-sensitive metadata.
"""
from __future__ import annotations

from typing import Protocol

from fastapi import APIRouter, Depends, HTTPException, Request

from service.detection_api.src.models.schemas import ModelsStatusResponse

router = APIRouter(prefix="/api/v1", tags=["models"])


class ModelsStatusProvider(Protocol):
    def status(self) -> ModelsStatusResponse: ...


def get_models_status_provider(request: Request) -> ModelsStatusProvider:
    provider = getattr(request.app.state, "models_status_provider", None)
    if provider is None:
        raise HTTPException(status_code=503, detail="model registry not initialised")
    return provider


@router.get("/models/status", response_model=ModelsStatusResponse)
async def models_status(provider: ModelsStatusProvider = Depends(get_models_status_provider)) -> ModelsStatusResponse:
    return provider.status()
