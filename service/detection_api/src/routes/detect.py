"""POST /api/v1/detect.

Error mapping: 422 for Pydantic validation (FastAPI's default handler, never
caught here); 200 + explanation_method="unavailable" for any explainer
failure (guaranteed by DetectionService's use of safe_explain — no explainer
exception can reach this route); 503 for an unready/unavailable detection
engine or a repository failure (DetectionUnavailableError); plain 500 only
for genuinely unexpected bugs, never masked.

Notifications (email/Slack) are dispatched via FastAPI's BackgroundTasks —
passed through to DetectionService.detect_flow — so SMTP/Slack latency never
rides on this response; persistence itself still happens synchronously, so a
returned alert always has a real ID.
"""
from __future__ import annotations

import time
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request

from service.detection_api.src.models.schemas import DetectRequest, DetectResponse, FlowInput
from service.detection_api.src.services.detection import DetectionUnavailableError

router = APIRouter(prefix="/api/v1", tags=["detection"])


def get_detection_service(request: Request):
    service = getattr(request.app.state, "detection_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="detection engine not initialised")
    return service


def _derive_flow_key(flow: FlowInput, request_flow_key: str | None) -> str:
    if request_flow_key:
        return request_flow_key
    if flow.src_ip is not None and flow.dst_ip is not None:
        return f"{flow.src_ip}->{flow.dst_ip}"
    return "__global__"


@router.post("/detect", response_model=DetectResponse)
async def detect(
    payload: DetectRequest, background_tasks: BackgroundTasks, service=Depends(get_detection_service)
) -> DetectResponse:
    start = time.perf_counter()
    request_id = str(uuid4())

    try:
        results = [
            await service.detect_flow(
                flow,
                flow_key=_derive_flow_key(flow, payload.flow_key),
                explain=payload.explain,
                background_tasks=background_tasks,
            )
            for flow in payload.flows
        ]
    except DetectionUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    elapsed_ms = (time.perf_counter() - start) * 1000
    return DetectResponse(results=results, request_id=request_id, elapsed_ms=elapsed_ms)
