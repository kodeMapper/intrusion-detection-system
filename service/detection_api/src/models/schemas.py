"""Pydantic schemas for the detection API.

Section 1 is the frozen PredictionResponse/ShapExplanation/TopFeature contract
from dev_assignment_plan.md — additive-only, never rename/remove a field.
Section 2 is the /detect request/response shape. Section 3 is /models/status.
Section 4 is provisional: Dev 2 owns the real Postgres-backed AlertRepository;
these types let Dev 1 proceed against the frozen Protocol without waiting.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, IPvAnyAddress, StringConstraints
from typing_extensions import Annotated

# Canonical 42-column UNSW-NB15 feature order the ML baseline was trained on.
# See DEV1_PROGRESS.md "Finding 0": feature_selector.pkl cannot realign columns
# on its own, so FlowInput.to_sample() must emit these keys in this exact order.
CANONICAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes", "rate",
    "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit",
    "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean",
    "dmean", "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl",
    "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst",
    "is_sm_ips_ports",
)

CategoricalStr = Annotated[
    str, StringConstraints(strip_whitespace=True, max_length=16, pattern=r"^[A-Za-z0-9_\-/.+]{1,16}$")
]

SEVERITY_LEVELS: tuple[str, ...] = ("low", "medium", "high", "critical")


# =============================================================================
# 1. Frozen contract (verbatim from dev_assignment_plan.md — additive-only)
# =============================================================================


class TopFeature(BaseModel):
    feature: str
    value: float | int | str
    shap_value: float
    direction: Literal["attack", "benign"]


class ShapExplanation(BaseModel):
    base_value: float
    predicted_value: float
    top_features: list[TopFeature]
    all_shap_values: list[float] | None
    feature_names: list[str] | None
    explanation_method: str


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    attack_probability: float
    model_name: str
    anomaly_score: float | None
    zero_day_flag: bool
    shap_explanation: ShapExplanation | None


# =============================================================================
# 2. /detect request/response
# =============================================================================


class FlowInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    dur: float = Field(ge=0, allow_inf_nan=False)
    proto: CategoricalStr
    service: CategoricalStr
    state: CategoricalStr
    spkts: int = Field(ge=0)
    dpkts: int = Field(ge=0)
    sbytes: int = Field(ge=0)
    dbytes: int = Field(ge=0)
    rate: float = Field(ge=0, allow_inf_nan=False)
    sttl: int = Field(ge=0)
    dttl: int = Field(ge=0)
    sload: float = Field(ge=0, allow_inf_nan=False)
    dload: float = Field(ge=0, allow_inf_nan=False)
    sloss: int = Field(ge=0)
    dloss: int = Field(ge=0)
    sinpkt: float = Field(ge=0, allow_inf_nan=False)
    dinpkt: float = Field(ge=0, allow_inf_nan=False)
    sjit: float = Field(ge=0, allow_inf_nan=False)
    djit: float = Field(ge=0, allow_inf_nan=False)
    swin: int = Field(ge=0)
    stcpb: int = Field(ge=0)
    dtcpb: int = Field(ge=0)
    dwin: int = Field(ge=0)
    tcprtt: float = Field(ge=0, allow_inf_nan=False)
    synack: float = Field(ge=0, allow_inf_nan=False)
    ackdat: float = Field(ge=0, allow_inf_nan=False)
    smean: int = Field(ge=0)
    dmean: int = Field(ge=0)
    trans_depth: int = Field(ge=0)
    response_body_len: int = Field(ge=0)
    ct_srv_src: int = Field(ge=0)
    ct_state_ttl: int = Field(ge=0)
    ct_dst_ltm: int = Field(ge=0)
    ct_src_dport_ltm: int = Field(ge=0)
    ct_dst_sport_ltm: int = Field(ge=0)
    ct_dst_src_ltm: int = Field(ge=0)
    is_ftp_login: int = Field(ge=0)
    ct_ftp_cmd: int = Field(ge=0)
    ct_flw_http_mthd: int = Field(ge=0)
    ct_src_ltm: int = Field(ge=0)
    ct_srv_dst: int = Field(ge=0)
    is_sm_ips_ports: int = Field(ge=0)

    flow_id: str | None = Field(default=None, max_length=128)
    src_ip: IPvAnyAddress | None = None
    dst_ip: IPvAnyAddress | None = None
    src_port: int | None = Field(default=None, ge=0, le=65535)
    dst_port: int | None = Field(default=None, ge=0, le=65535)

    def to_sample(self) -> dict[str, float | int | str]:
        """Exactly the 42 canonical keys, in training order, no metadata fields."""
        data = self.model_dump()
        return {col: data[col] for col in CANONICAL_FEATURE_COLUMNS}


class DetectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flows: list[FlowInput] = Field(min_length=1, max_length=100)
    flow_key: str | None = Field(default=None, max_length=128)
    explain: bool = True


class DetectResponse(BaseModel):
    results: list[PredictionResponse]
    request_id: str
    elapsed_ms: float


# =============================================================================
# 3. /models/status
# =============================================================================


class ModelPerformance(BaseModel):
    dataset: str
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    auc: float | None = None


class InferenceStats(BaseModel):
    total_predictions: int
    avg_latency_ms: float
    p99_latency_ms: float


class WindowState(BaseModel):
    seq_len: int
    tracked_keys: int
    warm_keys: int


class ModelStatus(BaseModel):
    name: str
    kind: Literal["ml_ensemble", "dl_stage1", "dl_stage2", "ae_vae"]
    version: str
    loaded: bool
    artifact: str
    loaded_at: datetime | None = None
    explainer: str
    threshold: float | None = None
    performance: ModelPerformance | None = None


class ModelsStatusResponse(BaseModel):
    active_model: str = "unified_ensemble"
    models: list[ModelStatus]
    inference_stats: InferenceStats
    window_state: WindowState


# =============================================================================
# 4. PROVISIONAL — Dev 2 owns the real Postgres-backed versions of these under
# src/api/db/. Field names must not drift once Dev 2's implementation lands.
# =============================================================================


class AlertCreate(BaseModel):
    flow_id: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    attack_type: str
    severity: str
    confidence: float
    model_name: str
    shap_explanation: dict | None = None


class Alert(AlertCreate):
    id: UUID
    status: str = "open"
    created_at: datetime
    updated_at: datetime


class AlertFilter(BaseModel):
    severity: str | None = None
    status: str | None = None
    attack_type: str | None = None


class AlertPage(BaseModel):
    items: list[Alert]
    total: int
    page: int
    per_page: int


class AlertRepository(Protocol):
    async def create(self, alert: AlertCreate) -> Alert: ...
    async def get_by_id(self, id: UUID) -> Alert | None: ...
    async def list(self, filters: AlertFilter, page: int, per_page: int) -> AlertPage: ...
    async def update_status(self, id: UUID, status: str, by: str) -> Alert: ...


def derive_severity(attack_type: str, confidence: float, zero_day: bool) -> str:
    """Frozen mapping shared with Dev 3's policy engine, which keys off this field.

    zero_day always escalates to at least "high" regardless of the label,
    since the AE canary firing is itself a strong signal independent of the
    (possibly wrong) supervised classification.
    """
    if attack_type == "BENIGN" and not zero_day:
        return "low"
    if zero_day:
        return "critical" if confidence >= 0.5 else "high"
    if confidence >= 0.9:
        return "critical"
    if confidence >= 0.7:
        return "high"
    if confidence >= 0.4:
        return "medium"
    return "low"
