from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

NonNeg = Annotated[float, Field(ge=0, le=1e9, allow_inf_nan=False)]
Pct = Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]
Frac = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Cat = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64,
                                       pattern=r"^[A-Za-z0-9_.\- ]+$")]
EventId = Annotated[str, StringConstraints(max_length=64, pattern=r"^[A-Za-z0-9_.\-]+$")]

EXAMPLE = dict(
    event_id="NXE-000001", service_name="inventory", environment="production", region="ap-south-1",
    deployment_type="none", customer_tier="standard", release_age_hours=6.3, requests_per_minute=3911,
    p95_latency_ms=55, error_rate_pct=0.48, cpu_utilization_pct=75.2, memory_utilization_pct=64.21,
    db_connections_pct=38.18, queue_depth=24, dependency_failures_15m=0, change_failure_rate_30d=0.177,
    active_alerts=6, incident_history_30d=0)

class EventIn(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE]})
    event_id: EventId | None = None          # optional correlation id; never used as a feature
    service_name: Cat
    environment: Cat
    region: Cat
    deployment_type: Cat
    customer_tier: Cat
    release_age_hours: NonNeg
    requests_per_minute: NonNeg
    p95_latency_ms: NonNeg
    error_rate_pct: Pct
    cpu_utilization_pct: Pct
    memory_utilization_pct: Pct
    db_connections_pct: Pct
    queue_depth: NonNeg
    dependency_failures_15m: NonNeg
    change_failure_rate_30d: Frac
    active_alerts: NonNeg
    incident_history_30d: NonNeg

class BatchIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    events: list[EventIn] = Field(min_length=1)

class Prediction(BaseModel):
    index: int = 0
    event_id: str | None = None
    risk_probability: float
    alert: bool                 # level > 0 (before suppression)
    level: int                  # 0 none, 1 review, 2 acknowledge, 3 page
    level_name: str
    suppressed: bool
    notify: bool                # alert and not suppressed
    unknown_categories: list[str] = Field(default_factory=list)

class RiskOut(Prediction):
    request_id: str
    threshold: float
    model: str
    artifact_version: str
    note: str

class BatchOut(BaseModel):
    request_id: str
    threshold: float
    model: str
    artifact_version: str
    count: int
    alerts: int
    notified: int
    results: list[Prediction]
    note: str

class ModelInfo(BaseModel):
    model: str
    artifact_version: str
    threshold: float
    alert_budget_per_1000: float | None = None
    features: list[str]
    known_categories: dict[str, list[str]]
    loaded_at: str | None = None
    note: str