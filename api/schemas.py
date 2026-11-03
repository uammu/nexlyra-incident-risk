from pydantic import BaseModel, Field


class EventIn(BaseModel):
    service_name: str = Field(min_length=1)
    environment: str = Field(min_length=1)  # e.g. "production"
    region: str = Field(min_length=1)
    deployment_type: str = Field(min_length=1)  # e.g. "none", "blue_green"
    customer_tier: str = Field(min_length=1)  # e.g. "standard", "enterprise"
    release_age_hours: float = Field(ge=0)
    requests_per_minute: float = Field(ge=0)
    p95_latency_ms: float = Field(ge=0)
    error_rate_pct: float = Field(ge=0, le=100)
    cpu_utilization_pct: float = Field(ge=0, le=100)
    memory_utilization_pct: float = Field(ge=0, le=100)
    db_connections_pct: float = Field(ge=0, le=100)
    queue_depth: float = Field(ge=0)
    dependency_failures_15m: float = Field(ge=0)
    change_failure_rate_30d: float = Field(ge=0, le=1)
    active_alerts: float = Field(ge=0)
    incident_history_30d: float = Field(ge=0)


class RiskOut(BaseModel):
    request_id: str
    risk_probability: float
    alert: bool  # level > 0 (before suppression)
    level: int  # 0 none, 1 review, 2 acknowledge, 3 page
    level_name: str
    suppressed: bool  # True = repeat inside cooldown
    notify: bool  # alert and not suppressed
    threshold: float
    model: str
    artifact_version: str
    note: str = "Synthetic-data model. Not for production alerting. A human decides every action."
