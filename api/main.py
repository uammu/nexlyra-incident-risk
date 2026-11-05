import hashlib
import json
import logging
import re
import sys
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
from starlette.exceptions import HTTPException as StarletteHTTPException

from api import metrics
from api.audit import audit
from api.errors import ApiError, error_body
from api.schemas import BatchIn, BatchOut, EventIn, ModelInfo, Prediction, RiskOut
from api.service import ModelService
from incident_risk.alert_policy import LEVELS, alert_level
from incident_risk.config import load_config

cfg = load_config()
MAX_BATCH = cfg.get("api", {}).get("max_batch", 500)
MAX_BODY = cfg.get("api", {}).get("max_body_bytes", 2_000_000)
NOTE = "Synthetic-data model. Not for production alerting. A human decides every action."
RID_OK = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
service = ModelService(cfg)

access = logging.getLogger("access")  # one JSON line per request: no bodies, no inputs
if not access.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(message)s"))
    access.addHandler(_h)
access.setLevel(logging.INFO)
access.propagate = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    service.load()  # startup check: logs and counts a failure, never crashes
    yield


app = FastAPI(title="Nexlyra Incident Risk API", version="0.3.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


# ---------- structured errors (task 20) ----------
def _resp(request: Request, status: int, code: str, message: str, details=None) -> JSONResponse:
    rid = getattr(request.state, "request_id", None)
    return JSONResponse(
        error_body(rid, code, message, details),
        status_code=status,
        headers={"X-Request-ID": rid} if rid else None,
    )


@app.exception_handler(ApiError)
async def _api_error(request: Request, exc: ApiError):
    return _resp(request, exc.status, exc.code, exc.message, exc.details)


@app.exception_handler(RequestValidationError)
async def _validation(request: Request, exc: RequestValidationError):
    errs = exc.errors()
    if any(e.get("type") == "json_invalid" for e in errs):
        return _resp(request, 400, "malformed_json", "Request body is not valid JSON.")
    details = [
        {
            "field": ".".join(str(x) for x in e["loc"] if x != "body") or "body",
            "problem": e["msg"],
            "type": e["type"],
        }
        for e in errs[:50]
    ]  # never echo e["input"]
    return _resp(request, 422, "validation_error", "Request failed validation.", details)


CODES = {400: "bad_request", 404: "not_found", 405: "method_not_allowed", 413: "payload_too_large"}


@app.exception_handler(StarletteHTTPException)
async def _http(request: Request, exc: StarletteHTTPException):
    return _resp(
        request, exc.status_code, CODES.get(exc.status_code, "http_error"), str(exc.detail)
    )


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    audit(
        event="unhandled_error",
        request_id=getattr(request.state, "request_id", None),
        error=type(exc).__name__,
    )
    return _resp(request, 500, "internal_error", "Unexpected server error.")


# ---------- request id, timing, metrics, access log (tasks 14, 21, 22, 24) ----------
@app.middleware("http")
async def observe(request: Request, call_next):
    rid = request.headers.get("x-request-id", "")
    rid = rid if RID_OK.match(rid) else uuid.uuid4().hex  # reject odd ids: log-injection guard
    request.state.request_id = rid
    t0, status = time.perf_counter(), 500
    try:
        cl = request.headers.get("content-length", "")
        if cl.isdigit() and int(cl) > MAX_BODY:
            response = _resp(
                request, 413, "payload_too_large", f"Body larger than {MAX_BODY} bytes."
            )
        else:
            response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = rid
        return response
    finally:
        route = getattr(request.scope.get("route"), "path", None) or (
            "/metrics" if request.url.path.startswith("/metrics") else "unmatched"
        )
        ms = (time.perf_counter() - t0) * 1000
        metrics.REQUESTS.labels(route, request.method, str(status)).inc()
        metrics.HTTP_SECONDS.labels(route).observe(ms / 1000)
        access.info(
            json.dumps(
                {
                    "request_id": rid,
                    "method": request.method,
                    "route": route,
                    "status": status,
                    "ms": round(ms, 2),
                }
            )
        )


# ---------- core scoring ----------
def _run(events: list[EventIn], kind: str, rid: str, suppress: bool = True):
    b = service.get()
    rows = [e.model_dump() for e in events]
    unknown = service.unknown_categories(rows)
    if service.strict and any(unknown):
        bad = sorted({c for u in unknown for c in u})
        raise ApiError(
            422,
            "unknown_category",
            f"Unseen category value in: {', '.join(bad)}.",
            [{"index": i, "fields": u} for i, u in enumerate(unknown) if u][:50],
        )
    probs = service.score(rows, kind)
    thr, preds = b["threshold"], []
    for i, (row, p, unk) in enumerate(zip(rows, probs, unknown)):
        p = float(p)
        level = alert_level(p, thr, row["environment"], service.policy)
        suppressed = False
        if suppress:
            with service.sup_lock:
                suppressed = service.suppressor.check(
                    (row["service_name"], row["region"], row["environment"]), level
                )
        metrics.PREDICTIONS.inc()
        metrics.SCORES.observe(p)
        if level:
            metrics.ALERTS.labels(LEVELS[level]).inc()
        if suppressed:
            metrics.SUPPRESSED.inc()
        for c in unk:
            metrics.UNKNOWN_CAT.labels(c).inc()
        audit(
            event="prediction",
            request_id=rid,
            index=i,
            model=b["meta"]["selected_model"],
            artifact_version=b["version"],
            service=row["service_name"],
            environment=row["environment"],
            input_sha256=hashlib.sha256(
                json.dumps(
                    {k: v for k, v in row.items() if k != "event_id"}, sort_keys=True
                ).encode()
            ).hexdigest(),
            risk=round(p, 5),
            threshold=round(thr, 5),
            level=level,
            suppressed=suppressed,
            unknown_categories=unk,
        )
        preds.append(
            Prediction(
                index=i,
                event_id=row["event_id"],
                risk_probability=p,
                alert=level > 0,
                level=level,
                level_name=LEVELS[level],
                suppressed=suppressed,
                notify=level > 0 and not suppressed,
                unknown_categories=unk,
            )
        )
    return preds, b


# ---------- endpoints ----------
@app.get("/health")  # liveness: is the process up?
def health():
    return {"status": "ok", "model_loaded": service.bundle is not None}


@app.get("/ready")  # readiness: can it actually score?
def ready():
    b = service.get()
    return {
        "status": "ready",
        "model": b["meta"]["selected_model"],
        "artifact_version": b["version"],
    }


@app.get("/model-info", response_model=ModelInfo)
def model_info():
    b = service.get()
    return ModelInfo(
        model=b["meta"]["selected_model"],
        artifact_version=b["version"],
        threshold=b["threshold"],
        alert_budget_per_1000=b["meta"].get("alert_budget_per_1000"),
        features=service.features,
        known_categories={k: sorted(v) for k, v in service.vocab.items()},
        loaded_at=service.loaded_at,
        note=NOTE,
    )


@app.post("/predict", response_model=RiskOut)
def predict(event: EventIn, request: Request):
    rid = request.state.request_id
    (pred,), b = _run([event], "single", rid)
    return RiskOut(
        request_id=rid,
        threshold=b["threshold"],
        model=b["meta"]["selected_model"],
        artifact_version=b["version"],
        note=NOTE,
        **pred.model_dump(),
    )


@app.post("/predict/batch", response_model=BatchOut)
def predict_batch(batch: BatchIn, request: Request, suppress: bool = True):
    """suppress=false skips the cooldown (use it when replaying history, not for live events)."""
    if len(batch.events) > MAX_BATCH:
        raise ApiError(
            413, "batch_too_large", f"Batch has {len(batch.events)} events; limit is {MAX_BATCH}."
        )
    rid = request.state.request_id
    preds, b = _run(batch.events, "batch", rid, suppress)
    return BatchOut(
        request_id=rid,
        threshold=b["threshold"],
        model=b["meta"]["selected_model"],
        artifact_version=b["version"],
        count=len(preds),
        alerts=sum(p.alert for p in preds),
        notified=sum(p.notify for p in preds),
        results=preds,
        note=NOTE,
    )
