from prometheus_client import Counter, Gauge, Histogram


REQUESTS = Counter("risk_http_requests_total", "HTTP requests", ["route", "method", "status"])
HTTP_SECONDS = Histogram("risk_http_request_seconds", "HTTP latency", ["route"],
                         buckets=(.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5))
INFER_SECONDS = Histogram("risk_inference_seconds", "Model scoring time per call", ["kind"],
                          buckets=(.0005, .001, .0025, .005, .01, .025, .05, .1, .25, .5, 1))
PREDICTIONS = Counter("risk_predictions_total", "Events scored")
SCORES = Histogram("risk_score", "Predicted risk probability",
                   buckets=(.01, .02, .05, .1, .2, .3, .4, .6, .8, 1.0))
ALERTS = Counter("risk_alerts_total", "Alerts raised", ["level"])
SUPPRESSED = Counter("risk_alerts_suppressed_total", "Alerts muted by cooldown")
UNKNOWN_CAT = Counter("risk_unknown_category_total", "Unseen category values", ["feature"])
MODEL_LOAD_FAILURES = Counter("risk_model_load_failures_total", "Model load failures")
MODEL_LOADED = Gauge("risk_model_loaded", "1 if a verified model is loaded")
MODEL_INFO = Gauge("risk_model_info", "Loaded model", ["version", "model"])
