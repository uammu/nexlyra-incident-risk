# Deployment, Throughput and Rollback

## Run
`make up` starts api (8000), dashboard (8501), prometheus (9090), grafana (3000, set GRAFANA_PASSWORD).
The model is mounted read-only from ./models, never baked into the image.
Check: `make smoke`. Image tag: `IMAGE_TAG=$(git rev-parse --short HEAD) make up`.

## Throughput limits (fill from reports/benchmark.json; state the machine)
| Scenario | p50 | p95 | p99 | Requests/s |
|---|---|---|---|---|
| Single, 1 client | | | | |
| Single, 8 clients | | | | |
| Batch of 100 (per call) | | | | |
Machine: <cpus, RAM, Docker or bare metal>. Benchmark is HTTP end to end on localhost; real networks add latency.
Known limits:
- One uvicorn worker: suppression state and metrics are per process. Scaling out needs shared state (Redis) and per-replica metrics scraping.
- Max 500 events per batch, 2 MB per body. Put a proxy body limit in front (chunked uploads skip the Content-Length check).
- Model loads at startup; a new model needs an API restart (no hot reload, on purpose).

## Release
`make train && make release` registers the model and archives it to models/archive/<version>/.

## Rollback procedure
Triggers: see governance.md section 4 and monitoring/alerts.yml.
1. Note the bad version (`curl localhost:8000/model-info`) and the time. Alerting stays human-confirmed.
2. `python scripts/model_release.py list`
3. `python scripts/model_release.py rollback <version>`  (checksum-verified, current model archived first)
4. `docker compose restart api`
5. Verify: `/ready` returns 200, `/model-info` shows the older version, Grafana "Model loaded" = 1.
6. Optional: repoint the MLflow alias `champion` to the same version so registry and API agree.
7. Record who, when, why and the metrics that triggered it (audit requirement).
If the API image itself is bad: `IMAGE_TAG=<previous git sha> docker compose up -d api` (tag every release).
Last resort: switch to the rule baseline or disable notifications; the API refuses to serve a corrupted model by design.