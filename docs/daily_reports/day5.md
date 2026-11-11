# Day 5: Deployment, API, dashboard, observability

| Tasks | Status | Evidence |
|---|---|---|
| 1-2 | Done | this file |
| 3-14 | Done | api/ package; /ready and /predict outputs; docs/api.md |
| 15-20 | Done | api/schemas.py; tests/test_api.py |
| 21-26 | Done | api/audit.py, api/metrics.py; /metrics/ |
| 27-32 | Done | tests/test_api.py; scripts/debug_batch.py (100-record batch) |
| 33-35 | Done | reports/benchmark.json, benchmark_batch.json; docs/deployment.md |
| 36-39 | Done | Dockerfile, docker-compose.yml; 'docker compose ps' shows 4 running |
| 40-46 | Not run | change to Done after the dashboard screenshot in docs/evidence/ |
| 47-48 | Done | examples/; docs/api.md |
| 49 | Done | .github/workflows/ci.yml |
| 50 | Done | docs/evidence/pip_audit.txt |
| 51 | Done | scripts/model_release.py; docs/deployment.md |
| 52 | Done | docs/demo_script.md; this report |

## Blockers and notes
- `curl` and `findstr` were unavailable; Python was used instead.
- Benchmark sampled too few rows for batches (fixed).
- `docker-compose.yml` was emptied by a commit and restored from git.

## Reproduce
`python -m pytest -q` and the commands in README.md