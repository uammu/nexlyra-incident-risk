# Day 6: Monitoring, final evaluation, hardening, handover

| Tasks | Status | Evidence |
|---|---|---|
| 1-2 | Done | this file |
| 3-4 | Done | models/artifact_manifest.json; MLflow champion alias |
| 5 | Done | tests/test_test_isolation.py; disclosure in docs/model_card.md |
| 6-22 | Done | reports/final/ (metrics, calibration, segments, windows, CIs) |
| 23-26 | Done | reports/monitoring/reference_profile.json; docs/monitoring_plan.md |
| 27 | Partial | Evidently incompatible with pandas 3.1; PSI and KS reports used instead |
| 28-34 | Done | reports/monitoring/drift_*.csv; docs/monitoring_plan.md |
| 35-38 | Done | docs/runbook.md |
| 39 | Done | docs/slo.md |
| 40-41 | Done | pytest 46 passed; ruff clean |
| 42 | Done | docs/evidence/pip_audit.txt |
| 43-44 | Done | docker compose ps; /ready from the container |
| 45 | Not run | change to Done after the dashboard screenshot |
| 46 | Not run | change to Done after the fresh-folder run |
| 47-48 | Done | README.md; docs/model_card.md |
| 49 | Not run | change to Done after recording the demo |
| 50-51 | Done | docs/executive_summary.md; docs/handover_checklist.md |
| 52-53 | Done | repository; these reports |

## Blockers and notes
- (write what actually blocked you)

## Reproduce
`python -m pytest -q` and the commands in README.md