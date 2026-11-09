# Nexlyra Incident Risk Predictor

> **SYNTHETIC DATA.** The dataset and its labels come from a simulated process. Results are NOT evidence of real-world accuracy. Do not use for production alerting without validation on real, consented telemetry.

Predicts whether a service will have a major incident within the next 60 minutes from operational telemetry. Includes a training pipeline, calibrated model, risk API, dashboard, monitoring and governance documents.

## Headline results (final test set, evaluated once)
| Metric | Value |
|---|---|
| Model | Logistic regression, sigmoid-calibrated |
| PR-AUC / ROC-AUC | 0.772 / 0.942 |
| Brier / ECE | 0.046 / 0.018 |
| Alert budget | 50 alerts per 1,000 events |
| Precision / recall at budget | 89.3% / 41.6% |
| False alerts per 1,000 | 5.3 |

Recall is capped near 47%: incidents occur at about 107 per 1,000 events, so 50 alerts cannot cover them all.

## Setup (Windows, Python 3.11)
```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```
Place `nexlyra_ml_incident_dataset_10000.csv` in `data\raw\`.

## Run order
```bat
python tasks.py validate
python tasks.py test                          :: 46 tests
python tasks.py train                         :: 7 models, calibration, frozen threshold (test not scored)
python -m incident_risk.register
python -m incident_risk.final_evaluation      :: run ONCE; locks itself
python tasks.py governance                    :: SHAP, robustness, subgroups, uncertainty
python -m incident_risk.monitoring
python scripts\export_dashboard_data.py
docker compose up --build -d                  :: API 8000, dashboard 8501, Prometheus 9090, Grafana 3000
```
Without Docker: `python tasks.py api` and `python tasks.py dash`.

## Folder map
`src/incident_risk` pipeline | `api` FastAPI service | `dashboard` Streamlit | `monitoring` Prometheus and Grafana config | `configs` settings | `tests` 46 tests | `reports` generated evidence | `docs` documentation | `examples` request payloads | `scripts` utilities

## Rules enforced
Chronological split with a 60-minute label gap; preprocessing fitted on train only; thresholds chosen on validation only; test scored once by `final_evaluation.py` (self-locking, guarded by a test); fixed seed; pinned dependencies; model checksum verified before loading.

## Documents
Problem statement, data dictionary, EDA report, model card, governance, monitoring plan, runbook, SLOs, API, deployment, executive summary, handover checklist and demo script are in `docs/`.

## Branching
Work on `master`, committed daily with evidence. For a team: short-lived feature branches, pull requests with CI green, tagged releases.