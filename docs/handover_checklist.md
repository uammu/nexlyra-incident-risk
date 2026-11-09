# Handover checklist

Tick a box only after you have opened the file and confirmed it is real.

- [ ] README and tested setup: README.md
- [ ] Data dictionary, EDA report: docs/data_dictionary.md, docs/eda_report.md
- [ ] Pipeline, baselines, advanced models, tracking: src/incident_risk, mlflow.db, reports/baselines/
- [ ] Test metrics (once): reports/final/final_test_metrics.json, FINAL_TEST_LOCK.json
- [ ] Model card, SHAP: docs/model_card.md, reports/explain/
- [ ] API, dashboard, Docker, CI: api/, dashboard/, docker-compose.yml, .github/workflows/ci.yml
- [ ] Security scan: docs/evidence/pip_audit.txt
- [ ] Drift design, runbook, SLOs: docs/monitoring_plan.md, runbook.md, slo.md
- [ ] Six daily reports: docs/daily_reports/
- [ ] Demo: docs/demo_script.md (and video if recorded)

## Known limitations and open risks
Synthetic data; latency-spike alert flood; sensitivity to missing values; small subgroups; single API worker (suppression state in memory); Evidently incompatible with pandas 3.1; thresholds and costs are assumptions.