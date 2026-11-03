0:00 Problem: predict a major incident in the next 60 min; alert budget of 50 per 1,000 events; synthetic data disclaimer.
0:45 Pipeline: chronological split with a 60-min gap, train-only preprocessing, 3+ model families, calibration. Show reports/split_boundaries.json and the comparison table.
1:45 Results: PR-AUC, recall at budget, false alerts per 1,000, calibration curve, bootstrap CI. Say plainly that numbers are not real-world accuracy.
2:30 API: `make up`; /ready, POST examples/single.json, show request_id and level; send invalid_cpu.json (structured 422); show unknown_category.json.
3:30 Dashboard: filters, threshold what-if (labelled simulation), SHAP panel for one event.
4:15 Observability: Grafana panels, send batch_100, watch the counters move; mention alert rules and rollback triggers.
4:45 Governance and handover: model card, alert policy, rollback script, limitations.