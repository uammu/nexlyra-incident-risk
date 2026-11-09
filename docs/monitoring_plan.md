# Monitoring plan

| Item | Definition |
|---|---|
| Reference | Training-data profile: `reports/monitoring/reference_profile.json` |
| Data drift | Daily PSI and KS per feature. Under 0.10 stable, 0.10 to 0.25 watch, over 0.25 investigate |
| Prediction drift | PSI of the score distribution, and alerts per 1,000 over 24 hours (healthy range 25 to 100) |
| Performance | Labels arrive 60 minutes after each event. Join weekly; PR-AUC, precision, recall at threshold; trust only with at least 30 incidents |
| Delayed labels | Until joined, performance is "unknown", never "fine" |
| Owners | ML owner: drift review weekly. On-call lead: alert volume and false-alarm feedback. Platform: API, metrics, artifact |

## Simulation evidence (`reports/monitoring/drift_scenarios_summary.csv`)
| Scenario | Features flagged | Score PSI | Alerts per 1,000 |
|---|---|---|---|
| Normal validation | 0 | 0.005 stable | 49 |
| Latency x1.5 | 1 investigate | 0.354 investigate | 176 |
| Traffic x2, errors x2 | 2 investigate | 0.078 stable | 77 |
| CPU and memory +15 points | 2 investigate | 0.119 watch | 72 |

Lesson: in the traffic case feature drift fired while score drift stayed quiet, so both must be watched.

## Retraining
Triggers: three or more of `p95_latency_ms`, `error_rate_pct`, `requests_per_minute` at "investigate" for 2 days; rolling precision below half of validated; false-alarm share above 80% for a week; a major architecture change. Schedule: review monthly; retrain on a trigger or at 90 days, whichever comes first.

## Tooling note
Evidently was attempted but is incompatible with pandas 3.1 in this environment, so the PSI and KS tables are the evidence.