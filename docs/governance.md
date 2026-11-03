# Governance: Alert Policy, Risk Register, Audit and Reproducibility

## 1. Human-in-the-loop alert policy
- The model recommends. A human confirms. No automatic restart, rollback or customer message.
- Every notification shows: risk, threshold, top drivers (SHAP), recent similar events.
- Operators can mark an alert "useful / false alarm / missed incident". This feeds the weekly review.

## 2. Escalation levels (from `alert_policy.py`)
| Level | Rule | Action | Who | Response target |
|---|---|---|---|---|
| 0 | risk < threshold | none | n/a | n/a |
| 1 review | threshold <= risk < 2x | ticket / chat message | service owner | same shift |
| 2 acknowledge | 2x <= risk < 3x | notify on-call | on-call | <proposed 15 min> |
| 3 page | risk >= 3x, production only | page + human confirms | on-call + lead | <proposed 5 min> |
Multipliers live in config.yaml. Levels are capped at risk 0.95.

## 3. Suppression rules
- Same service + region + environment: repeats at the same or lower level muted for 60 min.
- Escalation to a higher level always notifies.
- Planned maintenance window: manual suppression by on-call, logged with reason and expiry.
- Suppressed alerts are still scored and written to the audit log.

## 4. Rollback triggers (values are PROPOSALS, confirm with the team)
| Trigger | Action |
|---|---|
| Alerts per 1,000 above 2x budget for 24 h | Switch to previous champion or rule baseline |
| PSI > 0.25 on 3 or more top-SHAP features for 2 days | Investigate, freeze alerting at level 1 |
| Rolling 7-day precision below half of validation precision (once real labels exist) | Roll back |
| API 5xx above 2% or p95 latency above 200 ms for 15 min | Roll back deployment |
| Model checksum or feature-order mismatch | API refuses to serve (503) |
| Operator "false alarm" share above 80% for a week | Review threshold and budget |
How: repoint MLflow alias `champion` to the previous version, run `make register`, restart API.

## 5. Model risk register
| ID | Risk | Likelihood | Impact | Mitigation | Owner |
|---|---|---|---|---|---|
| R1 | Synthetic labels give false confidence | High | High | Model card disclaimer; no prod use | ML owner |
| R2 | Alert fatigue from false alerts | Med | High | Alert budget, suppression, weekly review | On-call lead |
| R3 | Data drift after deployment | Med | High | PSI monitoring, rollback triggers | ML owner |
| R4 | Label leakage (proxy features) | Low | High | Chronological split, 60-min gap, ablation | ML owner |
| R5 | Weak subgroup (service/region/tier) | Med | Med | segment report, low-sample flags | ML owner |
| R6 | Miscalibrated probabilities | Med | Med | calibration check per release | ML owner |
| R7 | Corrupted or swapped artifact | Low | High | checksum check at load | Platform |
| R8 | Single-process suppression state lost on restart | Med | Low | Move to Redis before production | Platform |

## 6. Audit logging requirements
- Each prediction: timestamp, request_id, model name, artifact version, input SHA-256, service, environment, risk, threshold, level, suppressed.
- Each model change: who, when, old and new version, reason, metrics, approver.
- Each manual override or suppression: who, when, reason, expiry.
- Append-only, time-synchronised, access-controlled, retained per the model card.

## 7. Reproducibility checklist
- [ ] `make setup` works on a clean machine
- [ ] `requirements.lock` committed
- [ ] data checksum matches `data/checksums.json`
- [ ] seed set in config; `make reproduce` gives the same metrics twice
- [ ] split boundaries saved in `reports/split_boundaries.json`
- [ ] test set evaluated exactly once
- [ ] `pytest` and `ruff` pass; Docker image builds
- [ ] MLflow run and manifest recorded for the champion
- [ ] model card and this document updated with final numbers