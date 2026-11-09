# Governance: alert policy, risk register, audit, reproducibility

## 1. Human-in-the-loop alert policy
- The model recommends; a human confirms. No automatic restart, rollback or customer message.
- Every notification shows risk, threshold, top drivers and the artifact version.
- Operators mark each alert "useful", "false alarm" or "missed incident"; this feeds the weekly review.

## 2. Escalation levels (`src/incident_risk/alert_policy.py`)
| Level | Rule | Action | Who | Response target (proposal) |
|---|---|---|---|---|
| 0 none | risk below threshold | none | n/a | n/a |
| 1 review | threshold up to 2x threshold | ticket | service owner | same shift |
| 2 acknowledge | 2x up to 3x | notify on-call | on-call | 15 minutes |
| 3 page | 3x or more, production only | page, human confirms | on-call and lead | 5 minutes |
Multipliers are in `configs/config.yaml`; scores are capped at 0.95.

## 3. Suppression rules
- Same service, region and environment: repeats at the same or lower level are muted for 60 minutes.
- Escalation to a higher level always notifies.
- Planned maintenance: manual suppression by on-call, logged with reason and expiry.
- Suppressed alerts are still scored and logged.

## 4. Rollback triggers (proposals; confirm with the team)
| Trigger | Action |
|---|---|
| Alerts above 100 per 1,000 for 24 hours | Roll back or switch to the rule baseline |
| Score PSI above 0.25 for 2 days | Investigate; hold alerting at level 1 |
| API 5xx above 2% or p95 above 200 ms for 15 minutes | Roll back the deployment |
| Model checksum or feature-order mismatch | API refuses to serve (503) |
| False-alarm share above 80% for a week | Review threshold and budget |
| Rolling precision below half of validated (once real labels exist) | Roll back |
How: `python scripts/model_release.py rollback <version>`, then `docker compose restart api`.

## 5. Model risk register
| ID | Risk | Likelihood | Impact | Mitigation | Owner |
|---|---|---|---|---|---|
| R1 | Synthetic labels give false confidence | High | High | Disclaimers everywhere; no production use | ML owner |
| R2 | Alert fatigue | Med | High | Alert budget, suppression, weekly review | On-call lead |
| R3 | Data drift after deployment | Med | High | PSI and KS monitoring, rollback triggers | ML owner |
| R4 | Label leakage through proxy features | Low | High | Chronological split, 60-minute gap, ablation, leakage screen | ML owner |
| R5 | Weak subgroup | Med | Med | Segment report, low-sample flags | ML owner |
| R6 | Miscalibrated probabilities | Med | Med | Calibration check every release | ML owner |
| R7 | Corrupted or swapped artifact | Low | High | SHA-256 check before loading | Platform |
| R8 | Suppression state lost on restart | Med | Low | Move to shared store before production | Platform |
| R9 | Latency surge floods alerts (x3 latency gave 587 per 1,000) | Med | High | Budget-burn alert at 100 per 1,000; suppression | On-call lead |
| R10 | Bad sensor value pages on-call | Low | High | API input caps; paging only in production | Platform |
| R11 | Missing telemetry degrades scores | Med | Med | Missing-value monitor; imputer fitted on train | ML owner |

## 6. Audit logging requirements
- Each prediction: timestamp, request id, model, artifact version, input SHA-256, service, environment, risk, threshold, level, suppressed.
- Each model change: who, when, old and new version, reason, metrics, approver.
- Each manual override or suppression: who, when, reason, expiry.
- Append-only, access-controlled, retained as in the model card.

## 7. Reproducibility checklist
- [ ] `python -m venv`, `pip install -r requirements.txt`, `pip install -e .` work on a clean machine
- [ ] `requirements.lock` and `requirements-api.lock` committed
- [ ] data checksum matches `data/checksums.json`
- [ ] seed fixed in config; retraining gives identical metrics
- [ ] split boundaries saved in `reports/split_boundaries.json`
- [ ] test set evaluated exactly once (`reports/final/FINAL_TEST_LOCK.json`)
- [ ] pytest and ruff pass; Docker image builds
- [ ] MLflow run and artifact manifest recorded