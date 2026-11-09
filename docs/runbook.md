# Runbook

## Safe retraining
1. Use new data only; chronological split with a 60-minute gap; fit preprocessing on train.
2. Compare the candidate with the current champion on the same validation window.
3. Promote only if PR-AUC and calibration are at least as good. Use a fresh, untouched test window.
4. Run pytest, then `register`, archive the old version (`python scripts/model_release.py archive`), restart the API.

## Rollback
1. Note the bad version (`/model-info`) and the time.
2. `python scripts/model_release.py list`, then `python scripts/model_release.py rollback <version>` (checksum-verified).
3. `docker compose restart api`; check `/ready` returns 200 and Grafana "Model loaded" is 1.
4. Record who, when and why.

## Alert triage (a human decides)
Check the level, the top drivers and that service's recent errors and latency. Choose acknowledge, escalate or mark "false alarm" with a reason. Levels: 1 review, 2 acknowledge, 3 page (production only).

## Incident response
Detect, acknowledge, check artifact version and drift status, mitigate, record in the audit log, review in the weekly meeting. If the model misbehaves, switch to the rule baseline or mute notifications.