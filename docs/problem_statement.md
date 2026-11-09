# Problem statement

**Business problem.** On-call engineers cannot watch every service event. We score each event for the chance of a major incident in the next 60 minutes, so they can review the riskiest few first.

**Incident severity (assumption).** "Major incident" means whatever `incident_next_60m = 1` encodes. The dataset has no severity levels, so all positives are treated equally. To confirm with the data owner.

**Prediction horizon.** 60 minutes after the event timestamp.

**Timestamp semantics.** `event_timestamp` is when the telemetry was observed (30-minute cadence, 2026-01-01 to 2026-07-28). The label is known only 60 minutes later, so splits keep a 60-minute gap. Assumption: features are measured at or before the timestamp.

**Stakeholders.** On-call engineers (consume alerts), service owners, on-call lead (alert volume), ML owner (model), platform team (API).

**Alert budget and costs (assumptions).** 50 alerts per 1,000 events. A missed incident costs 20 times a false alert (config values, not measured).

**Non-goals.** No automatic remediation, paging or rollback; no root-cause analysis; no ranking of people or teams; no real-world accuracy claim; no use on real telemetry before validation.