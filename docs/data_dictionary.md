# Data dictionary (10,000 rows, 20 columns, no missing values)

Meanings are inferred from column names; confirm with the data owner.

| Column | Type | Meaning | Observed range / values |
|---|---|---|---|
| event_id | id | Unique event id. NEVER a feature | NXE-000001 onward |
| event_timestamp | datetime | Observation time, 30-minute cadence | 2026-01-01 to 2026-07-28 |
| service_name | category | Service | 10 services (analytics, api_gateway, auth, billing, file_storage, inventory, notifications, payments, recommendations, search) |
| environment | category | Environment | production, staging, development |
| region | category | Cloud region | ap-south-1, eu-west-1, us-east-1 |
| deployment_type | category | Release strategy | none, blue_green, others in CSV |
| release_age_hours | numeric | Hours since last release | 0 to 775 |
| requests_per_minute | numeric | Traffic | 80 to 31,619 |
| p95_latency_ms | numeric | 95th percentile latency | 16 to 1,610 |
| error_rate_pct | percent | Error rate | 0 to 32.9 |
| cpu_utilization_pct | percent | CPU use | 3.7 to 99.8 |
| memory_utilization_pct | percent | Memory use | 6.0 to 99.7 |
| db_connections_pct | percent | DB pool use | 1.4 to 98.5 |
| queue_depth | count | Queue backlog | 0 to 809 |
| dependency_failures_15m | count | Dependency failures, last 15 min | 0 to 14 |
| change_failure_rate_30d | fraction 0 to 1 | Failed-change share, 30 days | 0.001 to 0.783 |
| active_alerts | count | Open alerts | 0 to 18 |
| incident_history_30d | count | Incidents in last 30 days | 0 to 20 |
| customer_tier | category | Customer tier | standard, business, enterprise |
| incident_next_60m | target 0/1 | Major incident within 60 minutes | rate 11.16% |

## Preprocessing contract (fitted on TRAIN only)
Numeric: median imputation with missing indicator, then robust scaling. Categorical: most-frequent imputation, one-hot, unseen values ignored. `event_id` and `event_timestamp` are dropped. Validation rules: percent columns 0 to 100, `change_failure_rate_30d` 0 to 1, no negatives, binary target.