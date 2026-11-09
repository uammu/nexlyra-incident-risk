# EDA report (train period only: 2026-01-01 to 2026-05-26, 6,999 rows)

Validation and test periods were not used. Figures are in `reports/eda/`.

| Check | Finding | Evidence |
|---|---|---|
| Target balance | 11.2% incidents (784 of 6,999) | eda_summary.md |
| By month | 0.097 to 0.123, no trend | target_by_month.png |
| By category | Small spread (region 0.109 to 0.116; service 0.088 to 0.124) | target_by_category.png |
| Correlation | No pair with absolute Spearman at or above 0.6; max VIF 1.00 | correlation.png, vif.csv |
| Outliers, skew | No values outside valid ranges | outliers.csv, distributions.png |
| Seasonality | Hourly 0.068 to 0.140 (about 290 rows per hour, mostly noise); weekday 0.101 to 0.125 | seasonality.png |
| Early vs late | Target 0.108 vs 0.116; largest KS shift 0.028 (db_connections_pct) | early_vs_late.csv |

## Proxy-leakage review
The single-feature ROC-AUC screen flagged `p95_latency_ms` (0.903, just above the 0.90 line). Reviewed and judged a legitimate signal:
1. Alone it reaches 0.90, against 0.94 for the full model, so it is not a near-copy of the label.
2. The ablation shows traffic and latency features together reach PR-AUC 0.715 versus 0.763 for all features.
3. `active_alerts` and `incident_history_30d` alone score PR-AUC 0.176, so they do not leak.

Open assumption: latency is measured at or before the event timestamp. Confirm with the data owner.

## Dataset limitations
Synthetic labels; independent features (VIF 1.0), unlike real telemetry; no bursts, outages or missing data; one 7-month window; label rule unknown; no severity field; context features carry almost no signal.