# Model card: Nexlyra Incident Risk Predictor

**Model.** Logistic regression (class-weighted), sigmoid-calibrated. Artifact `20261105-f64cf30`. Threshold 0.7828 (50 alerts per 1,000 events), frozen on validation.
**Task.** Probability of a major incident within 60 minutes of an event.
**Status.** SYNTHETIC data. Not validated for production use.

## Data and windows
| Period | Rows | From | To |
|---|---|---|---|
| Train | 6,999 | 2026-01-01 00:00 | 2026-05-26 19:00 |
| Validation | 1,499 | 2026-05-26 20:00 | 2026-06-27 01:00 |
| Test (scored once) | 1,500 | 2026-06-27 02:00 | 2026-07-28 07:30 |

Dataset: `nexlyra_ml_incident_dataset_10000.csv` (10,000 rows, 20 columns), SHA-256 `2901ace7bc50e83c84316be538c57d544d08a167d4bbaabf7eaeeadd4931c806`. Labels come from a simulated probabilistic process.

## Intended use
Engineering evaluation and demonstration of an ML-ops workflow; ranking events for a human reviewer. Decision support only.

## Prohibited uses
Production alerting, paging or auto-remediation before validation on real, consented telemetry; claiming real-world accuracy; evaluating engineers or teams; changing customer treatment by tier.

## Model comparison (validation, 50 alerts per 1,000)
| Model | PR-AUC | Precision | Recall |
|---|---|---|---|
| Logistic regression (champion) | 0.783 | 0.933 | 0.409 |
| LightGBM | 0.766 | 0.880 | 0.386 |
| Tuned LightGBM / tuned XGBoost | 0.763 | 0.907 | 0.398 |
| Random forest | 0.726 | 0.867 | 0.380 |
| Rule baseline | 0.350 | 0.493 | 0.216 |
| Majority class | 0.114 | n/a (no ranking) | n/a |

The simpler model won: the simulated labels look near-linear, and tuned boosters (CV PR-AUC 0.751 and 0.755) did not beat it.

## Calibration
Raw: ECE 0.135, Brier 0.0836. Sigmoid: ECE 0.017, Brier 0.0431. Isotonic: Brier 0.0423 but ECE 0.029 and PR-AUC 0.778 (ties flatten the ranking), so sigmoid was kept.

## Final test results (threshold fixed from validation)
| Metric | Value | 95% CI (block bootstrap) |
|---|---|---|
| PR-AUC | 0.772 | 0.721 to 0.819 |
| ROC-AUC | 0.942 | n/a |
| Brier / ECE | 0.046 / 0.018 | n/a |
| Precision at budget | 0.893 | 0.853 to 0.973 |
| Recall at budget | 0.416 | 0.383 to 0.485 |
| F1 | 0.568 | n/a |
| Confusion matrix | TN 1331, FP 8, FN 94, TP 67 | n/a |
| False alerts per 1,000 | 5.3 | n/a |

Validation PR-AUC (second half) was 0.806, so the test gap is -0.034, inside the validation interval (0.69 to 0.87). Recall is bounded near 47% by the budget, since incidents occur at about 107 per 1,000 events.

## What drives the score
Permutation importance: `p95_latency_ms` (PR-AUC drop 0.66), `error_rate_pct` (0.065), `cpu_utilization_pct` (0.049), `dependency_failures_15m` (0.040). SHAP ranks `customer_tier` and `region` high, but shuffling them costs nothing (SHAP vs permutation Spearman 0.39), and the ablation agrees that context features add nothing. On synthetic data these describe the simulator, not real causes.

## Known weaknesses
- **Latency spike (x3):** alerts rise from 51 to 587 per 1,000.
- **Missing values:** PR-AUC 0.806 falls to 0.747 at 5% missing and 0.601 at 40%.
- **Noise:** PR-AUC 0.669 at 20% feature noise.
- **Unbounded linear scores:** absurd inputs gave risk 1.0; the API now rejects values beyond set limits.
- **Small subgroups:** 11 of 19 validation segments had too few samples to judge.

## Uncertainty
The validation holdout has 750 rows (about 80 incidents), so intervals are wide. Seed variance is 0 because the model is deterministic. A predicted 0.78 corresponded to an observed 0.72 (95% range 0.61 to 0.81).

## Privacy and retention
No personal data in the dataset. The audit log stores an input hash, not raw inputs. Proposed retention: 90 days for prediction logs, one year for model-change logs.

## Disclosures
`train.py` scored the test set on its earlier runs (identical numbers every time, and nothing was tuned on them). It was patched on 3 Nov, and the formal single evaluation is recorded in `reports/final/FINAL_TEST_LOCK.json`. The artifact was re-serialised afterwards, so its file hash changed while the metrics and threshold stayed identical.