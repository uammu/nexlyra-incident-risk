# Day 2: Feature engineering, baselines, leakage control

| Tasks | Status | Evidence |
|---|---|---|
| 1-2 | Done | this file |
| 3-12 | Done | data.py, splits.py; tests/test_splits.py |
| 13-19 | Done | features.py; tests/test_features.py |
| 20-32 | Done | models.py; reports/baselines/baseline_table.csv, pr_curves.png, roc_curves.png |
| 33-37 | Done | validation_comparison.json; MLflow; config seed; reports/environment.json |
| 38-45 | Done | tests/; class weights in models.py; no resampling of validation or test |
| 46-49 | Done | configs/config.yaml alert block; docs/problem_statement.md; metrics.py |
| 50-52 | Done | reports/baselines/baseline_table.csv; this report |

## Blockers and notes
- pandas 3.1 and scikit-learn 1.9 needed a fix: `RuleBaseline` mixin order.
- `majority_class` recall at budget is meaningless (all scores equal).

## Reproduce
`python -m pytest -q` and the commands in README.md