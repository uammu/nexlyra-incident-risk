# Day 3: Advanced training, optimization, calibration

| Tasks | Status | Evidence |
|---|---|---|
| 1-2 | Done | this file |
| 3-8 | Done | boosters.py, cv.py; tests/test_cv.py |
| 9-16 | Done | tune.py; configs/best_params.json; MLflow experiment nexlyra-incident-risk-tuning |
| 17-28 | Done | validation_comparison.json; learning_curve.png; feature_ablation.csv |
| 29-33 | Done | reports/segment_validation.csv; sensitivity.png |
| 34-39 | Done | reports/calibration_comparison.json, calibration_curves.png |
| 40-46 | Done | validation_champion_summary.json; false_positive_samples.csv; false_negative_samples.csv |
| 47 | Done | docs/model_card.md (comparison and calibration) |
| 48-50 | Done | register.py; models/artifact_manifest.json; MLflow registry |
| 51-52 | Done | README run order; this report |

## Blockers and notes
- MLflow 3 needed `serialization_format=cloudpickle` for the custom calibrated model.
- Logistic regression beat tuned boosters; sigmoid kept over isotonic.

## Reproduce
`python -m pytest -q` and the commands in README.md