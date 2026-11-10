from pathlib import Path

D, P, N = "Done", "Partial", "Not run"

REPORTS = {
    1: (
        "Requirements, data contract, exploration",
        [
            ("1-2", D, "this file; README.md"),
            ("3", D, "git log; branching note in README"),
            ("4-9", D, "docs/problem_statement.md"),
            ("10", D, "README.md architecture; docs/architecture.md if created"),
            ("11-15", D, ".venv, requirements.txt, ruff, folder layout"),
            ("16", D, "reports/environment.json"),
            ("17-29", D, "scripts/validate_data.py ('Data OK'); data.py; data/checksums.json"),
            ("30-44", D, "reports/eda/*.png and *.csv"),
            ("45-46", D, "docs/eda_report.md; features.py drops event_id"),
            ("47-48", D, "docs/data_dictionary.md; reports/split_boundaries.json"),
            ("49", D, "notebooks/01_eda.ipynb"),
            ("50-52", D, "validate_data.py; docs/eda_report.md; this report"),
        ],
    ),
    2: (
        "Feature engineering, baselines, leakage control",
        [
            ("1-2", D, "this file"),
            ("3-12", D, "data.py, splits.py; tests/test_splits.py"),
            ("13-19", D, "features.py; tests/test_features.py"),
            (
                "20-32",
                D,
                "models.py; reports/baselines/baseline_table.csv, pr_curves.png, roc_curves.png",
            ),
            (
                "33-37",
                D,
                "validation_comparison.json; MLflow; config seed; reports/environment.json",
            ),
            ("38-45", D, "tests/; class weights in models.py; no resampling of validation or test"),
            ("46-49", D, "configs/config.yaml alert block; docs/problem_statement.md; metrics.py"),
            ("50-52", D, "reports/baselines/baseline_table.csv; this report"),
        ],
    ),
    3: (
        "Advanced training, optimization, calibration",
        [
            ("1-2", D, "this file"),
            ("3-8", D, "boosters.py, cv.py; tests/test_cv.py"),
            (
                "9-16",
                D,
                "tune.py; configs/best_params.json; MLflow experiment nexlyra-incident-risk-tuning",
            ),
            ("17-28", D, "validation_comparison.json; learning_curve.png; feature_ablation.csv"),
            ("29-33", D, "reports/segment_validation.csv; sensitivity.png"),
            ("34-39", D, "reports/calibration_comparison.json, calibration_curves.png"),
            (
                "40-46",
                D,
                "validation_champion_summary.json; false_positive_samples.csv; false_negative_samples.csv",
            ),
            ("47", D, "docs/model_card.md (comparison and calibration)"),
            ("48-50", D, "register.py; models/artifact_manifest.json; MLflow registry"),
            ("51-52", D, "README run order; this report"),
        ],
    ),
    4: (
        "Explainability, robustness, governance",
        [
            ("1-2", D, "this file"),
            ("3-4", D, "reports/explain/explain_summary.json"),
            ("5-16", D, "reports/explain/ (SHAP, dependence, local explanations, PDP and ICE)"),
            ("17-20", D, "reports/robustness/boundary.csv, unseen_category.csv; tests/test_api.py"),
            ("21-31", D, "reports/robustness/*.csv"),
            ("32-37", D, "reports/segment_performance.csv"),
            ("38", D, "reports/uncertainty.json, reliability_bins.csv"),
            ("39-43", D, "docs/model_card.md"),
            ("44-49", D, "docs/governance.md; alert_policy.py; api/audit.py"),
            ("50-51", D, "governance.md checklist; pytest 46 passed; ruff clean"),
            ("52", D, "this report"),
        ],
    ),
    5: (
        "Deployment, API, dashboard, observability",
        [
            ("1-2", D, "this file"),
            ("3-14", D, "api/ package; /ready and /predict outputs; docs/api.md"),
            ("15-20", D, "api/schemas.py; tests/test_api.py"),
            ("21-26", D, "api/audit.py, api/metrics.py; /metrics/"),
            ("27-32", D, "tests/test_api.py; scripts/debug_batch.py (100-record batch)"),
            ("33-35", D, "reports/benchmark.json, benchmark_batch.json; docs/deployment.md"),
            ("36-39", D, "Dockerfile, docker-compose.yml; 'docker compose ps' shows 4 running"),
            ("40-46", N, "change to Done after the dashboard screenshot in docs/evidence/"),
            ("47-48", D, "examples/; docs/api.md"),
            ("49", D, ".github/workflows/ci.yml"),
            ("50", D, "docs/evidence/pip_audit.txt"),
            ("51", D, "scripts/model_release.py; docs/deployment.md"),
            ("52", D, "docs/demo_script.md; this report"),
        ],
    ),
    6: (
        "Monitoring, final evaluation, hardening, handover",
        [
            ("1-2", D, "this file"),
            ("3-4", D, "models/artifact_manifest.json; MLflow champion alias"),
            ("5", D, "tests/test_test_isolation.py; disclosure in docs/model_card.md"),
            ("6-22", D, "reports/final/ (metrics, calibration, segments, windows, CIs)"),
            ("23-26", D, "reports/monitoring/reference_profile.json; docs/monitoring_plan.md"),
            ("27", P, "Evidently incompatible with pandas 3.1; PSI and KS reports used instead"),
            ("28-34", D, "reports/monitoring/drift_*.csv; docs/monitoring_plan.md"),
            ("35-38", D, "docs/runbook.md"),
            ("39", D, "docs/slo.md"),
            ("40-41", D, "pytest 46 passed; ruff clean"),
            ("42", D, "docs/evidence/pip_audit.txt"),
            ("43-44", D, "docker compose ps; /ready from the container"),
            ("45", N, "change to Done after the dashboard screenshot"),
            ("46", N, "change to Done after the fresh-folder run"),
            ("47-48", D, "README.md; docs/model_card.md"),
            ("49", N, "change to Done after recording the demo"),
            ("50-51", D, "docs/executive_summary.md; docs/handover_checklist.md"),
            ("52-53", D, "repository; these reports"),
        ],
    ),
}

out = Path("docs/daily_reports")
out.mkdir(parents=True, exist_ok=True)
for day, (title, rows) in REPORTS.items():
    lines = [f"# Day {day}: {title}", "", "| Tasks | Status | Evidence |", "|---|---|---|"]
    lines += [f"| {t} | {s} | {e} |" for t, s, e in rows]
    lines += [
        "",
        "## Blockers and notes",
        "- (write what actually blocked you)",
        "",
        "## Reproduce",
        "`python -m pytest -q` and the commands in README.md",
    ]
    (out / f"day{day}.md").write_text("\n".join(lines), encoding="utf-8")
print("wrote", len(REPORTS), "reports")
