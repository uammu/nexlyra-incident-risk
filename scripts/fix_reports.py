from pathlib import Path

NOTES = {
    1: "- Dataset timestamps are ISO format; the loader needed `time_format: ISO8601`.\n- `p95_latency_ms` was flagged by the leakage screen (AUC 0.903) and reviewed in docs/eda_report.md.",
    2: "- pandas 3.1 and scikit-learn 1.9 needed a fix: `RuleBaseline` mixin order.\n- `majority_class` recall at budget is meaningless (all scores equal).",
    3: "- MLflow 3 needed `serialization_format=cloudpickle` for the custom calibrated model.\n- Logistic regression beat tuned boosters; sigmoid kept over isotonic.",
    4: "- Champion is logistic regression, so SHAP uses the linear explainer.\n- PDP needed float columns. SHAP and permutation importance disagree on context features.",
    5: "- `curl` and `findstr` were unavailable; Python was used instead.\n- Benchmark sampled too few rows for batches (fixed).\n- `docker-compose.yml` was emptied by a commit and restored from git.",
    6: "- `train.py` scored the test set on earlier runs (disclosed in docs/model_card.md); the formal single evaluation is reports/final/FINAL_TEST_LOCK.json.\n- Evidently is incompatible with pandas 3.1; PSI and KS reports used instead.\n- Clean-install check (task 46) was blocked by a venv permission error; training was reproduced in the working folder with identical metrics.",
}
for d, note in NOTES.items():
    p = Path(f"docs/daily_reports/day{d}.md")
    t = p.read_text(encoding="utf-8").replace("- (write what actually blocked you)", note)
    if d == 6:
        t = t.replace("| 46 | Not run |", "| 46 | Partial |")
    p.write_text(t, encoding="utf-8")
print("done")
