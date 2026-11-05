from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "incident_risk"


def test_only_final_evaluation_scores_test():
    offenders = []
    for f in SRC.glob("*.py"):
        if f.name in {"final_evaluation.py", "splits.py"}:
            continue
        text = f.read_text()
        if (
            "X_te" in text
            or "test_predictions" in text
            or "chronological_split" in text
            and ", test =" in text
        ):
            offenders.append(f.name)
    assert not offenders, f"These modules score the test set: {offenders}"
