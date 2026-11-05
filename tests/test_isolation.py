from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "incident_risk"


def test_only_final_evaluation_scores_test():
    offenders = [
        f.name
        for f in SRC.glob("*.py")
        if f.name != "final_evaluation.py"
        and ("X_te" in f.read_text() or "test_predictions" in f.read_text())
    ]
    assert not offenders, f"These modules score the test set: {offenders}"
