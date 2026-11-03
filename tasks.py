"""Windows-friendly replacement for the Makefile.  Usage: python tasks.py <task> [<task> ...]"""

import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def run(cmd: list[str]) -> None:
    cmd = [PY if c == "python" else c for c in cmd]
    print(">", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode:
        sys.exit(r.returncode)


def pin_api() -> None:
    out = subprocess.run(
        [PY, "-m", "pip", "freeze", "--exclude-editable"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    keep = [
        l
        for l in out.splitlines()
        if re.match(r"(?i)^(scikit-learn|lightgbm|xgboost|pandas|numpy|scipy|joblib)==", l)
    ]
    (ROOT / "requirements-api.lock").write_text("\n".join(keep) + "\n")
    print("wrote requirements-api.lock:", *keep, sep="\n  ")


def freeze() -> None:
    out = subprocess.run(
        [PY, "-m", "pip", "freeze", "--exclude-editable"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    (ROOT / "requirements.lock").write_text(out)
    print("wrote requirements.lock")


def smoke() -> None:
    base = "http://localhost:8000"
    for path in ("/health", "/ready"):
        print(path, urllib.request.urlopen(base + path, timeout=5).read().decode())
    body = (ROOT / "examples" / "single.json").read_bytes()
    req = urllib.request.Request(
        base + "/predict", data=body, headers={"Content-Type": "application/json"}
    )
    print("/predict", urllib.request.urlopen(req, timeout=5).read().decode())


# A task is a list of steps. A step is a command list, a function, or "task:<other>".
T = {
    "validate": [["python", "scripts/validate_data.py"]],
    "train": [["python", "-m", "incident_risk.train"]],
    "tune": [["python", "-m", "incident_risk.tune"]],
    "calib": [["python", "-m", "incident_risk.calibration_compare"]],
    "champion": [["python", "-m", "incident_risk.champion_report"]],
    "ablate": [["python", "-m", "incident_risk.experiments", "--model", "lightgbm"]],
    "register": [["python", "-m", "incident_risk.register"]],
    "explain": [["python", "-m", "incident_risk.explain"]],
    "robust": [["python", "-m", "incident_risk.robustness"]],
    "subgroup": [["python", "-m", "incident_risk.subgroup"]],
    "uncertainty": [["python", "-m", "incident_risk.uncertainty"]],
    "governance": ["task:explain", "task:robust", "task:subgroup", "task:uncertainty"],
    "test": [["python", "-m", "pytest", "-q"]],
    "lint": [
        ["python", "-m", "ruff", "check", "."],
        ["python", "-m", "ruff", "format", "--check", "."],
    ],
    "format": [["python", "-m", "ruff", "format", "."]],
    "api": [["python", "-m", "uvicorn", "api.main:app", "--reload", "--port", "8000"]],
    "dash": [["python", "-m", "streamlit", "run", "dashboard/app.py"]],
    "mlflow": [["python", "-m", "mlflow", "ui", "--backend-store-uri", "sqlite:///mlflow.db"]],
    "examples": [["python", "scripts/make_examples.py"]],
    "dashboard-data": [["python", "scripts/export_dashboard_data.py"]],
    "bench": [["python", "scripts/benchmark.py", "--url", "http://localhost:8000"]],
    "pin-api": [pin_api],
    "freeze": [freeze],
    "up": [["docker", "compose", "up", "--build", "-d"]],
    "down": [["docker", "compose", "down"]],
    "smoke": [smoke],
    "release": ["task:register", ["python", "scripts/model_release.py", "archive"]],
    "reproduce": ["task:validate", "task:tune", "task:train", "task:champion", "task:register"],
}


def do(name: str) -> None:
    if name not in T:
        sys.exit(f"Unknown task '{name}'. Available: {', '.join(sorted(T))}")
    print(f"\n=== {name} ===")
    for step in T[name]:
        if callable(step):
            step()
        elif isinstance(step, str) and step.startswith("task:"):
            do(step[5:])
        else:
            run(step)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python tasks.py <task> ...\nTasks: " + ", ".join(sorted(T)))
    for t in sys.argv[1:]:
        do(t)
