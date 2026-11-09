import json
import os
import platform
import subprocess
import sys
from importlib.metadata import version

from incident_risk.config import ROOT

pkgs = [
    "pandas",
    "numpy",
    "scikit-learn",
    "lightgbm",
    "xgboost",
    "optuna",
    "shap",
    "mlflow",
    "fastapi",
    "pydantic",
    "streamlit",
    "scipy",
    "statsmodels",
    "pytest",
    "ruff",
]
try:
    commit = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
    )
except (OSError, subprocess.CalledProcessError):
    commit = "nogit"
env = {
    "python": sys.version,
    "platform": platform.platform(),
    "cpu_count": os.cpu_count(),
    "git_commit": commit,
    "packages": {p: version(p) for p in pkgs},
}
(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports" / "environment.json").write_text(json.dumps(env, indent=2))
print(json.dumps(env, indent=2))
