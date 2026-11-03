import datetime as dt
import hashlib
import json
import subprocess

import joblib
import mlflow
from mlflow.tracking import MlflowClient

from .config import ROOT, load_config

REG_NAME = "nexlyra-incident-risk"


def sha(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def git_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT)
            .decode()
            .strip()
        )
    except Exception:
        return "nogit"


def main():
    cfg = load_config()
    model_path = ROOT / cfg["paths"]["model"]
    b = joblib.load(model_path)
    version = f"{dt.date.today():%Y%m%d}-{git_hash()}"

    prep_path = ROOT / "models/preprocessor.joblib"
    base = b["model"].base
    if hasattr(base, "named_steps"):
        joblib.dump(base.named_steps["prep"], prep_path)  # fitted on TRAIN only

    manifest = {
        "version": version,
        "model": b["meta"]["selected_model"],
        "model_sha256": sha(model_path),
        "preprocessor_sha256": sha(prep_path) if prep_path.exists() else None,
        "data_sha256": b["meta"]["data_sha256"],
        "threshold": b["threshold"],
        "split": b["meta"]["split"],
        "seed": b["meta"]["seed"],
        "reproduce": "make validate && make tune && make train && make champion",
    }
    (ROOT / "models/artifact_manifest.json").write_text(json.dumps(manifest, indent=2))

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    with mlflow.start_run(run_name=f"champion_{version}"):
        mlflow.log_params({"version": version, "model": manifest["model"]})
        mlflow.log_artifact(str(ROOT / "models/artifact_manifest.json"))
        info = mlflow.sklearn.log_model(
            b["model"],
            name="model",
            registered_model_name=REG_NAME,
            serialization_format="cloudpickle",
        )
    MlflowClient().set_registered_model_alias(REG_NAME, "champion", info.registered_model_version)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
