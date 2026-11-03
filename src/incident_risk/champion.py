import joblib

from .config import ROOT
from .data import load_raw
from .splits import chronological_split


def features(cfg: dict) -> list[str]:
    return cfg["features"]["numeric"] + cfg["features"]["categorical"]


def load_champion(cfg: dict, source: str = "file") -> dict:
    """source='registry' loads the model tagged @champion in MLflow (threshold/meta still from the bundle)."""
    bundle = joblib.load(ROOT / cfg["paths"]["model"])
    if source == "registry":
        import mlflow

        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        bundle["model"] = mlflow.sklearn.load_model("models:/nexlyra-incident-risk@champion")
    return bundle


def verify_feature_order(bundle: dict, cfg: dict) -> bool:
    feats = features(cfg)
    assert list(bundle["meta"]["features"]) == feats, "bundle feature order differs from config"
    base = bundle["model"].base
    if hasattr(base, "named_steps"):
        seen = list(base.named_steps["prep"].feature_names_in_)
        assert seen == feats, f"preprocessor was fitted on a different order: {seen}"
    return True


def validation_holdout(cfg: dict, bundle: dict, full: bool = False):
    """Second half of validation (threshold-selection half), or all of validation. Never test."""
    _, val, _ = chronological_split(load_raw(cfg), cfg)
    hold = (val if full else val.iloc[len(val) // 2 :]).copy().reset_index(drop=True)
    feats = features(cfg)
    hold["risk"] = bundle["model"].predict_proba(hold[feats])[:, 1]
    hold["alert"] = hold["risk"] >= bundle["threshold"]
    return hold
