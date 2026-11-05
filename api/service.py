import hashlib
import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from api import metrics
from api.audit import audit
from api.errors import ApiError
from incident_risk.alert_policy import Suppressor
from incident_risk.config import ROOT

RETRY_SECONDS = 10  # do not hammer the disk when the artifact is broken


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _vocab(bundle) -> dict[str, set[str]]:
    """Category values the preprocessor saw in TRAINING (empty for non-pipeline baselines)."""
    try:
        prep = bundle["model"].base.named_steps["prep"]
        cols = {name: c for name, _, c in prep.transformers_}["cat"]
        enc = prep.named_transformers_["cat"].named_steps["onehot"]
        return {c: {str(v) for v in cats} for c, cats in zip(cols, enc.categories_)}
    except Exception:
        return {}


class ModelService:
    def __init__(self, cfg: dict):
        api = cfg.get("api", {})
        self.features = cfg["features"]["numeric"] + cfg["features"]["categorical"]
        self.categorical = cfg["features"]["categorical"]
        self.policy = cfg["policy"]
        self.strict = api.get("strict_categories", False)
        self.require_manifest = api.get("require_manifest", True)
        self.load_lock, self.sup_lock = threading.Lock(), threading.Lock()
        self.configure(ROOT / cfg["paths"]["model"], ROOT / "models/artifact_manifest.json")

    def configure(self, model_path, manifest_path) -> None:
        self.model_path, self.manifest_path = Path(model_path), Path(manifest_path)
        self.bundle, self.vocab, self.last_error, self.loaded_at = None, {}, None, None
        self._next_retry = 0.0
        self.suppressor = Suppressor(self.policy["suppression_cooldown_minutes"])

    def load(self) -> bool:
        try:
            if not self.model_path.exists():
                raise FileNotFoundError("model file not found (run make train && make register)")
            version = "unversioned"
            if self.manifest_path.exists():
                m = json.loads(self.manifest_path.read_text())
                if m["model_sha256"] != sha256(self.model_path):  # verified BEFORE unpickling
                    raise ValueError("model checksum differs from manifest")
                version = m["version"]
            elif self.require_manifest:
                raise FileNotFoundError("artifact manifest missing")
            bundle = joblib.load(self.model_path)
            if list(bundle["meta"]["features"]) != self.features:
                raise ValueError("feature order differs from config")
            bundle["version"] = version
            self.bundle, self.vocab, self.last_error = bundle, _vocab(bundle), None
            self.loaded_at = datetime.now(UTC).isoformat(timespec="seconds")
            metrics.MODEL_LOADED.set(1)
            metrics.MODEL_INFO.clear()
            metrics.MODEL_INFO.labels(version, bundle["meta"]["selected_model"]).set(1)
            return True
        except Exception as e:
            self.bundle, self.last_error = None, f"{type(e).__name__}: {str(e)[:200]}"
            metrics.MODEL_LOAD_FAILURES.inc()
            metrics.MODEL_LOADED.set(0)
            audit(event="model_load_failed", error=self.last_error)
            return False
        finally:
            self._next_retry = time.monotonic() + RETRY_SECONDS

    def get(self) -> dict:
        if self.bundle is None:
            with self.load_lock:
                if self.bundle is None and time.monotonic() >= self._next_retry:
                    self.load()
        if self.bundle is None:
            raise ApiError(
                503, "model_unavailable", "Model artifact is missing, corrupted or incompatible."
            )
        return self.bundle

    def unknown_categories(self, rows: list[dict]) -> list[list[str]]:
        return [
            [c for c in self.categorical if self.vocab.get(c) and str(r[c]) not in self.vocab[c]]
            for r in rows
        ]

    def score(self, rows: list[dict], kind: str) -> np.ndarray:
        bundle = self.get()
        X = pd.DataFrame(rows)[self.features]  # also drops event_id
        t0 = time.perf_counter()
        try:
            p = np.asarray(bundle["model"].predict_proba(X)[:, 1], dtype=float)
        except Exception as e:
            audit(event="scoring_failed", error=f"{type(e).__name__}: {str(e)[:200]}")
            raise ApiError(500, "scoring_failed", "Scoring failed.") from e
        metrics.INFER_SECONDS.labels(kind).observe(time.perf_counter() - t0)
        if not np.isfinite(p).all():
            raise ApiError(500, "scoring_failed", "Model returned a non-finite score.")
        return p
