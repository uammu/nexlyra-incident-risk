import hashlib

import pandas as pd

from .config import ROOT


def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(cfg: dict) -> pd.DataFrame:
    d = cfg["data"]
    df = pd.read_csv(ROOT / d["path"])
    df[d["time_col"]] = pd.to_datetime(df[d["time_col"]], format=d["time_format"], errors="raise")
    return df.sort_values(d["time_col"], kind="stable").reset_index(drop=True)


def validate_schema(df: pd.DataFrame, cfg: dict) -> list[str]:
    d, f = cfg["data"], cfg["features"]
    problems = []
    needed = [d["id_col"], d["time_col"], d["target"], *f["numeric"], *f["categorical"]]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return [f"Missing columns: {missing}"]
    if df[d["id_col"]].duplicated().any():
        problems.append("Duplicate event_id values")
    if df[d["time_col"]].duplicated().any():
        problems.append("Duplicate timestamps (check if expected)")
    if not df[d["time_col"]].is_monotonic_increasing:
        problems.append("Timestamps not sorted")
    if not set(df[d["target"]].dropna().unique()) <= {0, 1}:
        problems.append("Target is not binary 0/1")
    for c in f["percent_cols"]:
        bad = ((df[c] < 0) | (df[c] > 100)).sum()
        if bad:
            problems.append(f"{c}: {bad} values outside 0-100")
    for c in f.get("fraction_cols", []):
        bad = ((df[c] < 0) | (df[c] > 1)).sum()
        if bad:
            problems.append(f"{c}: {bad} values outside 0-1")
    for c in f["numeric"]:
        if (df[c] < 0).any():
            problems.append(f"{c}: negative values")
    return problems
