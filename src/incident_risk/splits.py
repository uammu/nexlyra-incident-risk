import json

import pandas as pd

from .config import ROOT


def chronological_split(df: pd.DataFrame, cfg: dict):
    """Oldest -> train, middle -> val, newest -> test, with a label-availability gap."""
    t = cfg["data"]["time_col"]
    gap = pd.Timedelta(minutes=cfg["split"].get("gap_minutes", 0))
    n = len(df)
    i_va = int(n * cfg["split"]["train_frac"])
    i_te = i_va + int(n * cfg["split"]["val_frac"])
    # Drop rows whose 60-min outcome window reaches into the next period.
    train = df.iloc[:i_va]
    train = train[train[t] <= df[t].iloc[i_va] - gap]
    val = df.iloc[i_va:i_te]
    val = val[val[t] <= df[t].iloc[i_te] - gap]
    test = df.iloc[i_te:]
    return train, val, test


def save_boundaries(train, val, test, cfg: dict) -> dict:
    t = cfg["data"]["time_col"]
    info = {
        name: {"rows": len(p), "start": str(p[t].min()), "end": str(p[t].max())}
        for name, p in [("train", train), ("val", val), ("test", test)]
    }
    out = ROOT / cfg["paths"]["reports"]
    out.mkdir(exist_ok=True, parents=True)
    (out / "split_boundaries.json").write_text(json.dumps(info, indent=2))
    return info
