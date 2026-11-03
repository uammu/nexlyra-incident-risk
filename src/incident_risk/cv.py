import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def gap_rows(df: pd.DataFrame, cfg: dict) -> int:
    """Convert the 60-minute label gap into a number of rows (2 rows at 30-min cadence)."""
    step = df[cfg["data"]["time_col"]].diff().dropna().median()
    return int(np.ceil(pd.Timedelta(minutes=cfg["split"]["gap_minutes"]) / step))


def time_folds(df: pd.DataFrame, cfg: dict, n_splits: int | None = None):
    n = n_splits or cfg["cv"]["n_splits"]
    return list(TimeSeriesSplit(n_splits=n, gap=gap_rows(df, cfg)).split(df))


def assert_chronological(df: pd.DataFrame, folds, cfg: dict) -> None:
    t = cfg["data"]["time_col"]
    gap = pd.Timedelta(minutes=cfg["split"]["gap_minutes"])
    for i, (tr, va) in enumerate(folds):
        assert not (set(tr) & set(va)), f"fold {i}: row overlap"
        assert df[t].iloc[tr].max() + gap <= df[t].iloc[va].min(), f"fold {i}: leaks future"
