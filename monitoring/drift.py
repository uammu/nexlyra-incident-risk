import numpy as np
import pandas as pd


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index: <0.1 stable, 0.1-0.25 watch, >0.25 investigate."""
    edges = np.unique(np.quantile(expected.dropna(), np.linspace(0, 1, bins + 1)))
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(np.clip(actual, edges[0], edges[-1]), edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_table(train: pd.DataFrame, recent: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = [{"feature": c, "psi": psi(train[c], recent[c])} for c in cols]
    out = pd.DataFrame(rows)
    out["status"] = pd.cut(
        out["psi"], [-1, 0.1, 0.25, 99], labels=["stable", "watch", "investigate"]
    )
    return out
