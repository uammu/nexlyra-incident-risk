import numpy as np
import pandas as pd
import pytest

from incident_risk.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture
def fake_df(cfg):
    rng = np.random.default_rng(0)
    n = 1000
    d, f = cfg["data"], cfg["features"]
    df = pd.DataFrame(
        {
            d["id_col"]: range(n),
            d["time_col"]: pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC"),
            d["target"]: rng.integers(0, 2, n),
        }
    )
    for c in f["numeric"]:
        df[c] = rng.uniform(0, 100, n)
    for c in f["categorical"]:
        df[c] = rng.choice(["a", "b", "c"], n)
    return df


@pytest.fixture
def good_payload():
    return dict(
        service_name="inventory",
        environment="production",
        region="ap-south-1",
        deployment_type="none",
        customer_tier="standard",
        release_age_hours=6.3,
        requests_per_minute=3911,
        p95_latency_ms=55,
        error_rate_pct=0.48,
        cpu_utilization_pct=75.2,
        memory_utilization_pct=64.21,
        db_connections_pct=38.18,
        queue_depth=24,
        dependency_failures_15m=0,
        change_failure_rate_30d=0.177,
        active_alerts=6,
        incident_history_30d=0,
    )
