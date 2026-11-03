import pandas as pd

from incident_risk.splits import chronological_split


def test_order_gap_and_no_overlap(fake_df, cfg):
    tr, va, te = chronological_split(fake_df, cfg)
    t, ids = cfg["data"]["time_col"], cfg["data"]["id_col"]
    gap = pd.Timedelta(minutes=cfg["split"]["gap_minutes"])
    assert tr[t].max() + gap <= va[t].min()
    assert va[t].max() + gap <= te[t].min()
    assert not (set(tr[ids]) & set(va[ids])) and not (set(va[ids]) & set(te[ids]))
    assert len(tr) + len(va) + len(te) <= len(fake_df)
