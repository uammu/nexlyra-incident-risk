import numpy as np

from incident_risk.features import build_preprocessor


def test_unseen_category_and_missing(fake_df, cfg):
    f = cfg["features"]
    prep = build_preprocessor(f["numeric"], f["categorical"]).fit(fake_df)
    new = fake_df.head(5).copy()
    new[f["categorical"][0]] = "BRAND_NEW_SERVICE"
    new.loc[new.index[0], f["numeric"][0]] = np.nan
    out = prep.transform(new)
    assert out.shape[0] == 5 and not np.isnan(out).any()


def test_event_id_not_in_features(fake_df, cfg):
    prep = build_preprocessor(cfg["features"]["numeric"], cfg["features"]["categorical"]).fit(
        fake_df
    )
    assert not any("event_id" in n for n in prep.get_feature_names_out())
