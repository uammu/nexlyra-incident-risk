import json
from datetime import datetime

import numpy as np

from incident_risk.champion import features, load_champion, validation_holdout
from incident_risk.config import ROOT, load_config
from incident_risk.explain import shap_matrix

def main():
    cfg = load_config()
    t, tc, ic = cfg["data"]["target"], cfg["data"]["time_col"], cfg["data"]["id_col"]
    b, feats, num = load_champion(cfg), features(cfg), cfg["features"]["numeric"]
    df = validation_holdout(cfg, b, full=True)

    drivers = ["n/a (champion is not a tree model)"] * len(df)
    base = b["model"].base
    if hasattr(base, "named_steps") and hasattr(base.named_steps["clf"], "feature_importances_"):
        prep, clf = base.named_steps["prep"], base.named_steps["clf"]
        names = prep.get_feature_names_out()
        sv = shap_matrix(clf, prep.transform(df[feats]))
        drivers = []
        for i in range(len(df)):
            top = np.argsort(-np.abs(sv[i]))[:5]
            drivers.append("; ".join(
                (f"{names[j]}={df.at[i, names[j]]:.3g}" if names[j] in num else str(names[j]))
                + f" ({sv[i, j]:+.2f})" for j in top))
    df["top_drivers"] = drivers

    out = df.rename(columns={t: "actual"})
    keep = [ic, tc, "service_name", "environment", "region", "customer_tier", "deployment_type",
            "risk", "alert", "actual", "top_drivers"] + num
    out[keep].to_csv(ROOT / cfg["paths"]["reports"] / "dashboard_scores.csv", index=False)
    meta = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "model": b["meta"]["selected_model"],
            "threshold": b["threshold"], "n": len(df), "period_start": str(df[tc].min()),
            "period_end": str(df[tc].max())}
    (ROOT / cfg["paths"]["reports"] / "dashboard_meta.json").write_text(json.dumps(meta, indent=2))
    print(meta)

if __name__ == "__main__":
    main()