import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

from .champion import features, load_champion, validation_holdout, verify_feature_order
from .config import ROOT, load_config
from .data import load_raw
from .splits import chronological_split

warnings.filterwarnings("ignore")


def to_original(names, cfg) -> np.ndarray:
    """Map transformed names (one-hot, missing-indicator) back to the raw column they came from."""
    cols, out = features(cfg), []
    for n in names:
        key = n.replace("missingindicator_", "")
        match = [c for c in cols if key == c or key.startswith(c + "_")]
        out.append(max(match, key=len) if match else n)
    return np.array(out)


def is_explainable(clf) -> bool:
    return hasattr(clf, "feature_importances_") or hasattr(clf, "coef_")


def shap_matrix(clf, X, background=None) -> np.ndarray:
    """Tree model -> TreeExplainer. Linear model -> exact LinearExplainer (log-odds units)."""
    if hasattr(clf, "coef_"):
        sv = shap.LinearExplainer(clf, X if background is None else background).shap_values(X)
    else:
        sv = shap.TreeExplainer(clf).shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    return sv[:, :, 1] if sv.ndim == 3 else sv


def main():
    cfg = load_config()
    t, seed, f = cfg["data"]["target"], cfg["seed"], cfg["features"]
    rep = ROOT / cfg["paths"]["reports"] / "explain"
    rep.mkdir(parents=True, exist_ok=True)
    feats = features(cfg)

    b = load_champion(cfg)
    verify_feature_order(b, cfg)  # task 4
    hold = validation_holdout(cfg, b)
    summary = {"feature_order_ok": True, "champion": b["meta"]["selected_model"]}

    try:  # task 3
        reg = load_champion(cfg, "registry")["model"]
        summary["registry_matches_file"] = bool(
            np.allclose(reg.predict_proba(hold[feats])[:, 1], hold["risk"].values, atol=1e-9)
        )
    except Exception as e:
        summary["registry_matches_file"] = f"could not load registry model: {str(e)[:120]}"

    base = b["model"].base
    if not hasattr(base, "named_steps") or not is_explainable(base.named_steps["clf"]):
        print("Champion is not a tree or linear model; skipping SHAP.")
        return
    prep, clf = base.named_steps["prep"], base.named_steps["clf"]
    names = prep.get_feature_names_out()
    orig = to_original(names, cfg)

    train, _, _ = chronological_split(load_raw(cfg), cfg)  # background = TRAIN sample
    bg = prep.transform(train[feats].sample(min(500, len(train)), random_state=seed))
    X = prep.transform(hold[feats])
    sv = shap_matrix(clf, X, bg)

    # --- global importance + summary plot (tasks 5, 6) ---
    shap.summary_plot(sv, X, feature_names=names, show=False)
    plt.savefig(rep / "shap_summary.png", bbox_inches="tight", dpi=150)
    plt.close()
    imp_shap = (
        pd.Series(np.abs(sv).mean(0), index=orig)
        .groupby(level=0)
        .sum()
        .sort_values(ascending=False)
    )

    # --- permutation importance vs SHAP (task 12) ---
    perm = permutation_importance(
        base,
        hold[feats],
        hold[t],
        scoring="average_precision",
        n_repeats=10,
        random_state=seed,
        n_jobs=1,
    )
    imp = pd.DataFrame(
        {
            "shap_mean_abs": imp_shap,
            "perm_pr_auc_drop": pd.Series(perm.importances_mean, index=feats),
        }
    )
    imp["shap_rank"] = imp.shap_mean_abs.rank(ascending=False)
    imp["perm_rank"] = imp.perm_pr_auc_drop.rank(ascending=False)
    imp.sort_values("shap_rank").to_csv(rep / "importance_shap_vs_permutation.csv")
    summary["shap_perm_spearman"] = float(spearmanr(imp.shap_rank, imp.perm_rank)[0])

    # --- dependence plots in raw units (task 7) ---
    top = [c for c in imp_shap.index if c in f["numeric"]][:4]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, c in zip(axes.ravel(), top):
        j = int(np.where(names == c)[0][0])
        ax.scatter(hold[c], sv[:, j], s=6, alpha=0.5)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel(c)
        ax.set_ylabel("SHAP (log-odds)")
    plt.tight_layout()
    plt.savefig(rep / "shap_dependence.png", dpi=150)
    plt.close()

    # --- twenty individual explanations (tasks 8 to 11) ---
    y = hold[t]
    groups = {
        "high_risk": hold.nlargest(5, "risk").index,
        "low_risk": hold.nsmallest(5, "risk").index,
        "false_positive": hold[hold["alert"] & (y == 0)].nlargest(5, "risk").index,
        "false_negative": hold[~hold["alert"] & (y == 1)].nlargest(5, "risk").index,
    }

    def fmt(i, j):
        n = names[j]
        shown = f"{n}={hold.at[i, n]:.3g}" if n in f["numeric"] else n
        return f"{shown} ({sv[i, j]:+.2f})"

    rows = []
    for g, idx in groups.items():
        for i in idx:
            order = np.argsort(-np.abs(sv[i]))[:5]
            rows.append(
                {
                    "group": g,
                    "event_id": hold.at[i, cfg["data"]["id_col"]],
                    "timestamp": hold.at[i, cfg["data"]["time_col"]],
                    "risk": round(hold.at[i, "risk"], 4),
                    "actual": int(y[i]),
                    "top_drivers": "; ".join(fmt(i, j) for j in order),
                }
            )
    pd.DataFrame(rows).to_csv(rep / "local_explanations.csv", index=False)

    # --- stability across samples (task 14) ---
    rng, ranks = np.random.default_rng(seed), []
    for _ in range(10):
        idx = rng.choice(len(X), size=min(500, len(X)), replace=False)
        ranks.append(
            pd.Series(np.abs(sv[idx]).mean(0), index=orig)
            .groupby(level=0)
            .sum()
            .reindex(imp_shap.index)
        )
    pair = [spearmanr(ranks[a], ranks[c])[0] for a in range(10) for c in range(a + 1, 10)]
    summary["shap_rank_stability_mean_spearman"] = float(np.mean(pair))
    summary["shap_rank_stability_min_spearman"] = float(np.min(pair))

    # --- correlated-feature caveat (task 13) ---
    corr = hold[f["numeric"]].corr()
    pairs = [
        {"a": a, "b": c, "r": round(corr.loc[a, c], 3)}
        for i, a in enumerate(corr.columns)
        for c in corr.columns[i + 1 :]
        if abs(corr.loc[a, c]) >= 0.6
    ]
    pd.DataFrame(pairs, columns=["a", "b", "r"]).to_csv(rep / "correlated_pairs.csv", index=False)
    summary["correlated_pairs_ge_0.6"] = len(pairs)

    # --- PDP + ICE (tasks 15, 16) ---
    sub = hold[feats].sample(min(300, len(hold)), random_state=seed)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    PartialDependenceDisplay.from_estimator(
        base, sub, top, kind="both", subsample=100, random_state=seed, ax=axes.ravel()[: len(top)]
    )
    plt.tight_layout()
    plt.savefig(rep / "pdp_ice.png", dpi=150)
    plt.close()

    (rep / "explain_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(imp.sort_values("shap_rank").round(4))


if __name__ == "__main__":
    main()
