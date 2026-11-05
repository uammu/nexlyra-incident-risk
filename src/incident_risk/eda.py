import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .config import ROOT, load_config
from .data import load_raw
from .splits import chronological_split

warnings.filterwarnings("ignore")


def bar_rate(ax, g, title):
    ax.bar(g.index.astype(str), g["rate"])
    ax.axhline(g.attrs["base"], color="k", ls=":", lw=1)
    ax.set_title(title, fontsize=9)
    ax.tick_params(axis="x", rotation=45, labelsize=7)


def main():
    cfg = load_config()
    f, t, tc = cfg["features"], cfg["data"]["target"], cfg["data"]["time_col"]
    num, cat = f["numeric"], f["categorical"]
    out = ROOT / cfg["paths"]["reports"] / "eda"
    out.mkdir(parents=True, exist_ok=True)
    train, _, _ = chronological_split(load_raw(cfg), cfg)
    base = float(train[t].mean())
    notes = [
        f"EDA uses the TRAIN period only: {train[tc].min()} to {train[tc].max()} ({len(train):,} rows).",
        f"Incident rate in train: {base:.4f} ({int(train[t].sum())} incidents).",
    ]

    # 1. target rate by month (drift in class balance)
    m = train.groupby(train[tc].dt.to_period("M"))[t].agg(rate="mean", n="size", incidents="sum")
    m.to_csv(out / "target_by_month.csv")
    m.attrs["base"] = base
    fig, ax = plt.subplots(figsize=(6, 3))
    bar_rate(ax, m, "Incident rate by month (train)")
    plt.tight_layout()
    plt.savefig(out / "target_by_month.png", dpi=150)
    plt.close()
    notes.append(f"Monthly incident rate ranges {m.rate.min():.3f} to {m.rate.max():.3f}.")

    # 2. target rate by category
    fig, axes = plt.subplots(1, len(cat), figsize=(4 * len(cat), 3.2))
    rows = []
    for ax, c in zip(axes, cat):
        g = train.groupby(c)[t].agg(rate="mean", n="size", incidents="sum")
        g.attrs["base"] = base
        bar_rate(ax, g, f"Rate by {c}")
        rows.append(g.reset_index().rename(columns={c: "level"}).assign(column=c))
        notes.append(f"{c}: rate spread {g.rate.min():.3f} to {g.rate.max():.3f}.")
    pd.concat(rows).to_csv(out / "target_by_category.csv", index=False)
    plt.tight_layout()
    plt.savefig(out / "target_by_category.png", dpi=150)
    plt.close()

    # 3. distributions
    desc = train[num].describe(percentiles=[0.01, 0.5, 0.99]).T
    desc["skew"] = train[num].skew()
    desc.to_csv(out / "numeric_summary.csv")
    fig, axes = plt.subplots(3, 4, figsize=(14, 8))
    for ax, c in zip(axes.ravel(), num):
        ax.hist(train[c], bins=40)
        ax.set_title(c, fontsize=8)
        if desc.loc[c, "skew"] > 2:
            ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(out / "distributions.png", dpi=130)
    plt.close()

    # 4. correlations and multicollinearity
    corr = train[num].corr(method="spearman")
    corr.to_csv(out / "correlation_spearman.csv")
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(num)), num, rotation=90, fontsize=7)
    ax.set_yticks(range(len(num)), num, fontsize=7)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(out / "correlation.png", dpi=150)
    plt.close()
    pairs = [
        (a, b, round(corr.loc[a, b], 3))
        for i, a in enumerate(num)
        for b in num[i + 1 :]
        if abs(corr.loc[a, b]) >= 0.6
    ]
    notes.append(f"Feature pairs with |Spearman| >= 0.6: {pairs if pairs else 'none'}.")
    Z = (train[num] - train[num].mean()) / train[num].std()
    Z.insert(0, "const", 1.0)
    vif = pd.Series(
        [variance_inflation_factor(Z.values, i) for i in range(1, Z.shape[1])], index=num
    )
    vif.rename("vif").to_csv(out / "vif.csv")
    notes.append(
        f"Max VIF = {vif.max():.2f} ({vif.idxmax()}). Above 5 would signal multicollinearity."
    )

    # 5. outliers
    rows = []
    for c in num:
        q1, q3 = train[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        far = ((train[c] < q1 - 3 * iqr) | (train[c] > q3 + 3 * iqr)).mean()
        rows.append(
            {
                "feature": c,
                "share_beyond_3iqr": far,
                "p99": train[c].quantile(0.99),
                "max": train[c].max(),
            }
        )
    pd.DataFrame(rows).to_csv(out / "outliers.csv", index=False)

    # 6. seasonality
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    hr = train.groupby(train[tc].dt.hour)[t].mean()
    dw = train.groupby(train[tc].dt.dayofweek)[t].mean()
    axes[0].plot(hr.index, hr.values, marker="o")
    axes[0].set_title("Incident rate by hour")
    axes[1].plot(dw.index, dw.values, marker="o")
    axes[1].set_title("by day of week (0 = Mon)")
    plt.tight_layout()
    plt.savefig(out / "seasonality.png", dpi=150)
    plt.close()
    notes.append(
        f"Hourly rate range {hr.min():.3f} to {hr.max():.3f}; weekday range {dw.min():.3f} to {dw.max():.3f}."
    )

    # 7. early vs late (inside train only)
    h = len(train) // 2
    early, late = train.iloc[:h], train.iloc[h:]
    ks = pd.DataFrame(
        [
            {
                "feature": c,
                "ks_stat": ks_2samp(early[c], late[c]).statistic,
                "early_mean": early[c].mean(),
                "late_mean": late[c].mean(),
            }
            for c in num
        ]
    )
    ks.to_csv(out / "early_vs_late.csv", index=False)
    notes.append(
        f"Early vs late target rate: {early[t].mean():.3f} vs {late[t].mean():.3f}. "
        f"Largest KS shift: {ks.loc[ks.ks_stat.idxmax(), 'feature']} ({ks.ks_stat.max():.3f})."
    )

    # 8. proxy-leakage screen: how well does ONE feature rank the target?
    uni = pd.Series({c: roc_auc_score(train[t], train[c]) for c in num})
    uni = pd.DataFrame({"auc": uni, "strength": (uni - 0.5).abs() + 0.5}).sort_values(
        "strength", ascending=False
    )
    uni.to_csv(out / "univariate_auc.csv")
    flagged = uni[uni.strength > 0.90].index.tolist()
    notes.append(
        "Leakage screen (single-feature ROC-AUC): "
        + (
            f"FLAG {flagged}"
            if flagged
            else "no feature above 0.90, none is a near-copy of the target"
        )
        + f". Strongest: {uni.index[0]} ({uni.strength.iloc[0]:.3f})."
    )
    notes.append("event_id is excluded from features by the preprocessor (remainder='drop').")

    (out / "eda_summary.md").write_text(
        "# EDA summary (auto-generated)\n\n" + "\n".join(f"- {n}" for n in notes)
    )
    print("\n".join(notes))
    print("\nFiles in", out)


if __name__ == "__main__":
    main()
