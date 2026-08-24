"""Turn the backtests and the risk table into the two figures SPEC 8 asks for,
plus a RESULTS.md that records the numbers so they survive the terminal.

  fig1_clustering.png   why Kupiec alone is not enough: three models with
                        near-identical exception RATES, wildly different
                        exception TIMING.
  fig2_limit_curve.png  risk tolerance -> safe oversubscription limit, the
                        curve a client actually buys.

    python par/report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import backtest as bt  # noqa: E402
import observability as obs  # noqa: E402
import par  # noqa: E402

HERE = Path(__file__).parent
FIG_DIR = HERE / "figures"
RESULTS_MD = HERE / "RESULTS.md"

SWEEP = [0.95, 0.99, 0.999]
# lead times in 5-minute steps, evaluated non-overlapping (SPEC 5.1)
HORIZONS = [(1, "5min"), (3, "15min"), (6, "30min"), (12, "1h"),
            (36, "3h"), (72, "6h")]
C = {"empirical": "#c44e52", "fhs": "#dd8452", "fhs_ar": "#4c72b0"}


def clustering_stats(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Exception rate vs exception clustering, per model."""
    rows = []
    for model in par.MODELS:
        fc = par.forecast_panel(df, model, alpha).dropna(subset=["var"])
        e = (fc["y"] > fc["var"]).to_numpy()
        ind = bt.christoffersen_independence(e)
        p01 = ind["n01"] / max(ind["n00"] + ind["n01"], 1)
        p11 = ind["n11"] / max(ind["n10"] + ind["n11"], 1)
        per_pdu = par.backtest_panel(fc, alpha, model)
        per_pdu = per_pdu[per_pdu.pdu != "POOLED"]
        rows.append(dict(model=model, alpha=alpha, rate=e.mean(),
                         p01=p01, p11=p11, ratio=p11 / p01 if p01 else np.nan,
                         cc_pass=int((per_pdu.cc_p > 0.05).sum()),
                         n_pdu=len(per_pdu)))
    return pd.DataFrame(rows)


def fig_clustering(stats: pd.DataFrame) -> None:
    """Same rate, different timing -- the case for Christoffersen."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    models = list(par.MODELS)
    x = np.arange(len(models))
    a99 = stats[stats.alpha == 0.99].set_index("model").loc[models]

    ax = axes[0]
    ax.bar(x, a99["rate"] * 100, color=[C[m] for m in models])
    ax.axhline(1.0, ls="--", c="k", lw=1, label="nominal 1%")
    ax.set_title("Exception rate\n(Kupiec sees only this)")
    ax.set_ylabel("% of intervals breaching PaR$_{99}$")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(x - 0.19, a99["p01"] * 100, 0.38, color="#bbb", label="after no breach")
    ax.bar(x + 0.19, a99["p11"] * 100, 0.38, color=[C[m] for m in models],
           label="after a breach")
    ax.set_title("Exception timing\n(Christoffersen sees this)")
    ax.set_ylabel("P(breach) %")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.bar(x, a99["cc_pass"], color=[C[m] for m in models])
    ax.set_ylim(0, a99["n_pdu"].iloc[0])
    ax.set_title("PDUs passing conditional\ncoverage (of %d)" % a99["n_pdu"].iloc[0])
    ax.set_ylabel("PDUs")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(r"Three models, near-identical exception rates, "
                 r"very different clustering ($\alpha=0.99$)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_clustering.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def limit_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Sweep risk tolerance AND lead time, and record the fleet limits.

    Risk tolerance alone barely moves the answer at h=1: the series has a
    lag-1 autocorrelation of 0.98, so five minutes ahead the load is
    essentially where it already is and every quantile collapses onto the
    current level. Lead time is what actually costs headroom, and it is what
    a client is buying -- "how much warning do I get, and what does it cost".
    """
    rows = []
    for h, label in HORIZONS:
        for alpha in SWEEP:
            raw = par.forecast_horizon(df, alpha, h).dropna(subset=["var", "anchor"])
            cal = par.calibrate_panel(raw, alpha, h)
            fc = par.apply_multiplier(raw, cal["m"])
            held = fc[fc.ts >= par.CALIB_SPLIT]
            lim = 1.0 / fc.groupby(["cell", "pdu"])["var"].mean() - 1.0
            rows.append(dict(h=h, lead=label, alpha=alpha,
                             m=cal["m"], raw_rate=cal["raw_test"],
                             breach_rate=cal["cal_test"],
                             breaches_per_pdu=cal["breaches_per_pdu"],
                             verdict=cal["verdict"], n_held=len(held),
                             p10=lim.quantile(0.10), median=lim.median(),
                             p90=lim.quantile(0.90), worst=lim.min()))
    return pd.DataFrame(rows)


def fig_limit_curve(curve: pd.DataFrame, table: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    for alpha, mk in zip(SWEEP, ("o-", "s--", "^:")):
        g = curve[curve.alpha == alpha]
        ax.plot(g.lead, g["median"] * 100, mk, label=r"$\alpha$=%g" % alpha)
    g = curve[curve.alpha == 0.99]
    ax.fill_between(g.lead, g.p10 * 100, g.p90 * 100, alpha=0.15, color="#4c72b0")

    # One month cannot validate the long lead times: the held-out half carries
    # ~1 breach per PDU at 3-6h, so those points are drawn but not certified.
    bad = g[g.verdict == "underpowered"]["lead"].tolist()
    if bad:
        ax.axvspan(g["lead"].tolist().index(bad[0]) - 0.5,
                   len(g) - 0.5, color="#c44e52", alpha=0.07)
        ax.text(len(g) - 0.6, ax.get_ylim()[1], "not validated\n(too few breaches)",
                ha="right", va="top", fontsize=7.5, color="#c44e52")
    ax.set_xlabel("lead time")
    ax.set_ylabel("safe oversubscription limit (%)")
    ax.set_title("What warning time costs\n"
                 r"(shaded: fleet 10-90th pct at $\alpha$=0.99)")
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    colors = {"ok": "#4c72b0", "degraded": "#dd8452",
              "tight": "#c44e52", "blind": "#8172b3"}
    for post, g in table.groupby("posture"):
        ax.scatter(g.obs_score, g.limit_adj * 100, s=42, alpha=0.85,
                   color=colors.get(post, "#888"), label=post,
                   edgecolor="white", linewidth=0.6)
    ax.set_xlabel("observability score (trusted share of intervals)")
    ax.set_ylabel("adjusted limit (%)")
    ax.set_title(r"Trust vs headroom, per PDU ($\alpha=0.99$)")
    ax.legend(fontsize=8, title="posture", title_fontsize=8)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_limit_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_results(stats: pd.DataFrame, curve: pd.DataFrame,
                  table: pd.DataFrame, results: pd.DataFrame) -> None:
    pooled = results[results.pdu == "POOLED"]
    md = ["# Results", "",
          "Generated by `par/report.py`. Data: 508,896 rows, 57 PDUs, May 2019.",
          "", "## A-layer backtests (pooled)", "",
          "| model | alpha | rate | expected | Kupiec p | indep p | ES Z2 | PDUs passing CC |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in pooled.iterrows():
        n_ok = int((results[(results.model == r.model) & (results.alpha == r.alpha)
                            & (results.pdu != "POOLED")].cc_p > 0.05).sum())
        md.append("| %s | %.2f | %.4f | %.4f | %.3g | %.3g | %+.3f | %d/57 |"
                  % (r.model, r.alpha, r["rate"], 1 - r.alpha,
                     r.kupiec_p, r.ind_p, r.es_z2, n_ok))

    a99 = stats[stats.alpha == 0.99]
    md += ["", "## Exception clustering", "",
           "| model | P(breach after none) | P(breach after breach) | ratio |",
           "|---|---|---|---|"]
    for _, r in a99.iterrows():
        md.append("| %s | %.4f | %.4f | %.1fx |" % (r.model, r.p01, r.p11, r.ratio))

    md += ["", "## Horizon calibration (out of sample)", "",
           "Multiplier fitted on days 8-18, applied unchanged to days 19-31.",
           "`breaches/PDU` is the calibration signal available in the held-out",
           "half -- below ~2 there is nothing to calibrate against.", "",
           "| lead | alpha | m | rate before | rate after | breaches/PDU | verdict |",
           "|---|---|---|---|---|---|---|"]
    for _, r in curve.iterrows():
        md.append("| %s | %g | %.3f | %.4f | %.4f | %.1f | %s |"
                  % (r.lead, r.alpha, r.m, r.raw_rate, r.breach_rate,
                     r.breaches_per_pdu, r.verdict))

    md += ["", "## Oversubscription limit vs lead time", "",
           "| lead | alpha | tightest PDU | fleet median | fleet p90 | verdict |",
           "|---|---|---|---|---|---|"]
    for _, r in curve.iterrows():
        md.append("| %s | %g | %+.1f%% | %+.1f%% | %+.1f%% | %s |"
                  % (r.lead, r.alpha, 100 * r.worst,
                     100 * r["median"], 100 * r.p90, r.verdict))

    counts = table.posture.value_counts().to_dict()
    md += ["", "## Fleet posture (alpha=0.99)", "",
           "| posture | PDUs |", "|---|---|"]
    md += ["| %s | %d |" % (k, v) for k, v in sorted(counts.items())]
    md += ["", "![clustering](figures/fig1_clustering.png)",
           "", "![limits](figures/fig2_limit_curve.png)", ""]
    RESULTS_MD.write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    df = pd.read_parquet(obs.PARQUET)
    df[obs.FLAG] = df[obs.FLAG].astype(bool)

    print("clustering stats ...")
    stats = pd.concat([clustering_stats(df, a) for a in (0.95, 0.99)],
                      ignore_index=True)
    fig_clustering(stats)

    print("limit curve ...")
    curve = limit_curve(df)
    table = obs.risk_table(df)
    fig_limit_curve(curve, table)

    results = pd.read_csv(par.RESULTS)
    write_results(stats, curve, table, results)

    print(stats.to_string(index=False, float_format=lambda v: "%.4f" % v))
    print()
    print(curve.to_string(index=False, float_format=lambda v: "%.4f" % v))
    print("\n-> %s\n-> %s" % (FIG_DIR, RESULTS_MD))


if __name__ == "__main__":
    main()
