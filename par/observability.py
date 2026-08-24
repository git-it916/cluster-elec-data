"""C'-layer: observability risk, and the margin it costs.

The original plan was to PREDICT meter faults. probe.py killed that (SPEC 3):
bad_measurement_data is 24 isolated single-sample dropouts with no degradation
to detect, and bad_production_power_data is not a hardware fault at all -- it
marks intervals where the CPU-interpolated production estimate is invalid.

So this layer characterises rather than predicts, which turns out to be the
more useful question anyway:

    You cannot cap what you cannot measure.

Capping headroom is measured_power_util - production_power_util. When the
production estimate is untrustworthy you do not know how much load is
sheddable, so you cannot count on capping to save you and must hold the
margin as spare capacity instead. That is Basel's k multiplier: the capital
add-on you pay for not trusting your own risk number.

    python par/observability.py     # -> data/risk_table.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import par  # noqa: E402

HERE = Path(__file__).parent
PARQUET = HERE / "data" / "power.parquet"
RISK_TABLE = HERE / "data" / "risk_table.csv"

MODEL = "fhs_ar"     # best conditional coverage of the three (SPEC 5.1)
ALPHA = 0.99
FLAG = "bad_production_power_data"


def observability(df: pd.DataFrame) -> pd.DataFrame:
    """Per-PDU trust score: the share of intervals whose production estimate
    the trace itself vouches for."""
    g = df.groupby(["cell", "pdu"])[FLAG]
    return pd.DataFrame({"obs_score": 1.0 - g.mean(), "n": g.size()}).reset_index()


def cappable_share(df: pd.DataFrame) -> pd.DataFrame:
    """Mean sheddable fraction of load, measured ONLY on intervals the trace
    marks trustworthy. PDUs with no trustworthy interval get NaN -- there is
    no honest number to report for them, and filling one in would be the exact
    mistake this layer exists to prevent."""
    ok = df[~df[FLAG]].copy()
    ok["cappable"] = (
        (ok["measured_power_util"] - ok["production_power_util"])
        / ok["measured_power_util"]
    ).clip(lower=0.0)
    s = ok.groupby(["cell", "pdu"])["cappable"].agg(["mean", "std", "size"])
    return s.rename(columns={"mean": "cappable_share",
                             "std": "cappable_sd",
                             "size": "n_trusted"}).reset_index()


def attribution_error(df: pd.DataFrame, obs: pd.DataFrame) -> float:
    """How wrong can the cappable share be on an untrusted PDU?

    PDUs the trace never flags are the reference group: their cappable share
    is believable. Comparing the spread of cappable share across trusted vs
    partially-trusted PDUs bounds the error an untrusted estimate can carry,
    and that bound is what the margin has to cover.
    """
    cs = cappable_share(df).merge(obs, on=["cell", "pdu"])
    clean = cs[cs.obs_score >= 0.999]["cappable_share"]
    dirty = cs[cs.obs_score < 0.5]["cappable_share"]
    print("  reference PDUs (never flagged): n=%d  cappable share %.3f +- %.3f"
          % (len(clean), clean.mean(), clean.std(ddof=0) if len(clean) > 1 else 0.0))
    print("  mostly-untrusted PDUs (<50%%):   n=%d  cappable share %.3f +- %.3f"
          % (len(dirty), dirty.mean(), dirty.std(ddof=0) if len(dirty) > 1 else 0.0))
    # conservative bound: the worst spread seen in either group
    lam = float(np.nanmax([clean.std(ddof=0) if len(clean) > 1 else 0.0,
                           dirty.std(ddof=0) if len(dirty) > 1 else 0.0]))
    print("  -> attribution error bound lambda = %.3f" % lam)
    return lam


def risk_table(df: pd.DataFrame) -> pd.DataFrame:
    """PaR / CVaR / trust / adjusted oversubscription limit, one row per PDU."""
    fc = par.forecast_panel(df, MODEL, ALPHA).dropna(subset=["var", "es"])
    risk = fc.groupby(["cell", "pdu"]).agg(
        par_99=("var", "mean"),
        cvar_99=("es", "mean"),
        peak_util=("y", "max"),
        mean_util=("y", "mean"),
    ).reset_index()

    obs = observability(df)
    lam = attribution_error(df, obs)
    t = risk.merge(obs, on=["cell", "pdu"]).merge(
        cappable_share(df)[["cell", "pdu", "cappable_share", "n_trusted"]],
        on=["cell", "pdu"], how="left")

    # Capping credit: extra load you may carry because it can be shed on
    # demand -- discounted by how much you trust the split, and by the
    # attribution error bound. Untrusted PDUs earn almost no credit.
    t["capping_credit"] = (t["cappable_share"].fillna(0.0) * t["obs_score"]
                           - lam * (1.0 - t["obs_score"])).clip(lower=0.0)

    # Headroom on measured power alone, then with capping credited.
    t["limit_raw"] = 1.0 / t["par_99"] - 1.0
    t["limit_adj"] = (1.0 + t["capping_credit"]) / t["par_99"] - 1.0
    t["margin_cost"] = t["limit_adj"] - t["limit_raw"]

    # A lightly loaded PDU ranks high on headroom even with zero instrument
    # trust -- "safe but blind" is a different posture from "safe and
    # observable", and conflating them is how a fleet gets surprised. Label it.
    tight = t["limit_adj"].quantile(0.25)
    t["posture"] = np.select(
        [t["cappable_share"].isna(), t["limit_adj"] < tight, t["obs_score"] < 0.5],
        ["blind", "tight", "degraded"], default="ok")
    return t.sort_values("limit_adj").reset_index(drop=True)


def main() -> None:
    df = pd.read_parquet(PARQUET)
    df[FLAG] = df[FLAG].astype(bool)

    print("== attribution error ==")
    t = risk_table(df)
    t.to_csv(RISK_TABLE, index=False)

    print("\n== observability across the fleet ==")
    print("  fully trusted (obs=1.00): %d PDUs" % (t.obs_score >= 0.999).sum())
    print("  degraded      (obs<0.50): %d PDUs" % (t.obs_score < 0.5).sum())
    print("  no trustworthy interval : %d PDUs" % t.cappable_share.isna().sum())

    print("\n== tightest 8 PDUs (least room to oversubscribe) ==")
    cols = ["cell", "pdu", "par_99", "cvar_99", "obs_score",
            "cappable_share", "limit_raw", "limit_adj", "posture"]
    print(t[cols].head(8).to_string(index=False,
                                    float_format=lambda v: "%7.3f" % v))
    print("\n  posture counts: %s"
          % t.posture.value_counts().to_dict())
    blind = t[t.posture == "blind"]
    if len(blind):
        print("  BLIND (headroom looks fine, instrumentation does not): %s"
              % ", ".join("%s/%s limit %+.1f%%" % (r.cell, r.pdu, 100 * r.limit_adj)
                          for r in blind.itertuples()))

    print("\n== what observability costs ==")
    hi = t[t.obs_score >= 0.999]
    lo = t[t.obs_score < 0.5]
    print("  trusted PDUs   (n=%2d): mean adjusted limit %+.1f%%" % (len(hi), 100 * hi.limit_adj.mean()))
    print("  untrusted PDUs (n=%2d): mean adjusted limit %+.1f%%" % (len(lo), 100 * lo.limit_adj.mean()))
    print("  fleet mean capping credit forgone: %.1f%% of capacity"
          % (100 * (t.cappable_share.fillna(0) - t.capping_credit).mean()))
    print("\n-> %s" % RISK_TABLE)


if __name__ == "__main__":
    main()
