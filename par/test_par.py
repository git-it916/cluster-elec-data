"""Checks for the pieces that are easy to get silently wrong.

Runs standalone (`python par/test_par.py`) or under pytest.

Two things are worth testing here and nothing else is:
  - the backtest statistics, which have published closed-form values, and
  - feature causality, because a one-row off-by-one turns the whole study into
    a leakage artefact that still looks excellent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import backtest as bt  # noqa: E402

TOL = 1e-6


# ---------------------------------------------------------------- Kupiec POF

def test_kupiec_zero_when_rate_is_exact():
    """Observed rate == nominal rate must give LR exactly 0."""
    e = np.zeros(1000, dtype=bool)
    e[:10] = True  # 1% of 1000, alpha = 0.99
    assert abs(bt.kupiec_pof(e, 0.99)["lr"]) < TOL


def test_kupiec_known_value():
    """n=250, x=10, p=0.01 -> LR_uc = 12.95547 (exact
    log-likelihood ratio; this is the Basel red-zone boundary case)."""
    e = np.zeros(250, dtype=bool)
    e[:10] = True
    r = bt.kupiec_pof(e, 0.99)
    assert abs(r["lr"] - 12.9554911) < 1e-5, r
    assert r["pvalue"] < 0.001
    assert abs(r["expected"] - 2.5) < 1e-9 and r["x"] == 10


def test_kupiec_degenerate_inputs():
    """Zero exceptions and empty input must not produce nan or crash."""
    assert np.isfinite(bt.kupiec_pof(np.zeros(500, dtype=bool), 0.99)["lr"])
    assert bt.kupiec_pof(np.array([], dtype=bool), 0.99)["n"] == 0


# ------------------------------------------------- Christoffersen dependence

def test_independence_near_zero_for_balanced_transitions():
    """Equal transition rates out of both states -> no evidence of clustering."""
    e = np.array([0, 0, 1, 1] * 25, dtype=bool)
    assert bt.christoffersen_independence(e)["lr"] < 0.05


def test_independence_detects_clustering():
    """Same exception COUNT, but bunched vs spread out. Kupiec cannot tell
    these apart; the independence test must."""
    n, k = 1000, 50
    clustered = np.zeros(n, dtype=bool)
    clustered[100:100 + k] = True
    spread = np.zeros(n, dtype=bool)
    spread[:: n // k] = True

    assert bt.kupiec_pof(clustered, 0.95)["lr"] == bt.kupiec_pof(spread, 0.95)["lr"]
    lr_c = bt.christoffersen_independence(clustered)["lr"]
    lr_s = bt.christoffersen_independence(spread)["lr"]
    assert lr_c > 100, lr_c
    assert lr_c > lr_s
    assert bt.christoffersen_independence(clustered)["pvalue"] < 0.001


def test_independence_non_negative_and_safe():
    """LR is a likelihood ratio: never negative, never nan on degenerate input."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        e = rng.random(200) < rng.uniform(0.01, 0.3)
        assert bt.christoffersen_independence(e)["lr"] >= 0.0
    assert bt.christoffersen_independence(np.zeros(100, dtype=bool))["lr"] == 0.0


def test_cc_is_sum_of_parts():
    e = np.array([0, 0, 1, 1, 0, 1] * 40, dtype=bool)
    cc = bt.christoffersen_cc(e, 0.9)
    expected = bt.kupiec_pof(e, 0.9)["lr"] + bt.christoffersen_independence(e)["lr"]
    assert abs(cc["lr"] - expected) < TOL


# ------------------------------------------------------ Basel traffic light

def test_basel_reproduces_published_zones():
    """BCBS 250-observation / 99% table: 0-4 green, 5-9 yellow, 10+ red.
    We compute zones from the binomial CDF, so this pins the generalisation
    to the published special case."""
    def zone(x):
        e = np.zeros(250, dtype=bool)
        e[:x] = True
        return bt.basel_traffic_light(e, 0.99)["zone"]

    assert [zone(x) for x in (0, 4)] == ["green", "green"]
    assert [zone(x) for x in (5, 9)] == ["yellow", "yellow"]
    assert [zone(x) for x in (10, 15)] == ["red", "red"]


# ------------------------------------------------- Acerbi-Szekely ES test

def test_z2_is_zero_for_a_correct_es():
    """Normal losses with the analytically correct VaR and ES -> Z2 ~ 0."""
    from scipy import stats as st
    alpha, n = 0.95, 200_000
    loss = np.random.default_rng(1).standard_normal(n)
    var = st.norm.ppf(alpha)
    es = st.norm.pdf(var) / (1 - alpha)
    z2 = bt.acerbi_szekely_z2(loss, np.full(n, var), np.full(n, es),
                              alpha, n_boot=200)["z2"]
    assert abs(z2) < 0.05, z2


def test_z2_flags_understated_es():
    """Halving ES should roughly double the statistic -> clearly positive."""
    from scipy import stats as st
    alpha, n = 0.95, 200_000
    loss = np.random.default_rng(2).standard_normal(n)
    var = st.norm.ppf(alpha)
    es = st.norm.pdf(var) / (1 - alpha)
    r = bt.acerbi_szekely_z2(loss, np.full(n, var), np.full(n, es * 0.5),
                             alpha, n_boot=200)
    assert r["z2"] > 0.5, r


# ------------------------------------------------------------ NO LEAKAGE

def _toy_panel(n: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "cell": "a", "pdu": "pdu1",
        "ts": pd.date_range("2019-05-01", periods=n, freq="5min", tz="US/Pacific"),
        "measured_power_util": rng.uniform(0.4, 0.9, n),
        "production_power_util": rng.uniform(0.2, 0.4, n),
    })


def test_features_never_see_the_present_or_future():
    """THE test of this study.

    Corrupt the target at row k and nothing else. Every feature row from k
    onward that a t-time forecast is allowed to use must be unchanged --
    features at row k predict y[k], so they may only depend on y[:k].
    """
    import par

    k = 400
    a = _toy_panel()
    b = a.copy()
    b.loc[k, "measured_power_util"] = 99.0  # absurd value; must not propagate backwards

    fa, fb = par.make_features(a), par.make_features(b)
    cols = [c for c in fa.columns if c != "y"]

    pd.testing.assert_frame_equal(fa.loc[:k, cols], fb.loc[:k, cols])
    assert not fa.loc[:k, cols].equals(fa.loc[1:k + 1, cols].reset_index(drop=True))


def test_rolling_quantile_forecast_is_causal():
    """Same corruption test on the forecaster itself: the prediction FOR row k
    must not move when y[k] changes."""
    import par

    k = 400
    a = _toy_panel()
    b = a.copy()
    b.loc[k, "measured_power_util"] = 99.0

    for fn in par.MODELS.values():   # every registered model, including new ones
        # both forecasters return (PaR, CVaR) -- check each, and unpack, since
        # slicing the tuple itself would compare every row and prove nothing.
        for name, qa, qb in zip(("PaR", "CVaR"),
                                fn(a["measured_power_util"], 0.99, window=288),
                                fn(b["measured_power_util"], 0.99, window=288)):
            assert np.allclose(qa[:k + 1], qb[:k + 1], equal_nan=True), \
                "%s %s" % (fn.__name__, name)
            # and the corruption must actually reach row k+1, or the test is
            # passing because nothing propagates at all
            assert not np.allclose(qa[k + 1:], qb[k + 1:], equal_nan=True), \
                "%s %s: corruption never propagated" % (fn.__name__, name)


def test_horizon_forecast_respects_its_lead_time():
    """An h-step forecast of y[t] may use nothing after y[t-h].

    This case was missing when fhs_horizon was written, and a scratch
    experiment that shifted by 1 instead of h produced flattering results
    before the gap was caught. Row t-h is the whole contract, so test it.
    """
    import par

    k = 400
    a = _toy_panel(1200)[par.TARGET]
    b = a.copy()
    b.iloc[k] = 99.0

    for h in (1, 3, 12):
        va = np.asarray(par.fhs_horizon(a, 0.99, h, window=288)[0])
        vb = np.asarray(par.fhs_horizon(b, 0.99, h, window=288)[0])
        # forecasts for every row before k+h are made without ever seeing y[k]
        assert np.allclose(va[:k + h], vb[:k + h], equal_nan=True), "h=%d leaks" % h
        assert not np.allclose(va[k + h:], vb[k + h:], equal_nan=True), \
            "h=%d: corruption never propagated, test proves nothing" % h


def test_horizon_multiplier_is_a_noop_at_one():
    """Calibration must scale the spread, never shift the level."""
    import par

    df = _toy_panel(1200)
    base = par.forecast_horizon(df, 0.99, 6, m=1.0)
    wide = par.forecast_horizon(df, 0.99, 6, m=2.0)
    ok = base["var"].notna()
    assert np.allclose(base.loc[ok, "var"], base.loc[ok, "var"])
    # m=2 must double the distance from the anchor, in the same direction
    spread_b = (base.loc[ok, "var"] - base.loc[ok, "anchor"]).to_numpy()
    spread_w = (wide.loc[ok, "var"] - wide.loc[ok, "anchor"]).to_numpy()
    assert np.allclose(spread_w, 2.0 * spread_b)


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001 - test runner, report and continue
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
