"""Checks for the fleet layer.

Same principle as test_par.py: test only what is easy to get silently wrong.
Here that is one thing above all -- every cross-unit feature is computed at
time t from a panel that also contains t+1..T, so a single missing shift turns
"pooling the fleet helps" into a leakage artefact that still looks excellent.

    python par/test_fleet.py        # or under pytest
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import fleet  # noqa: E402

RNG = np.random.default_rng(7)


def _toy(n: int = 900, k: int = 6) -> dict[str, pd.DataFrame]:
    """Small synthetic panel with the same shape contract as the real one."""
    idx = pd.date_range("2019-05-01", periods=n, freq="5min", tz="US/Pacific")
    units = [f"{'ab'[i % 2]}/pdu{i}" for i in range(k)]
    common = np.cumsum(RNG.normal(0, 0.002, n))[:, None]
    u = pd.DataFrame(0.6 + common + RNG.normal(0, 0.005, (n, k)), idx, units)
    p = u - 0.1
    z = pd.DataFrame(np.zeros((n, k)), idx, units)
    return {fleet.TARGET: u, fleet.SECOND: p,
            "bad_measurement_data": z, "bad_production_power_data": z}


# ------------------------------------------------------------ causality

def test_features_ignore_the_future():
    """Corrupt the panel from step T onward. Every feature row before T must
    be bit-identical. This is the test the whole fleet claim rests on."""
    panels = _toy()
    T = 500
    clean = fleet.build_features(panels)

    dirty = {k: v.copy() for k, v in panels.items()}
    dirty[fleet.TARGET].iloc[T:] += 0.25          # a huge, obvious future shock
    dirty[fleet.SECOND].iloc[T:] += 0.25
    after = fleet.build_features(dirty)

    a = clean[clean.step < T].drop(columns="step")
    b = after[after.step < T].drop(columns="step")
    assert a.shape == b.shape and list(a.columns) == list(b.columns)
    bad = [c for c in a.columns if not np.allclose(a[c], b[c], equal_nan=True)]
    assert not bad, f"future leaked into: {bad}"


def test_fleet_features_do_see_the_present_across_units():
    """The mirror image: a shock to OTHER units at time t must reach this
    unit's fleet features at t. Otherwise the test above passes trivially."""
    panels = _toy()
    T = 500
    clean = fleet.build_features(panels)
    dirty = {k: v.copy() for k, v in panels.items()}
    dirty[fleet.TARGET].iloc[T:, 1:] += 0.25
    after = fleet.build_features(dirty)

    unit0 = panels[fleet.TARGET].columns[0]
    row_c = clean.xs(unit0, level="unit")
    row_a = after.xs(unit0, level="unit")
    at_T = row_c.step == T
    assert not np.allclose(row_c.loc[at_T, "fleet_mean"], row_a.loc[at_T, "fleet_mean"])
    assert np.allclose(row_c.loc[at_T, "u"], row_a.loc[at_T, "u"])


def test_train_test_split_is_ordered_and_disjoint():
    panels = _toy(n=fleet.SPLIT + 400)
    long = fleet.build_features(panels)
    tr, te = long[long.step < fleet.SPLIT], long[long.step >= fleet.SPLIT]
    assert len(tr) and len(te)
    assert tr.step.max() < te.step.min()
    assert set(tr.index) & set(te.index) == set()


# ------------------------------------------------------------- coupling

def test_n_eff_hits_both_ends():
    """Independent columns -> N_eff == N. One shared factor -> N_eff == 1."""
    n, k = 4000, 12
    idx = pd.date_range("2019-05-01", periods=n, freq="5min", tz="US/Pacific")
    cols = [f"a/pdu{i}" for i in range(k)]

    indep = pd.DataFrame(np.cumsum(RNG.normal(0, 1, (n, k)), axis=0), idx, cols)
    assert fleet.coupling(indep)["increment_n_eff"] > 0.9 * k

    shared = pd.DataFrame(np.tile(np.cumsum(RNG.normal(0, 1, n))[:, None], k), idx, cols)
    assert fleet.coupling(shared)["increment_n_eff"] < 1.05


def test_reserve_inflation_is_one_when_independent():
    """With independent units the aggregate ramp sd must match the 1/sqrt(N)
    prediction, i.e. the inflation factor is ~1. If this drifts, the headline
    '2.6x' number is measuring an artefact of the estimator, not the fleet."""
    n, k = 6000, 20
    idx = pd.date_range("2019-05-01", periods=n, freq="5min", tz="US/Pacific")
    x = pd.DataFrame(np.cumsum(RNG.normal(0, 1, (n, k)), axis=0), idx,
                     [f"a/pdu{i}" for i in range(k)])
    assert abs(fleet.coupling(x)["reserve_inflation_5min"] - 1.0) < 0.1


# ------------------------------------------------------------ injection

def test_injected_bias_is_a_pure_residual_offset():
    """inject_faults() skips the refit by asserting the peer reconstruction
    does not depend on the faulted unit. Verify that against an actual refit."""
    from sklearn.linear_model import Ridge

    u = _toy(n=600, k=8)[fleet.TARGET]
    unit, peers = u.columns[0], list(u.columns[1:])
    m = Ridge(alpha=1.0).fit(u[peers].iloc[:300], u[unit].iloc[:300])
    resid = u[unit] - m.predict(u[peers])

    faulted = u.copy()
    faulted[unit] += 0.02
    m2 = Ridge(alpha=1.0).fit(faulted[peers].iloc[:300], faulted[unit].iloc[:300])
    # peers are untouched, so the only change in the residual is the offset
    # minus whatever the intercept absorbs -- the *shape* must be identical
    resid2 = faulted[unit] - m2.predict(faulted[peers])
    assert np.allclose(resid - resid.mean(), resid2 - resid2.mean(), atol=1e-9)


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
