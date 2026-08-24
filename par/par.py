"""A-layer: Power-at-Risk forecasting.

Predicts the conditional upper quantile of PDU power utilisation one step
(5 minutes) ahead, plus the conditional tail mean above it -- PaR and CVaR.

Two forecasters, in the complexity order set out in SPEC.md 5.1:

  empirical_quantile  historical simulation on the LEVEL. The naive claim
                      "load rarely exceeds its own recent 99th percentile".
  fhs_quantile        filtered historical simulation on the INCREMENT, scaled
                      by rolling volatility. Captures the volatility
                      clustering that a flat empirical quantile cannot.

Every forecast for row t uses only rows < t. That is enforced by shifting
before every rolling window, and verified by test_par.py.

    python par/par.py              # forecast + backtest the whole panel
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).parent))
import backtest as bt  # noqa: E402

PARQUET = Path(__file__).parent / "data" / "power.parquet"
RESULTS = Path(__file__).parent / "data" / "backtest_results.csv"

DAY = 288          # 5-minute intervals in a day
WINDOW = 7 * DAY   # 7-day estimation window (SPEC 5.1)
VOL_WINDOW = DAY   # 1 day of increments for the volatility filter
TARGET = "measured_power_util"


# --------------------------------------------------------------- tail helper

def rolling_tail(x: np.ndarray, alpha: float, window: int):
    """Rolling (quantile, mean-above-quantile) over a trailing `window`.

    Result at index t is computed from x[t-window:t] -- the caller is
    responsible for having already shifted x, so this function is only ever
    handed history.

    Uses np.partition rather than np.quantile: O(window) instead of
    O(window log window), and it hands back the tail values needed for the
    conditional mean in the same pass. The quantile is the order statistic at
    ceil(alpha*window)-1, which is the historical-simulation convention, not
    numpy's interpolated default.
    """
    n = x.size
    q = np.full(n, np.nan)
    es = np.full(n, np.nan)
    if n < window:
        return q, es

    # Window j spans x[j : j+window] and lands at slot j+window-1, so slot t
    # sees x[t-window+1 : t+1]. x is already shifted by the caller, so that is
    # y[t-window : t] -- all history through y[t-1] and nothing more. Dropping
    # the last window instead would silently throw away y[t-1].
    win = sliding_window_view(x, window)
    k = int(np.ceil(alpha * window)) - 1
    part = np.partition(win, k, axis=1)
    q[window - 1:] = part[:, k]
    tail = part[:, k + 1:]
    es[window - 1:] = tail.mean(axis=1) if tail.shape[1] else part[:, k]
    return q, es


# ------------------------------------------------------------------- models

def empirical_quantile(y, alpha, window=WINDOW):
    """Historical simulation on the level."""
    hist = y.shift(1).to_numpy(dtype=float)     # only the past
    return rolling_tail(hist, alpha, window)


def fhs_quantile(y, alpha, window=WINDOW, vol_window=VOL_WINDOW):
    """Filtered historical simulation on the increment.

        y_t    = y_{t-1} + sigma_t * z,  z ~ empirical over the past `window`
        PaR_t  = y_{t-1} + sigma_t * Q_alpha(z)
        CVaR_t = y_{t-1} + sigma_t * E[z | z > Q_alpha(z)]

    sigma_t is the rolling std of increments up to t-1, so a burst of
    volatility widens the interval immediately instead of a week later.
    """
    d = y.diff()
    sigma = d.shift(1).rolling(vol_window).std()
    z = (d / sigma).replace([np.inf, -np.inf], np.nan)

    zq, zes = rolling_tail(z.shift(1).to_numpy(dtype=float), alpha, window)
    base = y.shift(1).to_numpy(dtype=float)
    s = sigma.to_numpy(dtype=float)
    return base + s * zq, base + s * zes


def fhs_ar_quantile(y, alpha, window=WINDOW, vol_window=VOL_WINDOW):
    """FHS with an AR(1) term on the increment.

    Plain FHS draws z from the pool of past standardised increments, which
    treats them as exchangeable. They are not: measured increments carry a
    lag-1 autocorrelation of ~0.46 in this panel, so a step that just moved up
    is likely to keep moving up. Ignoring that is exactly why plain FHS gets
    the exception RATE right but fails the independence test -- its breaches
    arrive 22x more often right after another breach.

        d_t    = phi_t * d_{t-1} + sigma_t * z
        PaR_t  = y_{t-1} + phi_t * d_{t-1} + sigma_t * Q_alpha(z)

    phi_t is a trailing least-squares slope through the origin over the past
    `window` increment pairs, so it adapts per PDU and over time rather than
    being fitted once on the whole sample.
    """
    d = y.diff()
    d1 = d.shift(1)

    # sum(d_s * d_{s-1}) / sum(d_{s-1}^2) over s in [t-window, t-1]
    num = (d * d1).shift(1).rolling(window).sum()
    den = (d1 * d1).shift(1).rolling(window).sum()
    phi = (num / den).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-0.99, 0.99)

    resid = d - phi * d1                      # AR residual, known only at s
    sigma = resid.shift(1).rolling(vol_window).std()
    z = (resid / sigma).replace([np.inf, -np.inf], np.nan)

    zq, zes = rolling_tail(z.shift(1).to_numpy(dtype=float), alpha, window)
    center = (y.shift(1) + phi * d1).to_numpy(dtype=float)
    s = sigma.to_numpy(dtype=float)
    return center + s * zq, center + s * zes


MODELS = {
    "empirical": empirical_quantile,
    "fhs": fhs_quantile,
    "fhs_ar": fhs_ar_quantile,
}


def fhs_horizon(y, alpha, h, window=WINDOW, vol_window=VOL_WINDOW):
    """FHS at an h-step lead time (SPEC 5.1's operational curve).

    Same construction as fhs_quantile, but on the h-step increment
    y_t - y_{t-h}, so the forecast for y[t] uses nothing after y[t-h]. h=1
    reduces exactly to fhs_quantile.

    The AR(1) refinement is deliberately not carried over: increment momentum
    is a one-step effect (lag-1 autocorrelation ~0.46, lag-2 ~0.04), so it
    buys nothing here.

    Evaluate the output on every h-th row only -- consecutive rows share
    overlapping forecast windows, which would manufacture exactly the
    autocorrelation the independence test is looking for.
    """
    d = y.diff(h)
    sigma = d.shift(h).rolling(vol_window).std()
    z = (d / sigma).replace([np.inf, -np.inf], np.nan)

    zq, zes = rolling_tail(z.shift(h).to_numpy(dtype=float), alpha, window)
    base = y.shift(h).to_numpy(dtype=float)
    s = sigma.to_numpy(dtype=float)
    return base + s * zq, base + s * zes


def forecast_horizon(df, alpha, h, m=1.0):
    """fhs_horizon over the panel, thinned to non-overlapping rows.

    `m` scales the forecast SPREAD above the anchor y[t-h], not the level, so
    a multiplier of 1.0 is a no-op and calibration only ever widens or narrows
    the risk band. See calibrate_horizon.
    """
    out = []
    for (cell, pdu), g in df.groupby(["cell", "pdu"], sort=True):
        g = g.sort_values("ts")
        var, es = fhs_horizon(g[TARGET], alpha, h)
        anchor = g[TARGET].shift(h).to_numpy(dtype=float)
        f = pd.DataFrame({
            "cell": cell, "pdu": pdu, "ts": g["ts"].to_numpy(),
            "y": g[TARGET].to_numpy(), "anchor": anchor,
            "var": anchor + m * (var - anchor),
            "es": anchor + m * (es - anchor),
        })
        out.append(f.iloc[::h] if h > 1 else f)   # non-overlapping
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------- horizon calibration

CALIB_SPLIT = pd.Timestamp("2019-05-19", tz="US/Pacific")   # midpoint of days 8-31

# A horizon is only worth quoting if the held-out half carries enough breaches
# to have actually tested it. Thresholds are breaches per PDU and relative
# coverage error, fixed before looking at the results.
VERDICT_RULES = ((5.0, 0.15, "validated"), (2.0, 0.30, "marginal"))


def calibrate_panel(d, alpha, h=None, split=CALIB_SPLIT):
    """Fit ONE fleet-wide multiplier for an uncalibrated (m=1) panel.

    Per-PDU recalibration is hopeless at long lead times: at h=72 a PDU gets
    ~124 forecasts and ~1 breach, so there is nothing to calibrate against.
    Pooling all 57 PDUs turns that into thousands of forecasts, which supports
    a single shared number -- and only a single shared number.

    Fitted on the first half of the evaluable period and applied unchanged to
    the second, so the reported coverage is honest rather than in-sample. The
    verdict says whether the held-out half had enough breaches to have tested
    the horizon at all; at long lead times it usually did not, and quoting a
    calibrated number there would be false precision.
    """
    p = 1.0 - alpha
    d = d.dropna(subset=["var", "anchor"])
    tr, te = d[d.ts < split], d[d.ts >= split]

    def rate(x, mult):
        if not len(x):
            return float("nan")
        return float((x["y"] > x["anchor"] + mult * (x["var"] - x["anchor"])).mean())

    lo, hi = 0.5, 4.0
    for _ in range(40):                       # bisect: more breaches -> widen
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if rate(tr, mid) > p else (lo, mid)
    m = 0.5 * (lo + hi)

    cal = rate(te, m)
    per_pdu = rate(te, 1.0) * len(te) / max(d.pdu.nunique(), 1)
    err = abs(cal - p) / p
    verdict = "underpowered"
    for min_breaches, max_err, label in VERDICT_RULES:
        if per_pdu >= min_breaches and err <= max_err:
            verdict = label
            break
    return dict(h=h, alpha=alpha, m=m, n_train=len(tr), n_test=len(te),
                raw_test=rate(te, 1.0), cal_test=cal,
                breaches_per_pdu=per_pdu, rel_error=err, verdict=verdict)


def apply_multiplier(d, m):
    """Rescale a panel's risk band around its anchor. Immutable: new frame."""
    out = d.copy()
    for col in ("var", "es"):
        out[col] = d["anchor"] + m * (d[col] - d["anchor"])
    return out


def calibrate_horizon(df, alpha, h, split=CALIB_SPLIT):
    """Convenience wrapper: forecast the panel, then calibrate it."""
    return calibrate_panel(forecast_horizon(df, alpha, h), alpha, h, split)


# ----------------------------------------------------------------- features

def make_features(df):
    """Causal feature frame for one PDU (used by the GBM and for diagnostics).

    Every column derived from the target is shifted by at least one step, so
    row t is a legal input to a forecast of y[t]. test_par.py corrupts y[k]
    and asserts row k does not move.
    """
    y = df[TARGET]
    past = y.shift(1)
    tod = df["ts"].dt.hour * 60 + df["ts"].dt.minute

    f = pd.DataFrame(index=df.index)
    f["y"] = y
    for lag in (1, 2, 3, 6, 12):
        f["lag_%d" % lag] = y.shift(lag)
    f["lag_day"] = y.shift(DAY)
    f["lag_week"] = y.shift(7 * DAY)
    f["roll_mean_1h"] = past.rolling(12).mean()
    f["roll_std_1h"] = past.rolling(12).std()
    f["roll_mean_1d"] = past.rolling(DAY).mean()
    f["roll_max_1d"] = past.rolling(DAY).max()
    f["roll_std_1d"] = past.rolling(DAY).std()
    f["vol_ratio"] = f["roll_std_1h"] / f["roll_std_1d"]
    f["prod_lag_1"] = df["production_power_util"].shift(1)
    f["headroom_lag_1"] = (y - df["production_power_util"]).shift(1)
    # calendar features are known in advance -- no shift needed
    f["tod_sin"] = np.sin(2 * np.pi * tod / 1440)
    f["tod_cos"] = np.cos(2 * np.pi * tod / 1440)
    f["dow"] = df["ts"].dt.dayofweek
    return f


# ------------------------------------------------------------ panel driver

def forecast_panel(df, model, alpha):
    """Run one model over every PDU and return aligned forecasts."""
    fn = MODELS[model]
    out = []
    for (cell, pdu), g in df.groupby(["cell", "pdu"], sort=True):
        g = g.sort_values("ts")
        var, es = fn(g[TARGET], alpha)
        out.append(pd.DataFrame({
            "cell": cell, "pdu": pdu, "ts": g["ts"].to_numpy(),
            "y": g[TARGET].to_numpy(), "var": var, "es": es,
        }))
    return pd.concat(out, ignore_index=True)


def backtest_panel(fc, alpha, model):
    """Per-PDU backtests plus one pooled row."""
    rows = []
    for (cell, pdu), g in fc.groupby(["cell", "pdu"], sort=True):
        g = g.dropna(subset=["var", "es"])
        if g.empty:
            continue
        rows.append(dict(model=model, alpha=alpha, cell=cell, pdu=pdu,
                         **bt.run_all(g["y"].to_numpy(), g["var"].to_numpy(),
                                      g["es"].to_numpy(), alpha)))
    pooled = fc.dropna(subset=["var", "es"])
    rows.append(dict(model=model, alpha=alpha, cell="ALL", pdu="POOLED",
                     **bt.run_all(pooled["y"].to_numpy(), pooled["var"].to_numpy(),
                                  pooled["es"].to_numpy(), alpha)))
    return pd.DataFrame(rows)


def main():
    df = pd.read_parquet(PARQUET)
    print("%s rows | %d PDUs\n" % (format(len(df), ","),
                                   df.groupby(["cell", "pdu"]).ngroups))

    results = []
    for alpha in (0.95, 0.99):
        for model in MODELS:
            fc = forecast_panel(df, model, alpha)
            res = backtest_panel(fc, alpha, model)
            results.append(res)

            p = res[res.pdu == "POOLED"].iloc[0]
            per_pdu = res[res.pdu != "POOLED"]
            pad = " " * 22
            print("[%9s a=%.2f]  pooled: %s/%s exceptions (rate %.4f, expected %.4f)"
                  % (model, alpha, format(p.n_exceed, ","), format(p.n, ","),
                     p["rate"], 1 - alpha))
            print("%skupiec p=%.3g  ind p=%.3g  cc p=%.3g  ES Z2=%+.3f"
                  % (pad, p.kupiec_p, p.ind_p, p.cc_p, p.es_z2))
            zones = per_pdu.basel_zone.value_counts()
            print("%sbasel: %s" % (pad, "  ".join(
                "%s %d" % (z, zones.get(z, 0)) for z in ("green", "yellow", "red"))))
            print("%sPDUs passing conditional coverage: %d/%d\n"
                  % (pad, int((per_pdu.cc_p > 0.05).sum()), len(per_pdu)))

    pd.concat(results, ignore_index=True).to_csv(RESULTS, index=False)
    print("-> %s" % RESULTS)


if __name__ == "__main__":
    main()
