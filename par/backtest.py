"""VaR / ES backtests, ported from the market-risk literature.

Pure statistics: numpy + scipy only, no pandas, no I/O.  Every function takes
an `exceed` array -- the 0/1 indicator of whether the realised value breached
the predicted quantile -- and returns a plain dict.

References
  Kupiec (1995)            unconditional coverage
  Christoffersen (1998)    independence + conditional coverage
  BCBS (1996/2006)         traffic-light zones
  Acerbi & Szekely (2014)  ES backtest, "Test 2"
"""
from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = [
    "kupiec_pof",
    "christoffersen_independence",
    "christoffersen_cc",
    "basel_traffic_light",
    "acerbi_szekely_z2",
    "run_all",
]


def _xlogx(x: np.ndarray | float) -> np.ndarray | float:
    """x*log(x) with the 0*log(0) = 0 convention, so degenerate transition
    counts collapse to zero instead of producing nan."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, x * np.log(np.where(x > 0, x, 1.0)), 0.0)


def kupiec_pof(exceed: np.ndarray, alpha: float) -> dict:
    """Unconditional coverage: is the exception RATE right?

    H0: P(exception) == 1 - alpha.  LR_uc ~ chi2(1).
    """
    e = np.asarray(exceed, dtype=bool)
    n, x = e.size, int(e.sum())
    p = 1.0 - alpha
    if n == 0:
        return dict(n=0, x=0, expected=0.0, rate=np.nan, lr=np.nan, pvalue=np.nan)

    # log-likelihood under H0 minus log-likelihood at the MLE rate x/n
    ll_h0 = (n - x) * np.log1p(-p) + x * np.log(p)
    ll_mle = _xlogx(n - x) + _xlogx(x) - _xlogx(n)  # = (n-x)ln((n-x)/n) + x ln(x/n)
    lr = float(-2.0 * (ll_h0 - ll_mle))
    return dict(n=n, x=x, expected=n * p, rate=x / n, lr=lr,
                pvalue=float(stats.chi2.sf(lr, 1)))


def christoffersen_independence(exceed: np.ndarray) -> dict:
    """Independence: do exceptions CLUSTER?

    H0: P(exception | exception yesterday) == P(exception | none yesterday).
    LR_ind ~ chi2(1).  This is the test that matters most for power: breaches
    arrive in bursts (heat waves, batch surges), and a model can pass Kupiec
    while failing catastrophically here.
    """
    e = np.asarray(exceed, dtype=int)
    if e.size < 2:
        return dict(n00=0, n01=0, n10=0, n11=0, lr=np.nan, pvalue=np.nan)

    prev, cur = e[:-1], e[1:]
    n00 = int(np.sum((prev == 0) & (cur == 0)))
    n01 = int(np.sum((prev == 0) & (cur == 1)))
    n10 = int(np.sum((prev == 1) & (cur == 0)))
    n11 = int(np.sum((prev == 1) & (cur == 1)))

    # Restricted (single rate) vs unrestricted (row-specific rates) Markov chain.
    ll_r = _xlogx(n00 + n10) + _xlogx(n01 + n11) - _xlogx(n00 + n01 + n10 + n11)
    ll_u = (_xlogx(n00) + _xlogx(n01) - _xlogx(n00 + n01)
            + _xlogx(n10) + _xlogx(n11) - _xlogx(n10 + n11))
    lr = float(-2.0 * (ll_r - ll_u))
    lr = max(lr, 0.0)  # guard against -0.0 from float cancellation
    return dict(n00=n00, n01=n01, n10=n10, n11=n11, lr=lr,
                pvalue=float(stats.chi2.sf(lr, 1)))


def christoffersen_cc(exceed: np.ndarray, alpha: float) -> dict:
    """Conditional coverage = correct rate AND no clustering. LR_cc ~ chi2(2)."""
    uc = kupiec_pof(exceed, alpha)
    ind = christoffersen_independence(exceed)
    lr = uc["lr"] + ind["lr"]
    return dict(lr=float(lr), pvalue=float(stats.chi2.sf(lr, 2)),
                lr_uc=uc["lr"], lr_ind=ind["lr"])


def basel_traffic_light(exceed: np.ndarray, alpha: float, window: int = 250) -> dict:
    """BCBS zones on the most recent `window` observations.

    Zones are defined by the cumulative binomial probability of observing at
    most x exceptions, not by hardcoded counts -- so this generalises to any
    alpha and window while reproducing the published 250-obs/99% table
    (0-4 green, 5-9 yellow, 10+ red).
    """
    e = np.asarray(exceed, dtype=bool)[-window:]
    n, x, p = e.size, int(e.sum()), 1.0 - alpha
    cum = float(stats.binom.cdf(x, n, p))
    zone = "green" if cum < 0.95 else ("yellow" if cum < 0.9999 else "red")
    return dict(n=n, x=x, cum_prob=cum, zone=zone)


def acerbi_szekely_z2(loss: np.ndarray, var: np.ndarray, es: np.ndarray,
                      alpha: float, n_boot: int = 2000,
                      seed: int = 0) -> dict:
    """Acerbi-Szekely Test 2 for Expected Shortfall.

    Z2 = mean_t( loss_t * 1{loss_t > VaR_t} / ES_t ) / p - 1,  E[Z2] = 0 under H0.
    Z2 > 0 means ES is UNDERSTATED -- the dangerous direction, because ES is
    what sets how much load you must actually shed.

    ponytail: p-value comes from bootstrapping the observed exceedance losses,
    not from Monte Carlo under each model's predictive distribution. Adequate
    for ranking models; swap in full MC if a regulator-grade number is needed.
    """
    loss, var, es = (np.asarray(a, dtype=float) for a in (loss, var, es))
    p = 1.0 - alpha
    ex = loss > var
    n = loss.size
    if n == 0 or not ex.any():
        return dict(z2=np.nan, pvalue=np.nan, n_exceed=int(ex.sum()))

    contrib = np.where(ex, loss / es, 0.0)
    z2 = float(contrib.sum() / (n * p) - 1.0)

    # H0 reference: resample which observations breach, keeping the model's own
    # predicted breach probability p.
    rng = np.random.default_rng(seed)
    pool = (loss / es)[ex]
    draws = rng.binomial(n, p, size=n_boot)
    null = np.array([
        (rng.choice(pool, size=k, replace=True).sum() / (n * p) - 1.0) if k else -1.0
        for k in draws
    ])
    pvalue = float(np.mean(null >= z2))
    return dict(z2=z2, pvalue=pvalue, n_exceed=int(ex.sum()))


def run_all(loss: np.ndarray, var: np.ndarray, es: np.ndarray | None,
            alpha: float) -> dict:
    """Every test at once, flattened for a results table."""
    exceed = np.asarray(loss) > np.asarray(var)
    uc = kupiec_pof(exceed, alpha)
    ind = christoffersen_independence(exceed)
    cc = christoffersen_cc(exceed, alpha)
    tl = basel_traffic_light(exceed, alpha)
    out = dict(
        n=uc["n"], n_exceed=uc["x"], expected=uc["expected"], rate=uc["rate"],
        kupiec_lr=uc["lr"], kupiec_p=uc["pvalue"],
        ind_lr=ind["lr"], ind_p=ind["pvalue"],
        cc_lr=cc["lr"], cc_p=cc["pvalue"],
        basel_zone=tl["zone"], basel_x=tl["x"],
    )
    if es is not None:
        z2 = acerbi_szekely_z2(loss, var, es, alpha)
        out |= dict(es_z2=z2["z2"], es_p=z2["pvalue"])
    return out
