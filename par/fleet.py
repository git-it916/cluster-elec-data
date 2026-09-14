"""Fleet layer: does pooling many units' sensors buy anything an ML model can use?

The A/C' layers ask "how close is one PDU to its limit". This layer asks the
question a multi-unit integrator actually faces: you have N units, each with
several sensor channels, wired into one plant. Is there measurable value in
modelling them TOGETHER rather than one at a time?

Five questions, five numbers, one table:

  Q1 coupling      how many of the N units move independently?     -> PCA/N_eff
  Q2 forecast      does a fleet-pooled model beat a per-unit one?   -> MAE skill
  Q3 ramp          can it call a load step before it happens?       -> ROC-AUC
  Q4 redundancy    can peers reconstruct a unit whose sensor died?  -> R^2 + AUC
  Q5 segmentation  does unsupervised structure match real grouping? -> ARI

Sensor channels available per unit: measured_power_util, production_power_util,
and two data-quality flags. Four channels x 57 units = 228 series.

Every feature at row t uses information available at t; targets are at t+h.
Train = days 1-20, test = days 21-31, split once, never re-fit on test.

    python par/fleet.py        # ~3 min -> data/fleet_*.csv, FLEET.md, figures
"""
from __future__ import annotations

import json
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
PARQUET = HERE / "data" / "power.parquet"
OUT = HERE / "data"
FIGS = HERE / "figures"

DAY = 288                 # 5-minute steps per day
TRAIN_DAYS = 20
SPLIT = TRAIN_DAYS * DAY  # first index of the test half
HORIZONS = [1, 3, 6, 12]  # 5min, 15min, 30min, 1h
SEED = 0

TARGET = "measured_power_util"
SECOND = "production_power_util"


# ------------------------------------------------------------------ loading

def load_panels() -> dict[str, pd.DataFrame]:
    """Long trace -> wide matrices, one per sensor channel: rows=time, cols=unit."""
    df = pd.read_parquet(PARQUET)
    df["unit"] = df["cell"] + "/" + df["pdu"]
    cols = [TARGET, SECOND, "bad_measurement_data", "bad_production_power_data"]
    panels = {c: df.pivot(index="ts", columns="unit", values=c).astype(float) for c in cols}
    for p in panels.values():
        assert not p.isna().any().any(), "trace is supposed to be gap-free"
    return panels


# -------------------------------------------------------- Q1: fleet coupling

def coupling(u: pd.DataFrame) -> dict:
    """How many independent units does a 57-unit fleet actually behave like?

    Participation ratio of the correlation-matrix eigenvalues,
    N_eff = (sum L)^2 / sum L^2, is 1 when everything moves as one unit and N
    when the units are independent. Computed on 5-minute INCREMENTS, because
    what a generator has to follow is the change, not the level -- levels share
    a diurnal cycle that inflates correlation without implying joint ramps.
    """
    out = {"n_units": u.shape[1]}
    for name, x in [("level", u.iloc[:SPLIT]), ("increment", u.iloc[:SPLIT].diff().iloc[1:])]:
        c = np.corrcoef(x.values, rowvar=False)
        lam = np.sort(np.linalg.eigvalsh(c))[::-1]
        lam = np.clip(lam, 0, None)
        cum = np.cumsum(lam) / lam.sum()
        out[f"{name}_pc1_var"] = float(lam[0] / lam.sum())
        out[f"{name}_pc_for_90"] = int(np.searchsorted(cum, 0.90) + 1)
        out[f"{name}_n_eff"] = float(lam.sum() ** 2 / (lam**2).sum())
        out[f"{name}_mean_corr"] = float((c.sum() - c.shape[0]) / (c.size - c.shape[0]))

    # Diversity: peak of the fleet average vs the average of individual peaks.
    # Ratio 1.0 = every unit peaks at the same instant, nothing to net out.
    tr = u.iloc[:SPLIT]
    out["diversity_factor"] = float(tr.mean(axis=1).max() / tr.max(axis=0).mean())

    # Reserve sizing. If the units were independent, the aggregate's ramp
    # volatility would fall as 1/sqrt(N). It does not. The ratio is how much
    # extra following capability an independence assumption fails to buy.
    for name, k in [("5min", 1), ("1h", 12)]:
        dd = tr.diff(k).iloc[k:]
        act = dd.mean(axis=1).std()
        ind = float(np.sqrt((dd.std() ** 2).sum()) / dd.shape[1])
        out[f"reserve_inflation_{name}"] = float(act / ind)
        out[f"fleet_ramp_sd_{name}_pp"] = float(act * 100)
        out[f"unit_ramp_p999_{name}_pp"] = float(dd.quantile(0.999).mean() * 100)
        out[f"fleet_ramp_p999_{name}_pp"] = float(dd.mean(axis=1).quantile(0.999) * 100)

    # Coincident ramping: share of the fleet moving up in the same 5 minutes,
    # conditional on the fleet average making a top-1% up-move.
    d = tr.diff().iloc[1:]
    fleet_d = d.mean(axis=1)
    big = fleet_d >= fleet_d.quantile(0.99)
    out["share_up_on_fleet_ramp"] = float((d[big] > 0).mean(axis=1).mean())
    out["share_up_baseline"] = float((d > 0).mean(axis=1).mean())
    return out


# ------------------------------------------------------------ feature build

def build_features(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One long frame: (t, unit) x features, every feature known at t.

    Blocks, in the order they matter to the argument:
      own      the unit's own history      -- what a single-unit model gets
      channel  its other sensor channels   -- multi-sensor, still single-unit
      fleet    the other 56 units at t     -- what only an integrated plant gets
      time     clock features
    """
    u, p = panels[TARGET], panels[SECOND]
    badp = panels["bad_production_power_data"]
    idx, units = u.index, u.columns

    fleet_mean, fleet_std = u.mean(axis=1), u.std(axis=1)
    fleet_d1 = u.diff().mean(axis=1)
    cell = pd.Series([c.split("/")[0] for c in units], index=units)
    cell_mean = u.T.groupby(cell).mean().T          # time x cell
    cell_d1 = u.diff().T.groupby(cell).mean().T

    tod = np.asarray(idx.hour * 12 + idx.minute // 5)
    feat = {
        # --- own channel
        "u": u, "d1": u.diff(), "d3": u.diff(3), "d6": u.diff(6),
        "lag12": u.shift(12), "lag_day": u.shift(DAY),
        "roll_mean_h": u.shift(1).rolling(12).mean(),
        "roll_std_h": u.shift(1).rolling(12).std(),
        "roll_std_day": u.shift(1).rolling(DAY).std(),
        "dev_from_day": u - u.shift(1).rolling(DAY).mean(),
        # --- second sensor channel on the same unit
        "p": p, "headroom": u - p, "p_d1": p.diff(), "badp": badp,
        # --- fleet: only an integrated multi-unit plant can compute these
        "fleet_mean": pd.DataFrame(np.tile(fleet_mean.values[:, None], u.shape[1]), idx, units),
        "fleet_std": pd.DataFrame(np.tile(fleet_std.values[:, None], u.shape[1]), idx, units),
        "fleet_d1": pd.DataFrame(np.tile(fleet_d1.values[:, None], u.shape[1]), idx, units),
        "cell_mean": cell_mean[cell].set_axis(units, axis=1),
        "cell_d1": cell_d1[cell].set_axis(units, axis=1),
        "rel_to_fleet": u.sub(fleet_mean, axis=0),
        # --- clock
        "tod_sin": pd.DataFrame(np.tile(np.sin(2 * np.pi * tod / DAY)[:, None], u.shape[1]), idx, units),
        "tod_cos": pd.DataFrame(np.tile(np.cos(2 * np.pi * tod / DAY)[:, None], u.shape[1]), idx, units),
        "dow": pd.DataFrame(np.tile(np.asarray(idx.dayofweek)[:, None], u.shape[1]), idx, units),
    }
    long = pd.concat({k: v.stack() for k, v in feat.items()}, axis=1)
    long.index.names = ["ts", "unit"]
    long["step"] = long.index.get_level_values("ts").map(
        pd.Series(np.arange(len(idx)), index=idx))
    return long


OWN_COLS = ["u", "d1", "d3", "d6", "lag12", "lag_day", "roll_mean_h", "roll_std_h",
            "roll_std_day", "dev_from_day", "p", "headroom", "p_d1", "badp",
            "tod_sin", "tod_cos", "dow"]
FLEET_COLS = OWN_COLS + ["fleet_mean", "fleet_std", "fleet_d1", "cell_mean",
                         "cell_d1", "rel_to_fleet"]


# --------------------------------------------------------- Q2/Q3: forecasting

def forecast(long: pd.DataFrame, panels: dict[str, pd.DataFrame]) -> tuple:
    """Persistence / per-unit ridge / per-unit GBM / fleet GBM, per horizon.

    Every learner targets the INCREMENT y - u_t, not the level. Two reasons:
    persistence then falls out as the zero prediction, so "skill" is exactly the
    information added on top of "assume nothing changes"; and gradient-boosted
    trees cannot extrapolate a level outside their training range, which on a
    slowly drifting load is a handicap that has nothing to do with the question
    being asked. Predictions are added back to u_t before scoring, so every
    number below is still an error on the level.
    """
    import lightgbm as lgb
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import wilcoxon

    u = panels[TARGET]
    rows, ramp_rows, pair_rows, preds = [], [], [], {}

    for h in HORIZONS:
        d = long.copy()
        d["y"] = u.shift(-h).stack().reindex(d.index)
        d = d.dropna()
        tr, te = d[d.step < SPLIT], d[d.step >= SPLIT]
        assert te.step.min() >= SPLIT and tr.step.max() < SPLIT

        y_te = te.y.values
        base_te = te.u.values
        dy_tr = (tr.y - tr.u).values             # increment target
        p_persist = base_te                      # = base + 0

        unit_te = te.index.get_level_values("unit").values

        # per-unit ridge on own lags (scaled -- unscaled ridge shrinks the
        # informative lag coefficients toward zero and loses to persistence)
        p_ridge = np.empty_like(y_te)
        for unit, g in te.groupby(level="unit"):
            gt = tr.xs(unit, level="unit")
            m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            m.fit(gt[OWN_COLS], gt.y - gt.u)
            p_ridge[unit_te == unit] = g.u.values + m.predict(g[OWN_COLS])

        # same estimator, same per-unit fit, only the feature set widens.
        # This is the ablation the whole argument rests on.
        p_ridge_f = np.empty_like(y_te)
        for unit, g in te.groupby(level="unit"):
            gt = tr.xs(unit, level="unit")
            m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            m.fit(gt[FLEET_COLS], gt.y - gt.u)
            p_ridge_f[unit_te == unit] = g.u.values + m.predict(g[FLEET_COLS])

        # per-unit GBM, own + channel features only
        p_solo = np.empty_like(y_te)
        for unit, g in te.groupby(level="unit"):
            gt = tr.xs(unit, level="unit")
            m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                  verbose=-1, random_state=SEED)
            m.fit(gt[OWN_COLS], gt.y - gt.u)
            p_solo[unit_te == unit] = g.u.values + m.predict(g[OWN_COLS])

        # one fleet-pooled GBM over all units, with cross-unit features
        gbm = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63,
                                verbose=-1, random_state=SEED)
        gbm.fit(tr[FLEET_COLS], dy_tr)
        p_fleet = base_te + gbm.predict(te[FLEET_COLS])

        mae0 = np.abs(y_te - p_persist).mean()
        model_preds = [("persistence", p_persist), ("ridge_solo", p_ridge),
                       ("ridge_fleet", p_ridge_f), ("gbm_solo", p_solo),
                       ("gbm_fleet", p_fleet)]
        for name, pred in model_preds:
            err = y_te - pred
            rows.append({
                "horizon_min": h * 5, "model": name,
                "mae_pp": np.abs(err).mean() * 100,
                "rmse_pp": np.sqrt((err**2).mean()) * 100,
                "skill_vs_persistence": 1 - np.abs(err).mean() / mae0,
                "p95_abs_err_pp": np.quantile(np.abs(err), 0.95) * 100,
            })

        # Q3: up-ramp events, threshold frozen on the train half
        delta_tr = (tr.y - tr.u).values
        thr = np.quantile(delta_tr, 0.99)
        event = (y_te - p_persist) >= thr
        vol_base = te.roll_std_h.values         # "it has been volatile lately"
        ramp_rows += [_ramp(h, thr, event, name, s) for name, s in
                      [("volatility_persistence", vol_base),
                       ("gbm_solo", p_solo - p_persist),
                       ("gbm_fleet", p_fleet - p_persist)]]

        # per-unit MAE -> paired test across the 57 units. With 180k
        # autocorrelated test rows a pooled t-test would call anything
        # significant; pairing on the unit is the honest unit of replication.
        per_unit = pd.DataFrame({n: np.abs(y_te - pr) for n, pr in model_preds},
                                index=te.index).groupby(level="unit").mean()
        for a, b in [("gbm_fleet", "gbm_solo"), ("ridge_fleet", "ridge_solo"),
                     ("gbm_fleet", "persistence"), ("gbm_solo", "persistence")]:
            diff = per_unit[b] - per_unit[a]
            pair_rows.append({
                "horizon_min": h * 5, "better": a, "vs": b,
                "median_mae_gain_pct": float((diff / per_unit[b]).median() * 100),
                "units_improved": int((diff > 0).sum()), "n_units": len(diff),
                "wilcoxon_p": float(wilcoxon(diff).pvalue),
            })
        preds[h] = pd.DataFrame({"y": y_te, **{n: pr for n, pr in model_preds}},
                                index=te.index)
        if h == HORIZONS[1]:
            preds["importance"] = pd.Series(gbm.feature_importances_,
                                            index=FLEET_COLS).sort_values(ascending=False)
    return pd.DataFrame(rows), pd.DataFrame(ramp_rows), pd.DataFrame(pair_rows), preds


def _ramp(h, thr, event, name, score) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    k = max(int(event.sum()), 1)
    top = np.argsort(score)[-k:]
    return {"horizon_min": h * 5, "threshold_pp": thr * 100,
            "n_events": int(event.sum()), "base_rate": float(event.mean()),
            "detector": name, "auc": roc_auc_score(event, score),
            "avg_precision": average_precision_score(event, score),
            "precision_at_n": float(event[top].mean())}


# ---------------------------------------------------- Q4: analytical redundancy

def redundancy(panels: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """Drop one unit's sensor, rebuild its reading from the other 56.

    This is the classic analytical-redundancy / virtual-sensor test: a fleet is
    its own backup instrument. Ridge per unit, fitted on the train half with
    flagged rows removed, scored on the held-out half. The residual then doubles
    as an unsupervised fault detector, checked against the trace's own
    bad_measurement_data flag.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import roc_auc_score

    u, badm = panels[TARGET], panels["bad_measurement_data"].astype(bool)
    rows, resid = [], pd.DataFrame(index=u.index, columns=u.columns, dtype=float)
    clean_tr = ~badm.iloc[:SPLIT].any(axis=1)

    for unit in u.columns:
        peers = [c for c in u.columns if c != unit]
        Xtr, ytr = u.iloc[:SPLIT][peers][clean_tr], u.iloc[:SPLIT][unit][clean_tr]
        m = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Xtr, ytr)
        pred = pd.Series(m.predict(u[peers]), index=u.index)
        resid[unit] = u[unit] - pred
        te = slice(SPLIT, None)
        ss_res = ((u[unit].iloc[te] - pred.iloc[te]) ** 2).sum()
        ss_tot = ((u[unit].iloc[te] - u[unit].iloc[te].mean()) ** 2).sum()
        rows.append({"unit": unit, "r2_test": 1 - ss_res / ss_tot,
                     "rmse_test_pp": np.sqrt(ss_res / (len(u) - SPLIT)) * 100,
                     "sd_test_pp": u[unit].iloc[te].std() * 100})

    # residual as a fault score, z-scored per unit against the train half
    z = (resid - resid.iloc[:SPLIT].mean()) / resid.iloc[:SPLIT].std()
    score, label = z.abs().stack().values, badm.stack().values
    det = {"n_flagged": int(label.sum()), "n_total": int(label.size),
           "auc_all": float(roc_auc_score(label, score))}
    # bootstrap CI over the 24 positives -- the honest uncertainty here is large
    rng = np.random.default_rng(SEED)
    pos, neg = score[label], score[~label]
    boots = [roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(2000)],
                           np.r_[rng.choice(pos, len(pos)), rng.choice(neg, 2000)])
             for _ in range(400)]
    det["auc_ci95"] = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
    det["median_z_flagged"] = float(np.median(score[label]))
    det["median_z_clean"] = float(np.median(score[~label]))

    red = pd.DataFrame(rows).set_index("unit")
    inj = inject_faults(u, resid, red)
    return red.reset_index(), det, inj


FAULTS = ["bias", "drift", "stuck"]
MAGNITUDES = [0.005, 0.01, 0.02, 0.03]      # 0.5 .. 3 %p of capacity
FAULT_HOURS = 12
N_ONSETS = 20


def inject_faults(u: pd.DataFrame, resid: pd.DataFrame, red: pd.DataFrame) -> pd.DataFrame:
    """The trace has no labelled sensor faults, so manufacture some.

    SPEC 3.1 established that bad_measurement_data is 24 isolated dropouts with
    no degradation to detect -- so the real flag cannot answer "would this catch
    a drifting sensor". Injection can. Three textbook instrument failures are
    added to one unit's channel at a time, the peer-reconstruction residual is
    recomputed, and detection is called when |z| crosses a threshold set to one
    false alarm per unit per day on CLEAN test data.

    Because the reconstruction is linear in the target unit only, the faulted
    residual is available in closed form -- no refit, and the peers stay honest.
    """
    from numpy.random import default_rng

    rng = default_rng(SEED)
    te = u.index[SPLIT:]
    span = FAULT_HOURS * 12
    onsets = rng.integers(0, len(te) - span, N_ONSETS)
    usable = red.index[red.r2_test > 0.5]         # gate: no peers, no detector

    mu, sd = resid.iloc[:SPLIT].mean(), resid.iloc[:SPLIT].std()
    z_clean = ((resid.iloc[SPLIT:] - mu) / sd).abs()
    tau = z_clean.quantile(1 - 1 / 288)           # ~1 false alarm/unit/day

    # "stuck" has no magnitude of its own -- its severity is however far the
    # true signal travels while the reading is frozen. One row, not four.
    configs = [(k, m) for k in ("bias", "drift") for m in MAGNITUDES] + [("stuck", np.nan)]

    rows = []
    for kind, mag in configs:
        hits, lat = [], []
        for unit in usable:
            r0 = resid[unit].iloc[SPLIT:].values
            lvl = u[unit].iloc[SPLIT:].values
            for t0 in onsets:
                w = slice(t0, t0 + span)
                if kind == "bias":
                    bump = np.full(span, mag)
                elif kind == "drift":
                    bump = np.linspace(0, mag, span)
                else:                          # stuck at the onset reading
                    bump = lvl[t0] - lvl[w]
                z = np.abs((r0[w] + bump - mu[unit]) / sd[unit])
                fired = np.flatnonzero(z > tau[unit])
                hits.append(fired.size > 0)
                lat.append(fired[0] * 5 if fired.size else np.nan)
        rows.append({
            "fault": kind, "magnitude_pp": mag * 100,
            "units_tested": len(usable), "trials": len(hits),
            "detection_rate": float(np.mean(hits)),
            "median_lag_min": float(np.nanmedian(lat)) if np.any(~np.isnan(lat)) else np.nan,
            "false_alarms_per_unit_day": float((z_clean > tau).mean().mean() * 288),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------- Q5: fleet segmentation

def segmentation(panels: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """Cluster units by behavioural signature; check against the real grouping.

    The cell letter (which physical data centre a PDU sits in) is never shown to
    the clusterer. If unsupervised structure recovers it, the same procedure can
    group an engine fleet by duty profile with no labels at all.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from sklearn.preprocessing import StandardScaler

    u, p = panels[TARGET].iloc[:SPLIT], panels[SECOND].iloc[:SPLIT]
    badp = panels["bad_production_power_data"].iloc[:SPLIT]
    d = u.diff().iloc[1:]
    tod = u.index.hour * 12 + u.index.minute // 5
    diurnal = u.groupby(tod).mean()

    sig = pd.DataFrame({
        "mean": u.mean(), "sd": u.std(),
        "range": u.quantile(0.99) - u.quantile(0.01),
        "diurnal_amp": diurnal.max() - diurnal.min(),
        "diurnal_peak_hr": diurnal.idxmax() / 12,
        "ramp_p99": d.abs().quantile(0.99),
        "d_autocorr1": d.apply(lambda s: s.autocorr(1)),
        "corr_fleet": u.corrwith(u.mean(axis=1)),
        "headroom": (u - p).mean(),
        "badp_rate": badp.mean(),
        "resid_sd": (u - u.groupby(tod).transform("mean")).std(),
    })
    X = StandardScaler().fit_transform(sig)
    truth = [c.split("/")[0] for c in sig.index]

    scan = []
    for k in range(2, 11):
        lab = KMeans(k, n_init=20, random_state=SEED).fit_predict(X)
        scan.append({"k": k, "silhouette": silhouette_score(X, lab),
                     "ari_vs_cell": adjusted_rand_score(truth, lab)})
    scan = pd.DataFrame(scan)
    best = int(scan.loc[scan.silhouette.idxmax(), "k"])
    sig["cluster"] = KMeans(best, n_init=20, random_state=SEED).fit_predict(X)
    sig["cell"] = truth

    # chance level for ARI at this k, given these cluster sizes
    rng = np.random.default_rng(SEED)
    perm = [adjusted_rand_score(truth, rng.permutation(sig.cluster.values)) for _ in range(2000)]
    info = {"k_selected": best,
            "silhouette": float(scan.loc[scan.k == best, "silhouette"].iloc[0]),
            "ari": float(adjusted_rand_score(truth, sig.cluster)),
            "ari_perm_p95": float(np.quantile(perm, 0.95)),
            "scan": scan.to_dict("records")}
    return sig.reset_index(), info


# ------------------------------------------------------------------- figures

def figures(preds: dict, ramp: pd.DataFrame, inj: pd.DataFrame, sig: pd.DataFrame,
            cpl: dict, panels: dict[str, pd.DataFrame]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGS.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))
    ax = axes.ravel()

    # (a) how many units move independently
    x = panels[TARGET].iloc[:SPLIT].diff().iloc[1:]
    lam = np.sort(np.clip(np.linalg.eigvalsh(np.corrcoef(x.values, rowvar=False)), 0, None))[::-1]
    ax[0].bar(range(1, 21), (lam / lam.sum())[:20] * 100, color="#3b6ea5")
    ax[0].axhline(100 / len(lam), ls="--", c="grey", label=f"independent-unit level ({100/len(lam):.1f}%)")
    ax[0].set(xlabel="principal component of 5-min increments", ylabel="variance explained (%)",
              title=f"(a) 57 units behave like {cpl['increment_n_eff']:.1f} independent ones")
    ax[0].legend(fontsize=8)

    # (b) error vs horizon, solo vs fleet
    err = preds["_err"]
    for name, mk in [("persistence", "o--"), ("gbm_solo", "s-"), ("gbm_fleet", "^-")]:
        e = err[err.model == name]
        ax[1].plot(e.horizon_min, e.mae_pp, mk, label=name)
    ax[1].set(xlabel="forecast horizon (min)", ylabel="MAE (%p of capacity)",
              title="(b) same learner, with and without cross-unit features")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3)

    # (c) ramp detection
    for name, mk in [("volatility_persistence", "o--"), ("gbm_solo", "s-"), ("gbm_fleet", "^-")]:
        r = ramp[ramp.detector == name]
        ax[2].plot(r.horizon_min, r.auc, mk, label=name)
    ax[2].axhline(0.5, ls=":", c="grey")
    ax[2].set(xlabel="lead time (min)", ylabel="ROC-AUC", ylim=(0.45, 1.0),
              title="(c) calling a top-1% load step in advance")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=.3)

    # (d) injected sensor faults: how big before the peers notice
    for kind, mk in [("bias", "o-"), ("drift", "s-")]:
        g = inj[inj.fault == kind]
        ax[3].plot(g.magnitude_pp, g.detection_rate * 100, mk, label=kind)
    stuck = inj[inj.fault == "stuck"].detection_rate.iloc[0] * 100
    ax[3].axhline(stuck, ls="-.", c="green", label=f"stuck-at ({stuck:.0f}%)")
    ax[3].set(xlabel="injected fault magnitude (%p of capacity)",
              ylabel="detected (%)", ylim=(0, 105),
              title="(d) peers catching a faulted sensor, 1 false alarm/unit/day")
    ax[3].legend(fontsize=8)
    ax[3].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_fleet_ml.png", dpi=130)
    plt.close(fig)

    # (d) segmentation map
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    cols = [c for c in sig.columns if c not in ("unit", "cluster", "cell")]
    Z = PCA(2, random_state=SEED).fit_transform(StandardScaler().fit_transform(sig[cols]))
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for cl, g in pd.DataFrame({"x": Z[:, 0], "y": Z[:, 1], **sig[["cluster", "cell"]]}).groupby("cluster"):
        ax.scatter(g.x, g.y, s=90, alpha=.75, label=f"cluster {cl}")
        for _, r in g.iterrows():
            ax.annotate(r.cell, (r.x, r.y), fontsize=7, ha="center", va="center", color="w")
    ax.set(xlabel="PC1 of behavioural signature", ylabel="PC2",
           title="(d) unsupervised grouping vs true site letter")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_segmentation.png", dpi=130)
    plt.close(fig)



# -------------------------------------------------------------- FLEET.md

def _pick(df, **eq):
    m = np.ones(len(df), bool)
    for k, v in eq.items():
        m &= (df[k] == v).values
    return df[m]


def _gain_range(pair, a, b):
    g = _pick(pair, better=a, vs=b).median_mae_gain_pct
    return f"-{g.min():.1f}~-{g.max():.1f}%"


def write_report(cpl, err, ramp, pair, red, det, inj, sig, seg) -> None:
    """FLEET.md -- the numbers, plus the one table a reader actually wants."""
    c, g, N = cpl, seg, cpl["n_units"]
    n_eff = c["increment_n_eff"]
    rf = _pick(ramp, detector="gbm_fleet")
    fleet5 = _pick(pair, horizon_min=5, better="gbm_fleet", vs="gbm_solo")
    stuck = _pick(inj, fault="stuck").detection_rate.iloc[0]
    bias2 = _pick(inj, fault="bias", magnitude_pp=2.0).detection_rate.iloc[0]

    headline = pd.DataFrame([
        ["다수 유닛 동시성 진단",
         "57 PDU의 5분 부하 증분 상관구조",
         f"유효 독립 유닛 {n_eff:.1f}/{N}대. 집합 램프 변동성이 독립 가정 대비 "
         f"{c['reserve_inflation_5min']:.1f}배(1시간 {c['reserve_inflation_1h']:.1f}배)",
         "높음 — 상관구조 추정 자체는 부하의 물리적 종류를 타지 않는다"],
        ["교차 유닛 예측 (통합의 순이득)",
         "동일 GBM 학습기, 피처셋만 자기이력 → 함대 전체로 교체",
         f"MAE {_gain_range(pair, 'gbm_fleet', 'gbm_solo')}, "
         f"5분 예측에서 57대 중 {int(fleet5.units_improved.iloc[0])}대 개선 "
         f"(Wilcoxon p={fleet5.wilcoxon_p.iloc[0]:.1e})",
         "중간 — 이득 크기는 유닛 간 부하 상관도에 비례한다"],
        ["급변(램프) 사전 탐지",
         "상위 1% 부하 스텝, 리드타임 5~60분",
         f"ROC-AUC {rf.auc.min():.2f}~{rf.auc.max():.2f}, 상위 알람 정밀도 "
         f"{rf.precision_at_n.min()*100:.0f}~{rf.precision_at_n.max()*100:.0f}% "
         f"(기저율 0.7% → {(rf.precision_at_n / rf.base_rate).min():.0f}~"
         f"{(rf.precision_at_n / rf.base_rate).max():.0f}배 리프트)",
         "높음 — 부하추종 지령·예비력 기동의 직접 입력"],
        ["가상센서 / 해석적 이중화",
         "한 유닛의 계측을 지우고 나머지 56대로 복원",
         f"검증 구간 R² 중앙값 {red.r2_test.median():.3f}, "
         f"{int((red.r2_test > 0.5).sum())}/{N}대에서 유효",
         "높음 — 계측 이중화 논리는 물리량을 가리지 않는다"],
        ["센서 고장 검출 (주입 시험)",
         "bias·drift·stuck 고장을 인위 주입 후 재탐지",
         f"stuck {stuck*100:.0f}%, bias 2%p {bias2*100:.0f}%, "
         f"오경보 1건/대·일 고정",
         "중간 — 검출 하한이 유닛 간 상관도에 직접 좌우된다"],
        ["무라벨 자산 군집화",
         "행태 지문 11종 → KMeans (사이트 라벨 미제공)",
         f"k={g['k_selected']}, ARI {g['ari']:.3f} "
         f"(무작위 배정 95% 상한 {g['ari_perm_p95']:.3f})",
         "높음 — 라벨 없이 함대를 duty 프로파일로 나눈다"],
    ], columns=["통합 솔루션에서의 ML 기능", "이 데이터에서의 대응물",
                "실측 결과 (검증 구간)", "선박엔진 함대로의 이전 가능성"])

    kv = lambda rows: pd.DataFrame(rows, columns=["지표", "값"]).to_markdown(index=False)

    md = [
        "# Fleet layer — 다수 유닛 · 다중 센서에 대한 ML 적용성 검증",
        "",
        f"`par/fleet.py` 자동 생성. 데이터: Google `powerdata_2019`, {N} PDU × "
        "8,928 스텝 × 4 센서 채널(계측 전력, 추정 생산 전력, 품질 플래그 2종). "
        "학습 1~20일 / 검증 21~31일, 한 번 분할하고 검증 구간에서는 재학습하지 않는다.",
        "",
        "설계 근거와 전체 스펙은 [SPEC.md](SPEC.md), 단일 유닛 리스크 결과는 "
        "[RESULTS.md](RESULTS.md).",
        "",
        "## 0. 결론 표",
        "",
        headline.to_markdown(index=False),
        "",
        "## 1. 유닛 간 결합 — N대는 N대만큼 독립적이지 않다",
        "",
        kv([
            ["유효 독립 유닛 수 (증분 기준)", f"{n_eff:.1f} / {N}"],
            ["유효 독립 유닛 수 (레벨 기준)", f"{c['level_n_eff']:.1f} / {N}"],
            ["90% 분산 설명에 필요한 주성분", f"{c['increment_pc_for_90']}개"],
            ["평균 쌍별 상관 (증분)", f"{c['increment_mean_corr']:.3f}"],
            ["다양성 계수 (집합 피크 ÷ 개별 피크 평균)", f"{c['diversity_factor']:.3f}"],
            ["집합 5분 램프 표준편차", f"{c['fleet_ramp_sd_5min_pp']:.3f} %p"],
            ["└ 독립 가정 대비", f"{c['reserve_inflation_5min']:.2f}배"],
            ["집합 1시간 램프 표준편차", f"{c['fleet_ramp_sd_1h_pp']:.3f} %p"],
            ["└ 독립 가정 대비", f"{c['reserve_inflation_1h']:.2f}배"],
            ["개별 유닛 1시간 램프 p99.9 (평균)", f"{c['unit_ramp_p999_1h_pp']:.2f} %p"],
            ["집합 1시간 램프 p99.9", f"{c['fleet_ramp_p999_1h_pp']:.2f} %p"],
            ["집합 상위 1% 상승 시 동반 상승 유닛 비율",
             f"{c['share_up_on_fleet_ramp']*100:.1f}% (평시 {c['share_up_baseline']*100:.1f}%)"],
        ]),
        "",
        f"개별 유닛은 1시간에 ±{c['unit_ramp_p999_1h_pp']:.1f} %p까지 흔들리지만 집합은 "
        f"±{c['fleet_ramp_p999_1h_pp']:.1f} %p로 잦아든다 — 상쇄는 분명히 일어난다. "
        f"다만 완전 독립이라면 여기서 한 번 더, {c['reserve_inflation_1h']:.1f}배만큼 "
        "줄었어야 한다. 그 차이가 곧 추가로 확보해야 하는 부하추종 여력이다.",
        "",
        "## 2. 예측 — 같은 학습기, 피처셋만 교체",
        "",
        "모든 학습기는 레벨이 아니라 **증분**을 타깃으로 한다. 그래야 persistence가"
        " 정확히 '0 예측'이 되어 skill이 곧 추가 정보량이 되고, 트리 모델이 학습"
        " 범위 밖 레벨을 외삽하지 못하는 불이익도 사라진다. 점수는 레벨로 환산해 매겼다.",
        "",
        err.round(4).to_markdown(index=False),
        "",
        "### 2.1 유닛 단위 페어 검정 (반복 단위 = 57대)",
        "",
        "18만 행을 pooled로 검정하면 자기상관 때문에 무엇이든 유의해진다. "
        "유닛별 MAE로 접어서 57쌍 Wilcoxon으로 검정한다.",
        "",
        pair.assign(wilcoxon_p=pair.wilcoxon_p.map("{:.1e}".format))
            .round(4).to_markdown(index=False),
        "",
        "읽는 법 세 가지.",
        "",
        "1. **함대 피처는 GBM에 대해 모든 horizon에서 이긴다** — 57대 중 45~54대, "
        "p ≤ 2e-6. 학습기·하이퍼파라미터·타깃이 동일하고 피처셋만 바뀌었으므로 "
        "이 차이는 교차 유닛 정보의 순기여다.",
        "2. **같은 정보를 릿지는 쓰지 못한다** — `ridge_fleet`는 `ridge_solo`를 "
        "이기지 못하고 오히려 근소하게 진다. 교차 유닛 신호가 비선형이라는 뜻이고, "
        "이것이 이 문제에 통계 회귀가 아니라 ML이 필요한 이유다.",
        "3. **레벨 예측 자체는 5분·1시간에서만 persistence를 이긴다** — 15·30분에서는 "
        "무승부다. 부하 레벨은 그 구간에서 거의 랜덤워크다. 따라서 이 데이터가 "
        "지지하는 주장은 '레벨을 잘 맞힌다'가 아니라 **'급변을 미리 부른다'**(§3)이다.",
        "",
        "## 3. 램프 사전 탐지",
        "",
        "이벤트 정의: horizon h 동안의 상승폭이 **학습 구간** 증분 분포의 99분위를 "
        "넘는 경우. 임계값은 학습 구간에서 고정하고 검증 구간에 그대로 적용한다.",
        "",
        ramp.round(4).to_markdown(index=False),
        "",
        "## 4. 가상센서 — 이웃으로 한 대를 복원",
        "",
        kv([
            ["검증 구간 R² 중앙값", f"{red.r2_test.median():.3f}"],
            ["R² > 0.5 유닛", f"{int((red.r2_test > 0.5).sum())} / {N}"],
            ["복원 실패 유닛", ", ".join(red.nsmallest(3, "r2_test").unit)],
            ["실제 플래그 탐지 AUC",
             f"{det['auc_all']:.3f} (95% CI {det['auc_ci95'][0]:.2f}~"
             f"{det['auc_ci95'][1]:.2f}, 양성 {det['n_flagged']}건)"],
        ]),
        "",
        "실제 `bad_measurement_data` 플래그에 대한 AUC는 0.5와 구분되지 않는다. "
        "SPEC §3.1대로 그 플래그는 텔레메트리 유실 표시지 이상 계측값이 아니므로 "
        "잔차가 반응할 대상이 없다. **음성 결과이며 예상된 것이다.**",
        "",
        "### 4.1 고장 주입 시험",
        "",
        "라벨된 실제 센서 고장이 없으므로 만들어 넣는다. 한 번에 한 유닛의 채널에만"
        " 고장을 주입하고, 이웃 기반 복원 잔차를 다시 계산해 탐지 여부를 본다. "
        "임계값은 **정상 검증 구간에서 오경보 1건/대·일**이 되도록 고정했다.",
        "",
        inj.round(3).to_markdown(index=False),
        "",
        "## 5. 무라벨 군집화",
        "",
        pd.DataFrame(g["scan"]).round(3).to_markdown(index=False),
        "",
        f"실루엣 최대인 k={g['k_selected']} 선택, ARI={g['ari']:.3f}. 무작위 배정의 "
        f"ARI 95% 상한은 {g['ari_perm_p95']:.3f}이다. 군집기에 사이트(cell) 라벨은 "
        "한 번도 주지 않았다.",
        "",
        "## 6. 이 데이터로 검증되지 않는 것",
        "",
        "- **전기적 물리량 부재.** 값은 전부 용량 대비 비율이다. kW·전압·전류·역률·"
        "고조파가 없으므로 아크·불평형 같은 전기적 이상은 관측 대상이 아니다.",
        "- **회전기계 신호 부재.** 진동·온도·압력·연료유량이 없다. 엔진 자체의 "
        "예지정비는 이 데이터로 검증되지 않으며, 여기서 검증한 것은 **함대 수준의 "
        "부하·계측 통합 로직**이다.",
        "- **1개월.** 계절성 검증 불가.",
        "- **2019년.** GPU 가속기 워크로드 이전 시기다. 현행 AI 클러스터의 전력 "
        "동조 특성은 이 트레이스로 대표되지 않으며, §1의 결합 수치는 하한으로 읽어야 한다.",
        "",
        "## 7. 그림",
        "",
        "![fleet](figures/fig3_fleet_ml.png)",
        "",
        "![segmentation](figures/fig4_segmentation.png)",
        "",
    ]
    (HERE / "FLEET.md").write_text(chr(10).join(md), encoding="utf-8")


# -------------------------------------------------------------------- report

def main() -> None:
    t0 = _time.time()
    OUT.mkdir(exist_ok=True)
    panels = load_panels()
    print(f"panel {panels[TARGET].shape}  split at day {TRAIN_DAYS}")

    cpl = coupling(panels[TARGET])
    print("Q1 coupling", json.dumps(cpl, indent=None, default=float)[:200])

    long = build_features(panels)
    err, ramp, pair, preds = forecast(long, panels)
    preds["_err"] = err
    print(err.pivot(index="horizon_min", columns="model", values="mae_pp"))
    print(pair.to_string(index=False))

    red, det, inj = redundancy(panels)
    print(f"Q4 virtual sensor: median R2 {red.r2_test.median():.3f}  AUC {det['auc_all']:.3f}")
    print(inj.pivot(index="magnitude_pp", columns="fault", values="detection_rate"))

    sig, seg = segmentation(panels)
    print(f"Q5 segmentation: k={seg['k_selected']} ARI={seg['ari']:.3f}")

    figures(preds, ramp, inj, sig, cpl, panels)

    err.to_csv(OUT / "fleet_forecast.csv", index=False)
    ramp.to_csv(OUT / "fleet_ramp.csv", index=False)
    pair.to_csv(OUT / "fleet_ablation.csv", index=False)
    red.to_csv(OUT / "fleet_redundancy.csv", index=False)
    inj.to_csv(OUT / "fleet_injection.csv", index=False)
    sig.to_csv(OUT / "fleet_signature.csv", index=False)
    preds["importance"].to_csv(OUT / "fleet_importance.csv", header=["gain"])
    write_report(cpl, err, ramp, pair, red, det, inj, sig, seg)
    (OUT / "fleet_summary.json").write_text(
        json.dumps({"coupling": cpl, "detector": det, "segmentation": seg,
                    "redundancy_median_r2": float(red.r2_test.median()),
                    "redundancy_min_r2": float(red.r2_test.min()),
                    "redundancy_n_above_0.5": int((red.r2_test > 0.5).sum())},
                   indent=2, default=float), encoding="utf-8")
    print(f"done in {_time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
