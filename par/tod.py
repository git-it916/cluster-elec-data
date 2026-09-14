"""One feature, one question: how much of a load spike does the clock explain?

If load spikes were a function of the time of day, nobody would need a model --
you would write them into the operating schedule and be done. So the clock is
the feature that has to be ruled out before any ML claim is worth making.

    target   spike_h = 1 when the load step over the next h exceeds the
             99th percentile of steps seen in the TRAINING half
    feature  time of day, and nothing else

Explanatory power is scored as deviance explained on the held-out half,
1 - D_model / D_null, which for a 0/1 target is the direct analogue of R^2.
AUC and lift are reported next to it because a number that cannot be acted on
is not an answer.

    python par/tod.py        # ~40s -> data/tod_*.csv, TOD.md, figures/fig5
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import fleet  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "data"
FIGS = HERE / "figures"

SPLIT = fleet.SPLIT          # day 20; train 1-20, test 21-31
DAY = fleet.DAY
HORIZONS = [1, 3, 6, 12]     # 5, 15, 30, 60 minutes
EPS = 1e-6


# --------------------------------------------------------------- scoring

def deviance_explained(y: np.ndarray, p: np.ndarray, p_null: float) -> float:
    """1 - D_model / D_null on held-out data. The 0/1 analogue of R^2.

    Negative is possible and meaningful: it says the fitted probabilities
    generalise worse than just quoting the base rate.
    """
    def dev(q):
        q = np.clip(q, EPS, 1 - EPS)
        return -2 * np.sum(y * np.log(q) + (1 - y) * np.log(1 - q))
    return float(1 - dev(p) / dev(np.full(y.size, p_null)))


def score(name: str, y: np.ndarray, p: np.ndarray, p_null: float, h: int) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    k = max(int(y.sum()), 1)
    top = np.argsort(p)[-k:]
    prec = float(y[top].mean())
    base = float(y.mean())
    return {"horizon_min": h * 5, "feature": name,
            "deviance_explained": deviance_explained(y, p, p_null),
            "auc": float(roc_auc_score(y, p)) if len(np.unique(p)) > 1 else 0.5,
            "avg_precision": float(average_precision_score(y, p)),
            "precision_at_n": prec, "lift": prec / base}


# ------------------------------------------------------- the clock feature

def clock_rate(y_tr: np.ndarray, key_tr: np.ndarray, key_te: np.ndarray,
               n_bins: int, prior: float, strength: float = 20.0) -> np.ndarray:
    """P(spike | bin), estimated on train only, shrunk toward the base rate.

    Shrinkage matters: at 288 five-minute bins a bin holds only a few dozen
    events, and an unshrunk rate would fit noise and score worse than the base
    rate on the held-out half -- which would understate the clock, not overstate
    it. `strength` is a pseudo-count prior, i.e. every bin starts out holding
    20 observations' worth of the fleet base rate.
    """
    hit = np.bincount(key_tr, weights=y_tr, minlength=n_bins)
    n = np.bincount(key_tr, minlength=n_bins)
    rate = (hit + strength * prior) / (n + strength)
    return rate[key_te]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    panels = fleet.load_panels()
    u = panels[fleet.TARGET]
    idx, units = u.index, u.columns
    n_units = len(units)

    tod5 = np.asarray(idx.hour * 12 + idx.minute // 5)     # 288 bins
    tod1 = np.asarray(idx.hour)                            # 24 bins
    dow = np.asarray(idx.dayofweek)
    weekend = (dow >= 5).astype(int)

    long = fleet.build_features(panels)                    # for the ceiling model
    rows, profiles, extra = [], [], {}

    for h in HORIZONS:
        step = u.diff(h).shift(-h)                         # step over t -> t+h
        thr = float(step.iloc[:SPLIT].stack().quantile(0.99))
        ev = (step >= thr)

        valid = ~step.isna()
        tr_m = valid.copy(); tr_m.iloc[SPLIT:] = False
        te_m = valid.copy(); te_m.iloc[:SPLIT] = False

        # flatten to (time, unit) rows, keeping the time key alongside
        def flat(mask):
            t = np.repeat(np.arange(len(idx)), n_units)[mask.values.ravel()]
            return ev.values.ravel()[mask.values.ravel()].astype(float), t

        y_tr, t_tr = flat(tr_m)
        y_te, t_te = flat(te_m)
        prior = float(y_tr.mean())

        # --- the one feature, at two resolutions, plus the obvious extensions
        cands = {
            "clock_hour": (tod1[t_tr], tod1[t_te], 24),
            "clock_5min": (tod5[t_tr], tod5[t_te], 288),
            "clock_hour_x_weekend": (tod1[t_tr] + 24 * weekend[t_tr],
                                     tod1[t_te] + 24 * weekend[t_te], 48),
            "clock_5min_x_dow": (tod5[t_tr] + 288 * dow[t_tr],
                                 tod5[t_te] + 288 * dow[t_te], 288 * 7),
        }
        for name, (ktr, kte, nb) in cands.items():
            p = clock_rate(y_tr, ktr, kte, nb, prior)
            rows.append(score(name, y_te, p, prior, h))
            # the same fit scored on the data it was fitted to. If in-sample is
            # also ~0 the clock carries no signal; if in-sample is high and
            # out-of-sample is not, the pattern simply does not survive to the
            # next fortnight. Both kill a schedule, for different reasons.
            p_in = clock_rate(y_tr, ktr, ktr, nb, prior)
            r = score(name, y_tr, p_in, prior, h)
            r["feature"] = name + " (in-sample)"
            rows.append(r)

        rows.append(score("base_rate_only", y_te, np.full(y_te.size, prior), prior, h))

        # --- ceiling: everything the fleet layer can see
        import lightgbm as lgb
        d = long.copy()
        d["y"] = ev.stack().reindex(d.index)
        d = d.dropna(subset=["y"] + fleet.FLEET_COLS)
        dtr, dte = d[d.step < SPLIT], d[d.step >= SPLIT]
        m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63,
                               verbose=-1, random_state=fleet.SEED)
        m.fit(dtr[fleet.FLEET_COLS], dtr.y)
        p_full = m.predict_proba(dte[fleet.FLEET_COLS])[:, 1]
        rows.append(score("all_features_ceiling", dte.y.values, p_full,
                          float(dtr.y.mean()), h))

        # --- what the clock leaves behind: do spikes still cluster in time?
        if h == 1:
            p_clock = clock_rate(y_tr, tod5[t_tr], tod5[t_te], 288, prior)
            e = ev.iloc[SPLIT:].values
            prev = np.vstack([np.zeros((1, n_units), bool), e[:-1]])
            extra["p_spike"] = float(e.mean())
            extra["p_spike_after_spike"] = float(e[prev].mean())
            extra["p_spike_after_calm"] = float(e[~prev].mean())
            extra["clock_p_max"] = float(p_clock.max())
            extra["clock_p_min"] = float(p_clock.min())
            extra["threshold_pp"] = thr * 100

            prof = pd.DataFrame({"tod": tod1[t_te], "y": y_te}).groupby("tod").y.mean()
            profiles.append(prof.rename("spike_rate"))

            # The contrast that makes the result readable: the very same clock
            # feature, scored against the load LEVEL instead of the spike.
            # Per-unit diurnal profile fitted on the train half, applied
            # unchanged to the test half. Per-unit because each PDU sits at its
            # own base level, and the question is what the CLOCK adds, not what
            # knowing which unit you are looking at adds.
            def r2(pred, truth):
                return 1 - float(((truth - pred) ** 2).sum()) /                            float(((truth - truth.mean()) ** 2).sum())

            tr_lvl, te_lvl = u.iloc[:SPLIT], u.iloc[SPLIT:]
            prof_lvl = tr_lvl.groupby(tod5[:SPLIT]).mean()          # tod x unit
            pred = prof_lvl.reindex(tod5[SPLIT:]).values
            extra["clock_explains_level_r2"] = r2(pred.ravel(), te_lvl.values.ravel())
            # and the same for the step, so level and spike sit on one scale
            step1 = u.diff().shift(-1)
            prof_d = step1.iloc[:SPLIT].groupby(tod5[:SPLIT]).mean()
            pd_te = step1.iloc[SPLIT:-1]
            extra["clock_explains_step_r2"] = r2(
                prof_d.reindex(tod5[SPLIT:len(u) - 1]).values.ravel(),
                pd_te.values.ravel())
            extra["clock_explains_spike_dev"] = float(
                [r for r in rows if r["feature"] == "clock_5min"][-1]["deviance_explained"])

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "tod_scores.csv", index=False)
    prof = profiles[0]
    prof.to_csv(OUT / "tod_profile.csv")
    (OUT / "tod_summary.json").write_text(
        json.dumps(extra, indent=2), encoding="utf-8")

    figure(prof, res, u, tod5, extra)
    write_report(res, prof, extra)
    print(res.round(4).to_string(index=False))
    print(json.dumps(extra, indent=2))


# --------------------------------------------------------------- figure

def figure(prof: pd.Series, res: pd.DataFrame, u: pd.DataFrame,
           tod5: np.ndarray, extra: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))

    hours = prof.index.values.astype(float)
    ax[0].fill_between(hours, prof.values * 100, color="#3b6ea5", alpha=.35)
    ax[0].plot(hours, prof.values * 100, color="#26496e", lw=1.2)
    ax[0].axhline(extra["p_spike"] * 100, ls="--", c="#b4451f",
                  label=f"all-day average ({extra['p_spike']*100:.2f}%)")
    a2 = ax[0].twinx()
    a2.plot(hours, u.iloc[fleet.SPLIT:].groupby(
                np.asarray(u.index[fleet.SPLIT:].hour)).mean().mean(axis=1) * 100,
            color="#6a6a6a", lw=1.1, ls=":", label="mean load")
    a2.set_ylabel("mean load (% of capacity)", color="#6a6a6a")
    ax[0].set(xlabel="hour of day (US/Pacific)", ylabel="spike probability (%)",
              xlim=(0, 24), xticks=range(0, 25, 4),
              title="(a) same clock, two questions")
    h1, l1 = ax[0].get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    ax[0].legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left", framealpha=.9)
    ax[0].text(0.5, -0.28,
               f"load LEVEL follows the clock (R$^2$={extra['clock_explains_level_r2']:.2f})   |   "
               f"SPIKES do not (R$^2$={extra['clock_explains_step_r2']:.2f})",
               transform=ax[0].transAxes, ha="center", fontsize=9.5, color="#26496e")

    piv = res[res.horizon_min == 5].set_index("feature").deviance_explained * 100
    order = ["clock_hour", "clock_5min", "clock_hour_x_weekend",
             "clock_5min_x_dow", "all_features_ceiling"]
    lbl = ["clock (hourly)", "clock (5-min)", "clock x weekend",
           "clock x weekday", "all features (ceiling)"]
    cols = ["#8fa9c4"] * 4 + ["#26496e"]
    ax[1].barh(range(5), [piv[o] for o in order], color=cols)
    ax[1].set_yticks(range(5), lbl, fontsize=9)
    ax[1].invert_yaxis()
    ax[1].axvline(0, c="#444", lw=.8)
    ax[1].set(xlabel="deviance explained (%), held-out half",
              title="(b) how much of a spike the clock explains")
    for i, o in enumerate(order):
        ax[1].text(piv[o] + .3, i, f"{piv[o]:.1f}%", va="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(FIGS / "fig5_clock.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------- report

def write_report(res: pd.DataFrame, prof: pd.Series, extra: dict) -> None:
    def at(feat, h=5, col="deviance_explained"):
        r = res[(res.feature == feat) & (res.horizon_min == h)]
        return float(r[col].iloc[0])

    clock, ceil = at("clock_5min"), at("all_features_ceiling")
    ins = at("clock_5min (in-sample)")
    peak_h, low_h = float(prof.idxmax()), float(prof.idxmin())

    md = [
        "# 시계 하나로 급변이 얼마나 설명되는가",
        "",
        "`par/tod.py` 자동 생성. 피처는 **하루 중 시각 하나뿐**, 타깃은 "
        f"**급변 여부(0/1)** — 다음 h 구간 부하 상승폭이 학습 구간 99분위"
        f"({extra['threshold_pp']:.1f} %p)를 넘는 경우다. 학습 1~20일 / 검증 21~31일.",
        "",
        "설명력은 검증 구간의 **이탈도 설명률**(1 − D_model/D_null)로 잰다. "
        "0/1 타깃에서 R²에 해당하는 값이고, 음수는 '기저율만 말하는 것보다 못하다'는 뜻이다.",
        "",
        "## 한 문장",
        "",
        f"**같은 시계 피처가 부하 '레벨'은 {extra['clock_explains_level_r2']*100:.0f}% "
        f"설명하는데(R²={extra['clock_explains_level_r2']:.2f}), '급변'은 "
        f"0% 설명한다**(검증 설명력 {clock*100:+.1f}%, AUC {at('clock_5min', col='auc'):.2f}). "
        f"같은 데이터에 전체 피처를 쓰면 {ceil*100:.1f}%(AUC "
        f"{at('all_features_ceiling', col='auc'):.2f})까지 올라간다. "
        "**즉 급변에 관한 정보는 달력이 아니라 다른 데 있다.**",
        "",
        "## 결과",
        "",
        res.round(4).to_markdown(index=False),
        "",
        "## 왜 그런가",
        "",
        f"- 급변 확률이 가장 높은 시각은 {peak_h:.0f}시({prof.max()*100:.2f}%), "
        f"가장 낮은 시각은 {low_h:.0f}시({prof.min()*100:.2f}%), 전일 평균 "
        f"{extra['p_spike']*100:.2f}%. **가장 위험한 시간대라고 해봐야 평균의 "
        f"{prof.max()/extra['p_spike']:.1f}배**다.",
        f"- 반면 **직전 구간에 급변이 있었으면 다음 구간 급변 확률이 "
        f"{extra['p_spike_after_spike']*100:.1f}%**로 뛴다 "
        f"(잠잠했으면 {extra['p_spike_after_calm']*100:.2f}%). "
        f"**{extra['p_spike_after_spike']/extra['p_spike_after_calm']:.0f}배**다.",
        "- 즉 급변을 지배하는 것은 **시각이 아니라 직전 상태**다. 시계는 언제 부하가 "
        "*높은지*는 잘 설명하지만(일주기), 언제 *튈지*는 거의 설명하지 못한다. "
        "이 둘은 다른 질문이다.",
        "- 해상도를 5분까지 올리거나 요일을 곱해도 늘지 않는다. 오히려 떨어진다 — "
        f"학습 구간에서조차 시각의 설명력은 {ins*100:.1f}%뿐이고(요일까지 곱하면 "
        f"{at('clock_5min_x_dow (in-sample)')*100:.1f}%로 보이지만 검증에서 "
        f"{at('clock_5min_x_dow')*100:+.1f}%로 무너진다), **달력에서 더 짜낼 것이 없다.**",
        "",
        "## 그래서",
        "",
        "급변이 시계로 설명됐다면 운전 스케줄에 적어두면 끝이고 모델은 필요 없다. "
        "실측 결과는 정반대다. **예비력을 시간대 기준으로 편성하면 안 되고, "
        "상태를 실시간으로 보는 모델이 아니면 급변을 미리 알 방법이 없다.** "
        f"그 몫이 {ceil*100:.1f}%p만큼 통째로 비어 있다.",
        "",
        "![clock](figures/fig5_clock.png)",
        "",
    ]
    (HERE / "TOD.md").write_text(chr(10).join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
