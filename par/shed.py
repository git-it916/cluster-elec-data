"""Can the electrical side alone tell you how much load is sheddable?

A genset supplier sees volts and amps. It does not see the job scheduler. The
trace happens to contain both sides:

    measured_power_util     what the meter reads          -- electrical side
    production_power_util   CPU-interpolated estimate of
                            non-sheddable production load -- IT side

so headroom = measured - production is the sheddable share, and it is the one
number a capping policy needs. It is not metered anywhere; it is inferred from
IT telemetry the supplier has no access to, and the trace itself marks that
inference invalid 22.9% of the time.

This file asks the only question that matters for the proposal: predict
headroom from the METER ALONE -- the production channel is withheld from every
feature. Two baselines make the answer honest:

    const   the unit's own mean headroom from the training half
    naive   u_t minus the unit's own mean production from the training half
            (i.e. "assume the IT side is flat") -- the estimator any engineer
            would write down first, and the one the model has to beat

Train days 1-20, test days 21-31.

    python par/shed.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import fleet  # noqa: E402

HERE = Path(__file__).parent
SPLIT = fleet.SPLIT

# every feature that touches the IT side is withheld
METER_ONLY = [c for c in fleet.FLEET_COLS if c not in ("p", "headroom", "p_d1", "badp")]


def r2(truth: np.ndarray, pred: np.ndarray) -> float:
    return float(1 - ((truth - pred) ** 2).sum() / ((truth - truth.mean()) ** 2).sum())


def main() -> None:
    import lightgbm as lgb

    panels = fleet.load_panels()
    u, p = panels[fleet.TARGET], panels[fleet.SECOND]
    head = (u - p).stack()

    long = fleet.build_features(panels)
    d = long.copy()
    d["y"] = head.reindex(d.index)
    d = d.dropna(subset=["y"] + METER_ONLY)
    tr, te = d[d.step < SPLIT], d[d.step >= SPLIT]
    y = te.y.values
    unit_te = te.index.get_level_values("unit").values

    # baselines, per unit, fitted on the train half only
    mean_head = tr.groupby(level="unit").y.mean()
    mean_prod = (u.iloc[:SPLIT] - (u.iloc[:SPLIT] - p.iloc[:SPLIT])).mean()  # = mean p
    b_const = mean_head.reindex(unit_te).values
    b_naive = te.u.values - mean_prod.reindex(unit_te).values

    m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63,
                          verbose=-1, random_state=fleet.SEED)
    m.fit(tr[METER_ONLY], tr.y)
    pred = m.predict(te[METER_ONLY])

    rows = []
    for name, q in [("상수 (유닛 평균 헤드룸)", b_const),
                    ("나이브 (전력계 − 평균 프로덕션)", b_naive),
                    ("전기 계측 전용 모델", pred)]:
        per_unit = pd.Series(np.abs(y - q), index=te.index).groupby(level="unit").mean()
        rows.append({"추정기": name, "R2": r2(y, q),
                     "MAE_pp": np.abs(y - q).mean() * 100,
                     "최악유닛_MAE_pp": per_unit.max() * 100})
    res = pd.DataFrame(rows)

    # does it hold where the trace says the IT estimate is untrustworthy?
    bad = panels["bad_production_power_data"].stack().reindex(te.index).astype(bool).values
    seg = pd.DataFrame([
        {"구간": "IT 추정 신뢰 (플래그 없음)", "n": int((~bad).sum()),
         "모델_R2": r2(y[~bad], pred[~bad]),
         "나이브_R2": r2(y[~bad], b_naive[~bad])},
        {"구간": "IT 추정 무효 (플래그)", "n": int(bad.sum()),
         "모델_R2": r2(y[bad], pred[bad]) if bad.sum() else np.nan,
         "나이브_R2": r2(y[bad], b_naive[bad]) if bad.sum() else np.nan},
    ])

    print(res.round(4).to_string(index=False))
    print()
    print(seg.round(4).to_string(index=False))
    print()
    imp = pd.Series(m.feature_importances_, index=METER_ONLY).sort_values(ascending=False)
    print("상위 피처:", ", ".join(imp.head(6).index))
    res.to_csv(HERE / "data" / "shed_scores.csv", index=False)


if __name__ == "__main__":
    main()
