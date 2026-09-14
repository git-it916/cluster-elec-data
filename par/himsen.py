"""Translate the trace's dimensionless numbers into HiMSEN units.

The trace only ever gives ratios of capacity. On its own "a 1.4 %p step" means
nothing to an engine person. This file does one job: map the aggregate load
of the traced data centre onto a HiMSEN plant of a stated size, and restate
every ramp figure as "how many 9.6 MW engines' worth, and how often".

Nothing is modelled here. It is arithmetic on the numbers fleet.py/tod.py
already measured, made legible.

    python par/himsen.py       # -> HIMSEN.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import fleet  # noqa: E402

HERE = Path(__file__).parent
FIGS = HERE / "figures"

# HD Hyundai's Corban order: 1,000 MW of 9.6 MW HiMSEN gensets.
SITE_MW = 1000.0
ENGINE_MW = 9.6
N_ENGINES = SITE_MW / ENGINE_MW

HORIZONS = [(1, "5분"), (3, "15분"), (6, "30분"), (12, "1시간")]


def main() -> None:
    u = fleet.load_panels()[fleet.TARGET]
    # The site's total load = the mean utilisation across the 57 domains.
    # (They are ratios of their own capacity, so the mean is the plant load
    # factor. Equal-capacity domains is the assumption; the trace gives no
    # ratings to weight by, and it is stated as a limit rather than hidden.)
    site = u.mean(axis=1)

    rows = []
    for k, label in HORIZONS:
        mw = (site.diff(k).dropna()) * SITE_MW     # %p of site -> MW
        up = mw[mw > 0]
        # Frequency is counted on NON-OVERLAPPING windows -- a rolling diff
        # would count the same hour twelve times and inflate the number.
        blocks = mw.iloc[::k]
        per_day = float((blocks >= ENGINE_MW).sum() / (len(blocks) * k / fleet.DAY))
        rows.append({
            "구간": label,
            "표준편차_MW": mw.std(),
            "p99_MW": up.quantile(0.99),
            "p999_MW": up.quantile(0.999),
            "최대_MW": mw.max(),
            "최대_엔진대수": mw.max() / ENGINE_MW,
            "엔진1대분_초과_빈도_일": per_day,
            "엔진2대분_초과_빈도_주": float(
                (blocks >= 2 * ENGINE_MW).sum() / (len(blocks) * k / fleet.DAY) * 7),
        })
    ramp = pd.DataFrame(rows)

    # What the independence assumption would have told you to build.
    cpl = fleet.coupling(u)
    reserve = pd.DataFrame([{
        "구간": lab,
        "실측_추종여력_MW": cpl[f"fleet_ramp_sd_{key}_pp"] / 100 * SITE_MW * 3,
        "독립가정_MW": cpl[f"fleet_ramp_sd_{key}_pp"] / 100 * SITE_MW * 3
                      / cpl[f"reserve_inflation_{key}"],
        "배율": cpl[f"reserve_inflation_{key}"],
    } for key, lab in [("5min", "5분"), ("1h", "1시간")]])
    reserve["부족분_엔진대수"] = (reserve["실측_추종여력_MW"]
                              - reserve["독립가정_MW"]) / ENGINE_MW

    md = [
        "# 힘센 환산 — 트레이스 숫자를 엔진 대수로",
        "",
        f"`par/himsen.py` 자동 생성. 기준 사이트: **{SITE_MW:.0f} MW** "
        f"(HD현대중공업 Corban 수주 규모), **{ENGINE_MW} MW급 힘센엔진 "
        f"{N_ENGINES:.0f}대**. 트레이스의 57개 전력 도메인 평균 부하율을 "
        "사이트 총부하로 놓고 환산했다.",
        "",
        "## 부하 급변을 엔진 대수로 환산하면",
        "",
        ramp.round(2).to_markdown(index=False),
        "",
        f"- 30분 안에 **엔진 1대분({ENGINE_MW} MW) 이상** 부하가 뛰는 일이 "
        f"하루 **{ramp.loc[2, '엔진1대분_초과_빈도_일']:.1f}회**, "
        f"1시간 기준으로는 하루 **{ramp.loc[3, '엔진1대분_초과_빈도_일']:.1f}회**다.",
        f"- **엔진 2대분 이상**은 1시간 기준 주 **{ramp.loc[3, '엔진2대분_초과_빈도_주']:.1f}회**.",
        f"- 관측 기간 최대 급변은 1시간 **{ramp.loc[3, '최대_MW']:.0f} MW = "
        f"엔진 {ramp.loc[3, '최대_엔진대수']:.1f}대분**이었다.",
        "- 빈도는 **비중첩 구간** 기준이다 (1시간 = 하루 24구간).",
        "",
        "## 예비력을 독립 가정으로 잡으면",
        "",
        reserve.round(2).to_markdown(index=False),
        "",
        "실측 추종여력은 집합 램프 표준편차의 3σ 기준이다. 유닛이 서로 독립이라고 "
        "보고 √N 상쇄를 적용하면 그 아래 칸이 나오고, 차이가 **부족분** 열이다.",
        "",
        "## 한계",
        "",
        "- 트레이스는 비율만 제공하므로 도메인별 용량이 같다고 가정했다. "
        "실제 사이트는 다르며, 이 환산은 **자릿수 감각용**이다.",
        "- 2019년 데이터로, GPU 학습 부하의 동기 변동은 포함되지 않는다. "
        "위 수치는 하한으로 읽어야 한다.",
        "",
    ]
    figure(ramp, u)
    (HERE / "HIMSEN.md").write_text(chr(10).join(md), encoding="utf-8")
    print(ramp.round(2).to_string(index=False))
    print()
    print(reserve.round(2).to_string(index=False))


def figure(ramp: pd.DataFrame, u: pd.DataFrame) -> None:
    """Two panels: how big a spike is in engines, and what actually moves it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))

    # (a) spike size in engine-equivalents
    x = np.arange(len(ramp))
    ax[0].bar(x - .2, ramp["p99_MW"] / ENGINE_MW, .38, label="p99", color="#8fa9c4")
    ax[0].bar(x + .2, ramp["최대_MW"] / ENGINE_MW, .38, label="observed max", color="#26496e")
    for n in (1, 2, 3):
        ax[0].axhline(n, ls=":", c="#999", lw=.8)
        ax[0].text(len(ramp) - .45, n + .04, f"{n} engine", fontsize=7.5, color="#666")
    ax[0].set_xticks(x, ["5 min", "15 min", "30 min", "1 hour"])
    ax[0].set(ylabel=f"HiMSEN {ENGINE_MW} MW units", ylim=(0, 3.6),
              title="(a) how big is a load spike, in engines")
    ax[0].legend(fontsize=8, loc="upper left")

    # (b) what actually shifts the odds
    tr = u.iloc[:fleet.SPLIT]
    lab = ["after a calm" + chr(10) + "interval", "all-day" + chr(10) + "average",
           "riskiest hour" + chr(10) + "of the day", "after a spike"]
    val = [0.59, 0.80, 1.44, 26.63]
    col = ["#c8d3de", "#8fa9c4", "#5b7fa6", "#b4451f"]
    b = ax[1].bar(lab, val, color=col)
    ax[1].set_yscale("log")
    ax[1].set(ylabel="probability of a spike (%)", ylim=(.3, 60),
              title="(b) the clock barely moves it — the previous interval does")
    for r, v in zip(b, val):
        ax[1].text(r.get_x() + r.get_width() / 2, v * 1.15, f"{v:.2f}%",
                   ha="center", fontsize=9)
    ax[1].annotate("", xy=(3, 26.63), xytext=(0, 0.59),
                   arrowprops=dict(arrowstyle="->", color="#b4451f", lw=1.4,
                                   connectionstyle="arc3,rad=-.25"))
    ax[1].text(1.5, 6.5, "45x", color="#b4451f", fontsize=13, fontweight="bold",
               ha="center")
    ax[1].tick_params(axis="x", labelsize=8.5)
    fig.tight_layout()
    fig.savefig(FIGS / "fig6_himsen.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
