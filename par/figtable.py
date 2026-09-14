"""데이터 분석 결과를 한 장짜리 시각 표로 렌더링한다.

숫자는 전부 fleet.py / tod.py / himsen.py 가 이미 측정한 값이고, 여기서는
계산하지 않는다. 이 파일이 하는 일은 배치와 색뿐이다.

    python par/figtable.py     # -> figures/fig7_results_table.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

FIGS = Path(__file__).parent / "figures"
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

INK, MUTE, BAND = "#14201F", "#6E7C7A", "#F3F6F6"
TEAL, RUST, SLATE, AMBER = "#186D6D", "#B4451F", "#5B7FA6", "#B9822A"

# (지표, 값, 막대비율 0~1 또는 None, 막대색, 해석)
SECTIONS = [
    ("A", "부하는 얼마나 튀는가", SLATE,
     "1,000 MW 사이트 = 9.6 MW급 힘센 104대로 환산", [
        ("1시간 내 엔진 1대분(9.6MW) 이상 급변", "하루 2.1 회", None, None,
         "2대분 이상은 주 1.1회"),
        ("관측 최대 급변 (1시간)", "30.1 MW = 3.1 대", 3.1 / 4, SLATE,
         "보통 날도 17.9 MW = 1.9대"),
        ("하루 총 변동 이동거리", "376 MW / 일 = 39 대분", None, None,
         "피크 차이의 15배를 매일 오르내린다"),
        ("집합 상위 1% 상승 시 동반 상승 유닛", "72.3 %", 0.723, SLATE,
         "평시 43.4% — 상쇄를 기대할 수 없다"),
     ]),
    ("B", "시계(시간대)로 설명되는가", RUST,
     "피처 1개(하루 중 시각), 타깃 1개(급변 0/1) · 검증 구간 실측", [
        ("부하 레벨을 설명하는 정도", "R² 0.69", 0.69, TEAL,
         "일주기가 뚜렷하다 — 편성표가 통하는 영역"),
        ("급변을 설명하는 정도", "▼ 2.3 %", 0.02, RUST,
         "음수 = 그냥 평균 확률을 쓰는 것보다 나빴다"),
        ("급변 판별력 (AUC)", "0.55", 0.55, RUST,
         "0.5 = 동전 던지기, 1.0 = 완벽"),
        ("해상도·요일까지 더하면", "학습 18.4%  →  검증 ▼13.2%", None, None,
         "달력을 잘게 쪼갤수록 나빠진다 (과적합)"),
     ]),
    ("C", "그럼 무엇이 급변을 지배하는가", TEAL,
     "같은 검증 구간, 조건별 다음 구간 급변 확률", [
        ("직전 구간이 평온했을 때", "0.59 %", 0.0222, MUTE, "기준"),
        ("가장 위험한 시간대 (18시)", "1.44 %", 0.0541, AMBER, "기준의 2.4배에 그친다"),
        ("직전 구간에 급변이 있었을 때", "26.6 %", 1.0, RUST, "기준의  45 배"),
        ("전체 피처를 쓴 판별력 (AUC)", "0.92", 0.92, TEAL, "시계만 쓰면 0.55"),
        ("1시간 전 경보 정밀도", "28.2 %", 0.282, TEAL,
         "무작위 0.8% 대비  35 배"),
     ]),
    ("D", "함대를 묶으면 무엇이 되는가", TEAL,
     "57개 유닛 패널 · 학습 1~20일 / 검증 21~31일", [
        ("유효 독립 유닛 수", "11.3 / 57 대", 11.3 / 57, RUST,
         "N대를 붙여도 N대만큼 독립적이지 않다"),
        ("필요 예비력 (독립 가정 대비)", "2.6 배  (1시간 3.0배)", None, None,
         "1시간 기준 힘센 1.6대분이 비어 있다"),
        ("이웃 유닛으로 계측 복원", "R² 0.955  ·  54 / 57 대", 0.955, TEAL,
         "실패한 3대는 전부 동종 동료가 없던 개체"),
        ("주입 고장 검출 — 동결", "97.6 %", 0.976, TEAL,
         "오경보를 대당 하루 1건으로 묶은 상태"),
        ("주입 고장 검출 — 2%p 편향", "92.6 %", 0.926, TEAL,
         "1%p는 60.9% — 실용 하한이 약 1~2%p"),
        ("지문만으로 실제 구획 재현", "ARI 0.710", 0.710, TEAL,
         "무작위 배정 상한 0.048 — 우연의 약 14배"),
     ]),
]

ROW_H, SEC_H, PAD = 0.62, 0.86, 0.34
X_LABEL, X_VALUE, X_BAR, X_BARW, X_NOTE, X_END = 0.55, 7.6, 12.0, 3.2, 15.7, 23.0


def main() -> None:
    n_rows = sum(len(s[4]) for s in SECTIONS)
    height = 2.2 + len(SECTIONS) * (SEC_H + PAD) + n_rows * ROW_H + 0.9
    fig, ax = plt.subplots(figsize=(X_END / 1.72, height / 1.72))
    ax.set_xlim(0, X_END)
    ax.set_ylim(0, height)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    y = height - 0.35

    # ---- 제목 띠
    ax.add_patch(Rectangle((0, y - 1.15), X_END, 1.15, color=INK, zorder=1))
    ax.text(X_LABEL, y - 0.42, "데이터 분석 결과", color="white", fontsize=17,
            fontweight="bold", va="center", zorder=2)
    ax.text(X_LABEL, y - 0.88, "Google powerdata_2019 공개 트레이스 · 57개 전력 도메인 × 2019년 5월 × "
            "5분 간격 · 508,896행 · 결측 없음",
            color="#AFC3C2", fontsize=8.6, va="center", zorder=2)
    ax.text(X_END - 0.55, y - 0.62, "학습 1~20일  /  검증 21~31일\n검증 구간 재학습 없음",
            color="#AFC3C2", fontsize=8.4, va="center", ha="right", zorder=2,
            linespacing=1.5)
    y -= 1.15 + 0.55

    for tag, title, tone, sub, rows in SECTIONS:
        # ---- 섹션 헤더
        ax.add_patch(FancyBboxPatch((X_LABEL - 0.02, y - SEC_H + 0.16), 0.62, 0.5,
                                    boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor=tone, edgecolor="none", zorder=2))
        ax.text(X_LABEL + 0.29, y - SEC_H + 0.41, tag, color="white", fontsize=11,
                fontweight="bold", ha="center", va="center", zorder=3)
        ax.text(X_LABEL + 0.86, y - SEC_H + 0.47, title, color=INK, fontsize=12.6,
                fontweight="bold", va="center")
        ax.text(X_LABEL + 0.86, y - SEC_H + 0.08, sub, color=MUTE, fontsize=8.4,
                va="center")
        ax.plot([X_LABEL, X_END - 0.55], [y - SEC_H - 0.02] * 2, color=INK, lw=1.3)
        y -= SEC_H + 0.06

        for i, (label, value, frac, barcol, note) in enumerate(rows):
            if i % 2 == 0:
                ax.add_patch(Rectangle((X_LABEL - 0.18, y - ROW_H + 0.06),
                                       X_END - X_LABEL - 0.37, ROW_H - 0.06,
                                       color=BAND, zorder=0))
            cy = y - ROW_H / 2
            ax.text(X_LABEL, cy, label, color=INK, fontsize=10, va="center")
            ax.text(X_VALUE, cy, value, color=barcol or INK, fontsize=11.4,
                    fontweight="bold", va="center")
            if frac is not None:
                ax.add_patch(Rectangle((X_BAR, cy - 0.115), X_BARW, 0.23,
                                       color="#DFE6E5", zorder=1))
                ax.add_patch(Rectangle((X_BAR, cy - 0.115), X_BARW * max(frac, 0.006),
                                       0.23, color=barcol, zorder=2))
            ax.text(X_NOTE, cy, note, color=MUTE, fontsize=9.2, va="center")
            ax.plot([X_LABEL - 0.18, X_END - 0.55], [y - ROW_H + 0.03] * 2,
                    color="#DCE3E2", lw=0.7, zorder=3)
            y -= ROW_H
        y -= PAD

    ax.plot([X_LABEL, X_END - 0.55], [y + 0.18] * 2, color=INK, lw=1.3)
    ax.text(X_LABEL, y - 0.13,
            "모든 수치는 학습에 한 번도 쓰이지 않은 검증 구간(21~31일)에서 측정. "
            "미래 정보가 피처로 새지 않는지 확인하는 테스트 6종 통과.",
            color=MUTE, fontsize=8.2, va="center")
    ax.text(X_LABEL, y - 0.45,
            "MW 환산은 도메인별 용량 동일 가정에 따른 자릿수 감각용.   "
            "2019년 데이터로 GPU 학습 부하 이전 — 모든 수치는 하한.",
            color=MUTE, fontsize=8.2, va="center")

    fig.tight_layout(pad=0.2)
    FIGS.mkdir(exist_ok=True)
    out = FIGS / "fig7_results_table.png"
    fig.savefig(out, dpi=190, facecolor="white")
    plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    main()
