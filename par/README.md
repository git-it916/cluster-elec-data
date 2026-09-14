# par — Power-at-Risk

Google `powerdata_2019` 트레이스에 금융 VaR 방법론을 이식해 데이터센터 전력
리스크를 정량화한다. 설계 근거와 전체 스펙은 **[SPEC.md](SPEC.md)**.

> 이 디렉터리는 `google/cluster-data` 업스트림과 무관한 자체 작업물이다.
> 루트의 문서·bibliography는 건드리지 않는다.

## 빠른 시작

```bash
python par/fetch.py          # 3.25MB 다운로드 → par/data/power.parquet
python par/probe.py          # 플래그 진단 (폴라리티·에피소드 구조)
python par/test_par.py       # 12개 검증 (백테스트 기지값 + 인과성)
python par/par.py            # A층: 모델 3종 백테스트          (~30초)
python par/observability.py  # C'층: 57 PDU 리스크 테이블       (~15초)
python par/report.py         # 그림 2종 + RESULTS.md          (~100초)
python par/fleet.py          # 함대층: 다중 유닛·다중 센서 ML   (~70초)
python par/test_fleet.py     # 6개 검증 (누출·결합·주입 항등식)
```

**인증·과금·gcloud SDK 불필요.** 버킷 `powerdata_2019`가 평문 HTTPS로
공개되어 있어 `urllib`로 직접 받는다. 의존성은 `pandas` + `pyarrow`뿐.

BigQuery는 Phase 2(워크로드 조인)에만 필요하다.

## 데이터

508,896행 · 57 PDU · 2019-05-01 ~ 05-31 US/Pacific · 5분 간격 · 결측 없음.
값은 전부 용량 대비 **비율** [0,1]이다 (와트·전압·전류 없음).

## 알아둘 것 두 가지

**1. 플래그 폴라리티 — 공식 PDF가 틀렸다.**
문서는 `bad_measurement_data`에 대해 *"When false, indicates low-confidence"*
라고 쓰지만, `True`인 행이 508,896개 중 24개뿐이다. 문자 그대로 읽으면
99.995%가 저신뢰도라는 뜻이 되어 성립하지 않는다. **`True` = bad**로 다룬다.
근거는 [SPEC.md §2.1](SPEC.md).

**2. 계측기 고장 예지정비는 이 데이터로 불가능하다.**
`bad_measurement_data` 24건이 전부 **단일 5분 구간**이다. 연속 플래그가 한 번도
없어 열화 과정이 존재하지 않는다. `bad_production_power_data`(22.9%)는 하드웨어
고장이 아니라 CPU 보간 추정 모델의 무효 구간이며, PDU별로 거의 고정된 속성이다.

사전 확정한 결정 규칙에 따른 판정이며, 이 음성 결과 자체가 산출물이다.
자세한 근거는 [SPEC.md §3](SPEC.md).

## 스코프

| 층 | 질문 | 데이터 | 상태 |
|---|---|---|---|
| **A. PaR** | 용량 한계까지 여유가 얼마나 | 로컬 | 즉시 가능 |
| **C′. 관측 리스크** | 그 숫자를 믿을 수 있나 | 로컬 | 즉시 가능 |
| **B. Component VaR** | 어떤 부하가 리스크를 만드나 | BigQuery | Phase 2 |
| **F. 함대** | 여러 대를 묶으면 ML이 더 잘하나 | 로컬 | 즉시 가능 |

## 파일

| 파일 | 역할 |
|---|---|
| [SPEC.md](SPEC.md) | 설계 스펙 — 모델, horizon, 백테스트, 검정력, 한계 |
| [RESULTS.md](RESULTS.md) | 실측 결과표 + 그림 (자동 생성) |
| [fetch.py](fetch.py) | 공개 GCS → `data/power.parquet` |
| [probe.py](probe.py) | 폴라리티·에피소드·공통모드 진단 |
| [backtest.py](backtest.py) | Kupiec / Christoffersen / Basel / Acerbi–Székely |
| [par.py](par.py) | A층 — PaR·CVaR 예측 모델 3종, 다중 horizon |
| [observability.py](observability.py) | C′층 — 관측 점수 → 마진 가산 → 리스크 테이블 |
| [report.py](report.py) | 그림 + RESULTS.md 생성 |
| [test_par.py](test_par.py) | 12개 검증 — 통계 기지값과 인과성 |
| [FLEET.md](FLEET.md) | F층 결과표 — 다중 유닛·다중 센서 ML (자동 생성) |
| [fleet.py](fleet.py) | F층 — 결합·예측·램프·가상센서·군집화 |
| [test_fleet.py](test_fleet.py) | 6개 검증 — 교차 유닛 피처 누출 방지 |

## 핵심 결과

| 발견 | 값 |
|---|---|
| 최적 모델 | `fhs_ar` — 조건부 커버리지 20/57 PDU 통과 (α=0.99) |
| 경험 분위수 실패 | breach 직후 재breach 확률 76.1% (명목 1%) |
| 관측 리스크 비용 | 용량의 4.3%가 귀속 불신만으로 사장 |
| 경보 시간의 대가 | 6시간 경보 시 한도 38.4% → 25.1% |

전체는 [RESULTS.md](RESULTS.md), 해석은 [SPEC.md §10](SPEC.md).

## 함대층 핵심 결과

| 발견 | 값 |
|---|---|
| 유효 독립 유닛 수 | 57대가 **11.3대**처럼 움직인다 (5분 증분 기준) |
| 필요 부하추종 여력 | 독립 가정 대비 **2.6배** (1시간 램프는 3.0배) |
| 교차 유닛 피처의 순이득 | GBM MAE −3.8~−6.0%, 57대 중 45~54대 개선 |
| 단, 선형 모델은 못 쓴다 | `ridge_fleet`는 `ridge_solo`를 이기지 못한다 — 신호가 비선형 |
| 램프 사전 탐지 | ROC-AUC 0.83~0.89, 상위 알람 정밀도 22~33% (기저율 0.7%) |
| 가상센서 복원 | R² 중앙값 0.955, 54/57대 유효 |
| 주입 고장 탐지 | stuck 98%, bias 2%p 93% (오경보 1건/대·일) |
| 무라벨 군집화 | ARI 0.710 vs 실제 사이트 (무작위 상한 0.048) |

전체는 [FLEET.md](FLEET.md).
