"""Decide which analysis branch the meter-fault study can support.

Answers three questions that the design hinges on, none of which can be
settled from the documentation alone:

  1. POLARITY  - does bad_*_data == True actually mean "bad"?  The PDF says
                 "When false, indicates low-confidence", which contradicts the
                 field name.  Getting this backwards inverts every label.
  2. EPISODES  - faults are bursty, so the effective sample size is the number
                 of contiguous fault runs, not the number of flagged rows.
  3. COMMON-MODE - if all PDUs fail together it is one telemetry-pipeline
                 outage, not 57 independent meter faults, and N collapses.

    python par/probe.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PARQUET = Path(__file__).parent / "data" / "power.parquet"
FLAGS = ["bad_measurement_data", "bad_production_power_data"]
STEP = pd.Timedelta("5min")


def episodes(df: pd.DataFrame, flag: str) -> pd.DataFrame:
    """Contiguous runs of flag==True within each PDU."""
    rows = []
    for (cell, pdu), g in df.groupby(["cell", "pdu"], sort=True):
        f = g[flag].to_numpy()
        # run boundaries: where the flag flips
        starts = np.flatnonzero(f & ~np.r_[False, f[:-1]])
        ends = np.flatnonzero(f & ~np.r_[f[1:], False])
        for s, e in zip(starts, ends):
            rows.append(
                dict(cell=cell, pdu=pdu, n_intervals=e - s + 1,
                     start=g["ts"].iloc[s], end=g["ts"].iloc[e])
            )
    return pd.DataFrame(rows)


def polarity_oracle(df: pd.DataFrame) -> None:
    """production_power_util is a COMPONENT of measured_power_util, so
    measured >= production must hold physically.  Violations mark bad data,
    independent of what the flags claim -- that makes them a ground truth we
    can test the flag polarity against."""
    viol = df["measured_power_util"] < df["production_power_util"]
    print(f"  physical violations (measured < production): {viol.sum():,} "
          f"({viol.mean():.3%} of rows)")
    if viol.sum() == 0:
        print("  -> oracle unusable; falling back to stuck-meter test only")
    for flag in FLAGS:
        for val in (True, False):
            sub = df[flag] == val
            rate = viol[sub].mean() if sub.any() else float("nan")
            print(f"  P(violation | {flag}=={val!s:5}) = {rate:.4%}   n={sub.sum():,}")

    # stuck meter: identical consecutive measured value, per PDU
    stuck = df.groupby(["cell", "pdu"])["measured_power_util"].diff().eq(0)
    print(f"  frozen readings (delta == 0): {stuck.sum():,} ({stuck.mean():.3%})")
    for flag in FLAGS:
        for val in (True, False):
            sub = df[flag] == val
            print(f"  P(frozen | {flag}=={val!s:5}) = {stuck[sub].mean():.4%}")


def main() -> None:
    df = pd.read_parquet(PARQUET)
    for f in FLAGS:
        df[f] = df[f].astype(bool)
    n_pdu = df.groupby(["cell", "pdu"]).ngroups
    print(f"{len(df):,} rows | {n_pdu} PDUs | {df['ts'].min()} .. {df['ts'].max()}\n")

    print("== 1. POLARITY ==")
    polarity_oracle(df)

    for flag in FLAGS:
        print(f"\n== 2. PREVALENCE & EPISODES: {flag} ==")
        print(f"  flagged rows: {df[flag].sum():,} ({df[flag].mean():.3%})")
        ep = episodes(df, flag)
        if ep.empty:
            print("  no episodes")
            continue
        print(f"  EPISODES: {len(ep)}  across {ep['pdu'].nunique()} PDUs")
        d = ep["n_intervals"] * 5
        print(f"  duration (min): median {d.median():.0f}  mean {d.mean():.0f}  "
              f"max {d.max():.0f}  |  >=6h: {(d >= 360).sum()}")
        print("  affected PDUs (top 10 by episode count):")
        print(ep.groupby(["cell", "pdu"]).size().sort_values(ascending=False)
              .head(10).to_string())

        print(f"\n== 3. COMMON-MODE: {flag} ==")
        per_ts = df.groupby("ts")[flag].sum()
        hit = per_ts[per_ts > 0]
        print(f"  timestamps with >=1 flagged PDU: {len(hit):,} / {len(per_ts):,}")
        if len(hit):
            print(f"  PDUs flagged simultaneously: median {hit.median():.0f}  "
                  f"max {hit.max():.0f} / {n_pdu}")
            print(f"  timestamps where ALL PDUs flagged: {(per_ts == n_pdu).sum():,}")
            # global events = contiguous runs where any PDU is flagged
            g = (per_ts > 0).to_numpy()
            n_global = int(np.sum(g & ~np.r_[False, g[:-1]]))
            print(f"  GLOBAL EVENTS (contiguous cluster-wide runs): {n_global}")


if __name__ == "__main__":
    main()
