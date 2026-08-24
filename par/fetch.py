"""Download the Google powerdata_2019 traces from the public GCS bucket.

No auth, no gcloud SDK, no billing: the bucket is world-readable over plain
HTTPS. 57 PDU files + 1 machine->PDU mapping, ~3.3 MB total.

    python par/fetch.py            # download + build par/data/power.parquet
    python par/fetch.py --force    # re-download even if cached
"""
from __future__ import annotations

import gzip
import io
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

BUCKET = "powerdata_2019"
LIST_URL = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?fields=items(name,size),nextPageToken"
OBJ_URL = f"https://storage.googleapis.com/{BUCKET}/{{name}}"

RAW_DIR = Path(__file__).parent / "data" / "raw"
PARQUET = Path(__file__).parent / "data" / "power.parquet"

# Trace starts 2019-05-01 00:00 PT; `time` counts microseconds from 600s BEFORE
# that instant (see power_trace_documentation.pdf).
TRACE_EPOCH_OFFSET_US = 600 * 1_000_000


def _get(url: str, timeout: int = 60) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def list_objects() -> list[tuple[str, int]]:
    """Return [(name, size_bytes)] for every object in the bucket."""
    out, url = [], LIST_URL
    while url:
        page = json.loads(_get(url))
        out += [(o["name"], int(o["size"])) for o in page.get("items", [])]
        token = page.get("nextPageToken")
        url = f"{LIST_URL}&pageToken={token}" if token else None
    return sorted(out)


def download(name: str, size: int, force: bool = False) -> Path:
    dest = RAW_DIR / name
    if not force and dest.exists() and dest.stat().st_size == size:
        return dest
    dest.write_bytes(_get(OBJ_URL.format(name=name)))
    if dest.stat().st_size != size:  # truncated transfer -> fail loudly
        raise IOError(f"{name}: expected {size} bytes, got {dest.stat().st_size}")
    return dest


def read_pdu(path: Path) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(gzip.decompress(path.read_bytes())))
    # Rows arrive unsorted; every downstream lag/rolling feature depends on order.
    return df.sort_values("time", ignore_index=True)


def main(force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    objects = list_objects()
    print(f"{len(objects)} objects, {sum(s for _, s in objects) / 1e6:.2f} MB")

    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(lambda o: download(*o, force=force), objects))

    pdu_paths = [p for p in paths if p.name != "machine_to_pdu_mapping.csv.gz"]
    df = pd.concat([read_pdu(p) for p in pdu_paths], ignore_index=True)

    # Absolute wall-clock, so time-of-day / day-of-week features are meaningful.
    df["ts"] = pd.Timestamp("2019-05-01 00:00", tz="US/Pacific") + pd.to_timedelta(
        df["time"] - TRACE_EPOCH_OFFSET_US, unit="us"
    )
    df = df.sort_values(["cell", "pdu", "ts"], ignore_index=True)
    df.to_parquet(PARQUET, index=False)

    print(f"{len(df):,} rows x {df['pdu'].nunique()} PDUs -> {PARQUET}")
    print(f"span {df['ts'].min()} .. {df['ts'].max()}")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
