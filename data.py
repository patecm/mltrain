from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class PartitioningSpec:
    style: str = "hive"      # "hive" expects year=YYYY/month=MM folders
    year_key: str = "year"
    month_key: str = "month"


def _parse_ymd(s: str) -> date:
    # expects YYYY-MM-DD
    return datetime.strptime(s, "%Y-%m-%d").date()


def _month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def _next_month(d: date) -> date:
    if d.month == 12:
        return date(d.year + 1, 1, 1)
    return date(d.year, d.month + 1, 1)


def iter_months_inclusive(start: date, end: date) -> Iterable[Tuple[int, int]]:
    cur = _month_start(start)
    end_m = _month_start(end)
    while cur <= end_m:
        yield (cur.year, cur.month)
        cur = _next_month(cur)


def month_partition_uri(base_uri: str, year: int, month: int, part: PartitioningSpec) -> str:
    """
    Hive style: <base>/year=YYYY/month=MM
    If your layout differs, change this function.
    """
    base = base_uri.rstrip("/")
    if part.style != "hive":
        raise ValueError(f"Unsupported partitioning style: {part.style}")
    return f"{base}/{part.year_key}={year}/{part.month_key}={month:02d}"


def read_parquet_any(uri: str, storage_options: Dict[str, Any]) -> pd.DataFrame:
    # pandas + pyarrow + s3fs handles s3:// URIs and partitioned dirs
    return pd.read_parquet(uri, storage_options=storage_options or {})


def read_partitioned_range(
    base_uri: str,
    start: date,
    end: date,
    storage_options: Dict[str, Any],
    part: PartitioningSpec,
) -> pd.DataFrame:
    """
    Reads only the year/month partitions between start and end (inclusive),
    concatenates, then returns the combined DF.
    """
    uris = [month_partition_uri(base_uri, y, m, part) for (y, m) in iter_months_inclusive(start, end)]
    dfs: List[pd.DataFrame] = []
    for u in uris:
        dfs.append(read_parquet_any(u, storage_options))
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def add_date_from_ymd(df: pd.DataFrame, ymd_cols: List[str], out_col: str = "__date") -> pd.DataFrame:
    y, m, d = ymd_cols
    # robust conversion
    yy = pd.to_numeric(df[y], errors="coerce").astype("Int64")
    mm = pd.to_numeric(df[m], errors="coerce").astype("Int64")
    dd = pd.to_numeric(df[d], errors="coerce").astype("Int64")
    # build datetime; invalid rows become NaT
    dt = pd.to_datetime(
        {"year": yy.astype("float"), "month": mm.astype("float"), "day": dd.astype("float")},
        errors="coerce",
        utc=True,
    )
    df = df.copy()
    df[out_col] = dt.dt.date
    return df


def filter_date_range(df: pd.DataFrame, date_col: str, start: date, end: date) -> pd.DataFrame:
    if df.empty:
        return df
    mask = df[date_col].notna() & (df[date_col] >= start) & (df[date_col] <= end)
    return df.loc[mask].copy()


def build_splits(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    storage_options = cfg.get("data", {}).get("s3", {}).get("storage_options", {}) or {}
    ds = cfg["data"]["datasets"]

    # Explicit paths mode
    time_split = cfg.get("data", {}).get("time_split", {}).get("enabled", False)
    if not time_split and any(k in ds for k in ("train", "val", "test")):
        train = read_parquet_any(ds["train"], storage_options)
        val = read_parquet_any(ds["val"], storage_options) if "val" in ds else None
        test = read_parquet_any(ds["test"], storage_options) if "test" in ds else None
        return train, val, test

    # Time split mode (partition-aware)
    if not time_split:
        # fallback: read full if provided
        if "full" in ds:
            full = read_parquet_any(ds["full"], storage_options)
            return full, None, None
        raise ValueError("Provide explicit train/val/test OR enable time_split with datasets.full")

    base = ds["full"]
    part_cfg = cfg.get("data", {}).get("partitioning", {}) or {}
    part = PartitioningSpec(
        style=part_cfg.get("style", "hive"),
        year_key=part_cfg.get("year_key", "year"),
        month_key=part_cfg.get("month_key", "month"),
    )

    ymd_cols = cfg["data"]["ymd_cols"]

    ts_cfg = cfg["data"]["time_split"]
    tr_s, tr_e = _parse_ymd(ts_cfg["train"]["start"]), _parse_ymd(ts_cfg["train"]["end"])
    va_s, va_e = _parse_ymd(ts_cfg["val"]["start"]), _parse_ymd(ts_cfg["val"]["end"])
    te_s, te_e = _parse_ymd(ts_cfg["test"]["start"]), _parse_ymd(ts_cfg["test"]["end"])

    # Read only the months needed for each split
    train_df = read_partitioned_range(base, tr_s, tr_e, storage_options, part)
    val_df = read_partitioned_range(base, va_s, va_e, storage_options, part)
    test_df = read_partitioned_range(base, te_s, te_e, storage_options, part)

    # Filter precisely to day boundaries using the columns in the data
    train_df = filter_date_range(add_date_from_ymd(train_df, ymd_cols), "__date", tr_s, tr_e)
    val_df = filter_date_range(add_date_from_ymd(val_df, ymd_cols), "__date", va_s, va_e)
    test_df = filter_date_range(add_date_from_ymd(test_df, ymd_cols), "__date", te_s, te_e)

    return train_df, val_df, test_df


def prepare_xy(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: Optional[List[str]] = None,
    id_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    drop_cols = set(drop_cols or [])
    id_cols = set(id_cols or [])

    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in dataframe columns")

    y = df[target_col]
    X = df.drop(columns=[target_col], errors="ignore")

    # Drop helper column if present
    if "__date" in X.columns:
        X = X.drop(columns=["__date"])

    to_drop = list(drop_cols | id_cols)
    if to_drop:
        X = X.drop(columns=[c for c in to_drop if c in X.columns], errors="ignore")

    return X, y
