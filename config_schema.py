from __future__ import annotations

from typing import Any, Dict


def _get(cfg: Dict[str, Any], path: str):
    cur: Any = cfg
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def validate_config(cfg: Dict[str, Any]) -> None:
    # Core required
    required = [
        ("run", dict),
        ("run.model", str),
        ("run.output_dir", str),
        ("run.run_name_template", str),

        ("data", dict),
        ("data.datasets", dict),
        ("data.target_col", str),

        ("model", dict),
        ("model.params", dict),
    ]
    for key, t in required:
        v = _get(cfg, key)
        if v is None:
            raise ValueError(f"Missing required config key: {key}")
        if not isinstance(v, t):
            raise ValueError(f"Config key '{key}' must be {t.__name__}, got {type(v).__name__}")

    ds = cfg["data"]["datasets"]
    time_split = cfg.get("data", {}).get("time_split", {}).get("enabled", False)

    has_explicit = any(k in ds for k in ("train", "val", "test"))
    has_full = "full" in ds

    if time_split:
        if not has_full:
            raise ValueError("data.time_split.enabled=true requires data.datasets.full")
        # Ensure time split blocks
        for split in ("train", "val", "test"):
            s = cfg["data"]["time_split"].get(split)
            if not s or "start" not in s or "end" not in s:
                raise ValueError(f"data.time_split.{split} must have start and end")
        # Ensure ymd_cols exist (or you can extend to a date_col later)
        ymd = cfg["data"].get("ymd_cols")
        if not (isinstance(ymd, list) and len(ymd) == 3):
            raise ValueError("For time_split, set data.ymd_cols: [year, month, day]")
    else:
        # Non time_split: allow either explicit splits or full (then you can random split in trainer if you want)
        if not (has_explicit or has_full):
            raise ValueError("Provide data.datasets.full OR explicit train/val/test paths.")
