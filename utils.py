from __future__ import annotations

import csv
import getpass
import json
import os
import platform
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def build_meta(config_path: str) -> Dict[str, Any]:
    return {
        "config_path": config_path,
        "created_at": datetime.now().isoformat(),
        "user": getpass.getuser(),
        "cwd": os.getcwd(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_sha": git_sha(),
        "hostname": platform.node(),
    }


def append_run_index(out_root: Path, row: Dict[str, Any]) -> None:
    """
    Appends a single run row to runs/index.csv (creates if missing).
    Keeps it simple: stringifies values.
    """
    path = out_root / "index.csv"
    ensure_dir(path.parent)

    # Stable-ish column order
    keys = list(row.keys())
    if path.exists():
        # If exists, preserve existing header
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
        if header:
            keys = header

    # Write
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if write_header:
            writer.writeheader()
        # ensure all keys present
        safe = {k: str(row.get(k, "")) for k in keys}
        writer.writerow(safe)
