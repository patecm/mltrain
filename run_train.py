from __future__ import annotations

import argparse
import getpass
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from mltrain.config_schema import validate_config
from mltrain.trainers import REGISTRY
from mltrain.utils import (
    append_run_index,
    build_meta,
    ensure_dir,
    now_timestamp,
    slugify,
    write_text,
)


def setup_logging(run_dir: Path):
    import logging

    log_dir = run_dir / "logs"
    ensure_dir(log_dir)

    logger = logging.getLogger("mltrain")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    fh = logging.FileHandler(log_dir / "train.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_scalar(s: str) -> Any:
    lowered = s.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        return s


def set_by_dotpath(d: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur: Any = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def render_run_name(template: str, model: str) -> str:
    ts = now_timestamp()
    user = slugify(getpass.getuser())
    return template.format(model=model, timestamp=ts, user=user)


def main() -> None:
    p = argparse.ArgumentParser(description="Standard ML training runner")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument(
        "--override",
        action="append",
        default=[],
        help='Override values like model.params.max_depth=8 (repeatable)',
    )
    args = p.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = load_yaml(cfg_path)

    for ov in args.override:
        if "=" not in ov:
            raise ValueError(f"Bad override (missing '='): {ov}")
        key, val = ov.split("=", 1)
        set_by_dotpath(cfg, key.strip(), parse_scalar(val.strip()))

    validate_config(cfg)

    model_name = cfg["run"]["model"]
    if model_name not in REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {sorted(REGISTRY)}")

    out_root = Path(cfg["run"]["output_dir"]).expanduser().resolve()
    ensure_dir(out_root)

    run_name = render_run_name(cfg["run"]["run_name_template"], model_name)
    run_dir = out_root / run_name
    ensure_dir(run_dir)

    logger = setup_logging(run_dir)
    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Model: {model_name}")

    # Freeze config + metadata
    write_text(run_dir / "config.yaml", yaml.safe_dump(cfg, sort_keys=False))
    write_text(run_dir / "meta.json", json.dumps(build_meta(str(cfg_path)), indent=2))

    trainer_fn = REGISTRY[model_name]
    metrics = trainer_fn(cfg, run_dir, logger)

    # Append to index.csv for quick experiment scanning
    index_row = {
        "run_name": run_name,
        "created_at": json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))["created_at"],
        "model": model_name,
        "output_dir": str(run_dir),
        "dataset_full": cfg.get("data", {}).get("datasets", {}).get("full", ""),
        "dataset_train": cfg.get("data", {}).get("datasets", {}).get("train", ""),
        "dataset_val": cfg.get("data", {}).get("datasets", {}).get("val", ""),
        "dataset_test": cfg.get("data", {}).get("datasets", {}).get("test", ""),
        "metrics_path": str(run_dir / "metrics.json"),
    }
    # include a couple common metric keys if present
    for k in ("val_auc", "val_logloss", "test_auc", "test_logloss"):
        if k in metrics:
            index_row[k] = metrics[k]

    append_run_index(out_root, index_row)

    logger.info("Done.")


if __name__ == "__main__":
    main()
