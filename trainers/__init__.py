from __future__ import annotations

from typing import Callable, Dict
from pathlib import Path

TrainerFn = Callable[[dict, Path, object], dict]

from .xgboost_training import train as xgboost_train  # noqa: E402

REGISTRY: Dict[str, TrainerFn] = {
    "xgboost": xgboost_train,
}
