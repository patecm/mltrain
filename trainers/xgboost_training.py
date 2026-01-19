from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd
from xgboost import XGBClassifier


def _extract_xgb_history(model: XGBClassifier) -> Optional[pd.DataFrame]:
    try:
        res = model.evals_result()
    except Exception:
        return None
    rows = []
    for dataset_name, metrics in res.items():
        for metric_name, values in metrics.items():
            for i, v in enumerate(values):
                rows.append({"iter": i, "dataset": dataset_name, "metric": metric_name, "value": v})
    return pd.DataFrame(rows) if rows else None


def train(cfg: Dict[str, Any], run_dir: Path, logger) -> Dict[str, Any]:
    from mltrain.data import build_splits, prepare_xy
    from mltrain.utils import write_text

    train_df, val_df, test_df = build_splits(cfg)

    target_col = cfg["data"]["target_col"]
    drop_cols = cfg["data"].get("drop_cols", [])
    id_cols = cfg["data"].get("id_cols", [])

    X_train, y_train = prepare_xy(train_df, target_col, drop_cols, id_cols)
    X_val, y_val = (None, None)
    X_test, y_test = (None, None)

    if val_df is not None and len(val_df) > 0:
        X_val, y_val = prepare_xy(val_df, target_col, drop_cols, id_cols)
    if test_df is not None and len(test_df) > 0:
        X_test, y_test = prepare_xy(test_df, target_col, drop_cols, id_cols)

    # Save features + dtypes for tracking
    features_payload = {
        "features": list(X_train.columns),
        "dtypes": {c: str(t) for c, t in X_train.dtypes.items()},
        "target_col": target_col,
        "drop_cols": list(drop_cols),
        "id_cols": list(id_cols),
        "n_features": int(X_train.shape[1]),
    }
    write_text(run_dir / "features.json", json.dumps(features_payload, indent=2))

    params = cfg["model"]["params"].copy()
    model = XGBClassifier(**params)

    eval_metrics = cfg.get("training", {}).get("eval_metrics", None)  # None | str | list
    log_every_n = int(cfg.get("training", {}).get("log_every_n", 25))
    early_stopping = cfg.get("training", {}).get("early_stopping_rounds", None)

    eval_set = []
    if X_val is not None:
        eval_set.append((X_val, y_val))

    logger.info(f"Train: rows={len(X_train)} features={X_train.shape[1]}")
    if X_val is not None:
        logger.info(f"Val:   rows={len(X_val)}")
    if X_test is not None:
        logger.info(f"Test:  rows={len(X_test)}")

    # For sklearn wrapper, "verbose" can be an int: print eval every N rounds.
    # We'll keep console clean and log later from evals_result() into history.csv,
    # but verbose can still be helpful. Set to log_every_n if eval_set present.
    verbose = log_every_n if eval_set else False

    fit_kwargs: Dict[str, Any] = {
        "eval_metric": eval_metrics,
        "verbose": verbose,
    }
    if eval_set:
        fit_kwargs["eval_set"] = eval_set
    if early_stopping is not None and eval_set:
        fit_kwargs["early_stopping_rounds"] = int(early_stopping)

    model.fit(X_train, y_train, **fit_kwargs)

    # Save model
    model_path = run_dir / "model.joblib"
    joblib.dump(model, model_path)

    # Save per-iteration history (loss/metrics)
    history = _extract_xgb_history(model)
    if history is not None:
        history_path = run_dir / "history.csv"
        history.to_csv(history_path, index=False)
        logger.info(f"Wrote history: {history_path}")

        # Also log last values per metric for quick scan
        last = (
            history.sort_values(["dataset", "metric", "iter"])
                   .groupby(["dataset", "metric"], as_index=False)
                   .tail(1)
        )
        for _, r in last.iterrows():
            logger.info(f"Final {r['dataset']} {r['metric']}: {r['value']}")

    # Optional final metrics on val/test (generic probability-based metrics if feasible)
    final: Dict[str, Any] = {"model_path": str(model_path)}
    try:
        from sklearn.metrics import roc_auc_score, log_loss

        if X_val is not None and hasattr(model, "predict_proba"):
            p = model.predict_proba(X_val)[:, 1]
            final["val_auc"] = float(roc_auc_score(y_val, p))
            final["val_logloss"] = float(log_loss(y_val, p))
        if X_test is not None and hasattr(model, "predict_proba"):
            p = model.predict_proba(X_test)[:, 1]
            final["test_auc"] = float(roc_auc_score(y_test, p))
            final["test_logloss"] = float(log_loss(y_test, p))
    except Exception as e:
        logger.info(f"Skipped sklearn final metrics: {e}")

    write_text(run_dir / "metrics.json", json.dumps(final, indent=2))
    return final
