from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


@dataclass
class ReportConfig:
    ks: Tuple[int, ...] = (50, 100, 200)
    # If you care more about recall, use beta>1 (e.g., 2.0)
    fbeta: float = 1.0
    # If provided, compute metrics at this threshold too
    fixed_threshold: Optional[float] = None
    calibration_bins: int = 10


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def precision_recall_fbeta_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float, beta: float) -> Dict[str, float]:
    y_pred = (y_prob >= thr).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    beta2 = beta * beta
    fbeta = _safe_div((1 + beta2) * precision * recall, (beta2 * precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "threshold": float(thr),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        f"f{beta:g}": float(fbeta),
    }


def best_threshold_by_fbeta(y_true: np.ndarray, y_prob: np.ndarray, beta: float) -> Dict[str, float]:
    p, r, thr = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns thresholds of length n-1; align
    thr = np.append(thr, 1.0)

    beta2 = beta * beta
    denom = (beta2 * p + r)
    f = np.where(denom > 0, (1 + beta2) * p * r / denom, 0.0)

    idx = int(np.nanargmax(f))
    best_thr = float(thr[idx])
    out = precision_recall_fbeta_at_threshold(y_true, y_prob, best_thr, beta)
    out[f"f{beta:g}"] = float(f[idx])
    out["chosen_by"] = f"max_f{beta:g}"
    return out


def topk_metrics(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> Dict[str, float]:
    n = len(y_true)
    k = int(min(k, n))
    order = np.argsort(-y_prob)
    top = y_true[order[:k]]

    tp_topk = int(top.sum())
    total_pos = int(y_true.sum())
    precision_k = _safe_div(tp_topk, k)
    recall_k = _safe_div(tp_topk, total_pos)

    base_rate = _safe_div(total_pos, n)
    lift_k = _safe_div(precision_k, base_rate) if base_rate > 0 else 0.0

    return {
        "k": k,
        "precision@k": float(precision_k),
        "recall@k": float(recall_k),
        "lift@k": float(lift_k),
        "tp_in_topk": tp_topk,
        "total_pos": total_pos,
        "n": n,
        "base_rate": float(base_rate),
    }


def compute_metrics_bundle(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cfg: ReportConfig,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    out: Dict[str, Any] = {}

    # “Global” metrics
    out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    out["logloss"] = float(log_loss(y_true, y_prob))
    out["brier"] = float(brier_score_loss(y_true, y_prob))

    # Thresholded metrics
    out["best_threshold"] = best_threshold_by_fbeta(y_true, y_prob, cfg.fbeta)
    if cfg.fixed_threshold is not None:
        out["fixed_threshold"] = precision_recall_fbeta_at_threshold(y_true, y_prob, cfg.fixed_threshold, cfg.fbeta)

    # Top-K metrics
    out["topk"] = [topk_metrics(y_true, y_prob, k) for k in cfg.ks]

    return out


def make_plotly_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metrics: Dict[str, Any],
    title: str = "Model Performance Report",
    cfg: ReportConfig = ReportConfig(),
) -> go.Figure:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    # Curves
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=cfg.calibration_bins, strategy="quantile")

    # Score distributions
    pos_scores = y_prob[y_true == 1]
    neg_scores = y_prob[y_true == 0]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Precision–Recall", "ROC", "Calibration", "Score distribution"),
        horizontal_spacing=0.12, vertical_spacing=0.15,
    )

    # PR
    fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="PR curve"), row=1, col=1)
    fig.update_xaxes(title_text="Recall", row=1, col=1)
    fig.update_yaxes(title_text="Precision", row=1, col=1)

    # ROC
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC curve"), row=1, col=2)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")), row=1, col=2)
    fig.update_xaxes(title_text="False Positive Rate", row=1, col=2)
    fig.update_yaxes(title_text="True Positive Rate", row=1, col=2)

    # Calibration
    fig.add_trace(go.Scatter(x=mean_pred, y=frac_pos, mode="lines+markers", name="Calibration"), row=2, col=1)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line=dict(dash="dash")), row=2, col=1)
    fig.update_xaxes(title_text="Mean predicted probability", row=2, col=1)
    fig.update_yaxes(title_text="Fraction of positives", row=2, col=1)

    # Distributions (use histograms with transparency)
    fig.add_trace(go.Histogram(x=neg_scores, name="Negative", nbinsx=50, opacity=0.6), row=2, col=2)
    fig.add_trace(go.Histogram(x=pos_scores, name="Positive", nbinsx=50, opacity=0.6), row=2, col=2)
    fig.update_xaxes(title_text="Predicted probability", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    fig.update_layout(barmode="overlay")

    # Put key metrics in a compact annotation block
    bt = metrics.get("best_threshold", {})
    lines = [
        f"PR-AUC: {metrics.get('pr_auc', float('nan')):.4f}",
        f"ROC-AUC: {metrics.get('roc_auc', float('nan')):.4f}",
        f"LogLoss: {metrics.get('logloss', float('nan')):.4f}",
        f"Brier: {metrics.get('brier', float('nan')):.4f}",
        "",
        f"Best thr (by F{cfg.fbeta:g}): {bt.get('threshold', float('nan')):.4f}",
        f"Precision: {bt.get('precision', float('nan')):.4f}",
        f"Recall: {bt.get('recall', float('nan')):.4f}",
        f"F{cfg.fbeta:g}: {bt.get(f'f{cfg.fbeta:g}', float('nan')):.4f}",
    ]

    fig.update_layout(
        title=title,
        height=800,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80),
        annotations=[
            dict(
                text="<br>".join(lines),
                x=1.02, y=0.5, xref="paper", yref="paper",
                showarrow=False, align="left",
                bordercolor="rgba(0,0,0,0.2)", borderwidth=1,
            )
        ],
    )
    return fig


def write_report_artifacts(
    run_dir: Path,
    split_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cfg: ReportConfig = ReportConfig(),
) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    out_dir = run_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_metrics_bundle(y_true, y_prob, cfg)

    # Save metrics json + a flat CSV table
    (out_dir / f"{split_name}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    flat_rows = []
    flat_rows.append({"metric": "pr_auc", "value": metrics["pr_auc"]})
    flat_rows.append({"metric": "roc_auc", "value": metrics["roc_auc"]})
    flat_rows.append({"metric": "logloss", "value": metrics["logloss"]})
    flat_rows.append({"metric": "brier", "value": metrics["brier"]})

    bt = metrics["best_threshold"]
    flat_rows.append({"metric": "best_threshold", "value": bt["threshold"]})
    flat_rows.append({"metric": "precision_at_best", "value": bt["precision"]})
    flat_rows.append({"metric": "recall_at_best", "value": bt["recall"]})
    flat_rows.append({"metric": f"f{cfg.fbeta:g}_at_best", "value": bt[f"f{cfg.fbeta:g}"]})

    for row in metrics["topk"]:
        k = row["k"]
        flat_rows.append({"metric": f"precision@{k}", "value": row["precision@k"]})
        flat_rows.append({"metric": f"recall@{k}", "value": row["recall@k"]})
        flat_rows.append({"metric": f"lift@{k}", "value": row["lift@k"]})

    pd.DataFrame(flat_rows).to_csv(out_dir / f"{split_name}_metrics_table.csv", index=False)

    # HTML plot report
    fig = make_plotly_report(y_true, y_prob, metrics, title=f"{split_name.upper()} Performance", cfg=cfg)
    fig.write_html(out_dir / f"{split_name}_report.html", include_plotlyjs="cdn")

    return metrics
