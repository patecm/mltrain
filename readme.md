# Training script(s)  

## Structure

```
mltrain/
  __init__.py
  run_train.py
  config_schema.py
  utils.py
  data.py
  trainers/
    __init__.py
    xgboost_training.py
configs/
  xgb_time_split.yaml
```

## Run it:
```python -m mltrain.run_train --config configs/xgb_time_split.yaml```

Override example:
```
python -m mltrain.run_train --config configs/xgb_time_split.yaml \
  --override model.params.max_depth=8 \
  --override training.early_stopping_rounds=25
```
## Artifacts created:

```
runs/xgboost_20260118_.../
  config.yaml
  meta.json
  features.json
  metrics.json
  history.csv        # per-iter logloss/auc etc (if val split present)
  logs/train.log
runs/index.csv
```
# Metrics  

For severe class imbalance, you’ll usually get the most honest picture from metrics that focus on the positive class and/or the ranking quality at the top of the list. Here are the ones I’d prioritize for XGBoost (binary classification).

## Top recommendations  
1) PR-AUC (Average Precision / AUC-PR)

Best default “single number” when positives are rare.

Unlike ROC-AUC, it reflects how precision collapses when you chase recall.

In XGBoost, use: eval_metric: aucpr

2) Precision / Recall at an operating point

Pick thresholds that match your use case and report:

Recall (how many positives you catch)

Precision (how many flagged are truly positive)

F1 (balanced) or Fβ (tilt toward recall if missing positives is costly; e.g., F2)

This is the metric set people actually act on.

3) Recall@K / Precision@K / Lift@K

If you’re going to review the “top N” cases (fraud, alerts, leads):

Recall@K: “How many of all true positives appear in the top K?”

Precision@K: “How clean is the top K?”

Lift@K: “How much better than random are we in top K?”
These are often more aligned with business value than any AUC.  

## Also very useful (depending on needs)  
4) Confusion matrix metrics for the minority class

Specificity is often huge by default in imbalanced sets; don’t let it fool you.

Consider Balanced Accuracy (average of TPR and TNR) if you want a simple correction for imbalance.

5) MCC (Matthews Correlation Coefficient)

Strong “overall” metric that stays informative under imbalance.

Especially good when you want one thresholded score but don’t want accuracy’s pitfalls.

6) Calibration metrics (if you use probabilities)

If decisions depend on predicted probabilities (risk scoring, expected cost):

Log loss (eval_metric: logloss) – good for probabilistic quality, but can be dominated by the majority class unless you weight appropriately.  

Brier score + calibration curves – great for “are these probabilities meaningful?”

### Metrics to treat carefully
ROC-AUC  
Can look great even when your model is useless for finding positives at practical thresholds.  
Still fine as a secondary metric for ranking quality, but don’t rely on it alone.  


If you want a consistent experiment report:

AUC-PR (aucpr) primary  
Recall@K / Precision@K (choose K based on your review budget)  
Recall, Precision, F1 or F2 at a chosen threshold  
MCC at that same threshold  
Log loss (and optionally calibration plot) if you care about probability quality  

### XGBoost eval_metric suggestions  

In your config, a good starting list is:

aucpr  
logloss  
(optionally) auc as secondary   

```
training:
  eval_metrics: [aucpr, logloss, auc]
```
