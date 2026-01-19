from mltrain.eval_report import ReportConfig, write_report_artifacts


# after training:
# runs/<run_name>/reports/
#   val_metrics.json
#   val_metrics_table.csv
#   val_report.html
#   test_metrics.json
#   test_metrics_table.csv
#   test_report.html

if X_val is not None and hasattr(model, "predict_proba"):
    p_val = model.predict_proba(X_val)[:, 1]
    write_report_artifacts(
        run_dir=run_dir,
        split_name="val",
        y_true=y_val.to_numpy(),
        y_prob=p_val,
        cfg=ReportConfig(ks=(50, 100, 200), fbeta=2.0),  # F2 if you care more about recall
    )

if X_test is not None and hasattr(model, "predict_proba"):
    p_test = model.predict_proba(X_test)[:, 1]
    write_report_artifacts(run_dir, "test", y_test.to_numpy(), p_test)
