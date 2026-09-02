#!/usr/bin/env bash
set -euo pipefail
DEVICE="${1:-cuda}"; WORKERS="${2:-8}"; OUT="${3:-missing_experiments_results}"
python code/analyze_input_statistics.py --output_dir "$OUT/input_statistics"
python code/run_loso_all_models.py --output_dir "$OUT/loso_all_models" --device "$DEVICE" --num_workers "$WORKERS"
python code/extract_feature_drift.py --checkpoint_root checkpoints --output_dir "$OUT/feature_drift" --device "$DEVICE" --num_workers "$WORKERS"
python code/run_frontend_persistence_filter.py --checkpoint_root checkpoints --output_dir "$OUT/frontend_filter" --device "$DEVICE" --num_workers "$WORKERS"
python code/plot_drift_performance_relation.py --statistics_csv "$OUT/input_statistics/input_statistics_summary.csv" --performance_csv "$OUT/loso_all_models/loso_degradation_auc_mean_std.csv" --output_dir "$OUT/drift_performance"
python code/run_significance_tests.py --fold_csv "$OUT/loso_all_models/loso_degradation_auc_by_fold.csv" --output_dir "$OUT/significance"
