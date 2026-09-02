param([string]$Device="cuda", [int]$Workers=8, [string]$Out="missing_experiments_results")
$ErrorActionPreference="Stop"
python code\analyze_input_statistics.py --output_dir "$Out\input_statistics"
python code\run_loso_all_models.py --output_dir "$Out\loso_all_models" --device $Device --num_workers $Workers
python code\extract_feature_drift.py --checkpoint_root checkpoints --output_dir "$Out\feature_drift" --device $Device --num_workers $Workers
python code\run_frontend_persistence_filter.py --checkpoint_root checkpoints --output_dir "$Out\frontend_filter" --device $Device --num_workers $Workers
python code\plot_drift_performance_relation.py --statistics_csv "$Out\input_statistics\input_statistics_summary.csv" --performance_csv "$Out\loso_all_models\loso_degradation_auc_mean_std.csv" --output_dir "$Out\drift_performance"
python code\run_significance_tests.py --fold_csv "$Out\loso_all_models\loso_degradation_auc_by_fold.csv" --output_dir "$Out\significance"
