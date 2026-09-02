# 修订稿缺失实验代码

每个实验均为独立入口，默认路径相对于 `server_training_package`：

1. `python code/analyze_input_statistics.py`：输入统计漂移。
2. `python code/extract_feature_drift.py --checkpoint_root checkpoints`：内部特征余弦保持率与 AUC。
3. `python code/plot_drift_performance_relation.py`：合并统计漂移与已有性能 AUC；只使用实际 CSV，不补造数据。
4. `python code/run_frontend_persistence_filter.py --checkpoint_root checkpoints`：前端时间持久性滤波，含未滤波对照、sit/stand 召回率。
5. `python code/run_loso_all_models.py --device cuda`：九个方法（3 个传统 + 6 个深度）的 7 折 LOSO，并开启缺失、噪声、合成杂波和结构化杂波评估。已有权重需要复用时加 `--skip_clean_train`。
6. `python code/run_significance_tests.py`：折级 bootstrap 95% CI、Friedman/Wilcoxon（需要 scipy）。

先安装 `requirements.txt`，再按服务器资源调整 `--num_workers`、`--batch_size` 与 `--max_windows_per_file`。结果目录均会自动创建，原始明细和汇总 CSV 分开保存。
