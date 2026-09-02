# 服务器训练顺序

## 1. 先跑主结果

```powershell
python code\run_loso_7fold.py --output_dir experiment_loso_7fold_results --methods radar_stnet --device cuda --num_workers 8
```

## 2. 再跑机制实验的已有消融模型退化评估

```powershell
python code\run_mechanism_ablation_degradation.py --checkpoint_root checkpoints --ablation_output_dir checkpoints --output_dir experiment_mechanism_results --eval_time_shuffle --device cuda --num_workers 8
```

说明：
- `full` 会直接用 `checkpoints\train_p1-p5_val_p6_test_p7_radar_stnet.pth`
- 现成消融模型会直接从 `checkpoints\*_best.pth` 读取

## 3. 补训新变体

如果 `no_raw_stats_best.pth`、`max_only_best.pth`、`first_order_motion_best.pth` 不存在，就先跑：

```powershell
python code\experiment3_ablation_tsne_v2.py --output_dir experiment3_results\mechanism --variants no_raw_stats max_only first_order_motion --device cuda --num_workers 8 --skip_tsne
```

## 4. 再把补训后的模型回到机制实验退化评估里

```powershell
python code\run_mechanism_ablation_degradation.py --checkpoint_root checkpoints --ablation_output_dir experiment3_results\mechanism --output_dir experiment_mechanism_results --eval_time_shuffle --device cuda --num_workers 8
```

## 5. 最后看导出的结果

- `experiment_mechanism_results\mechanism_clean_results.csv`
- `experiment_mechanism_results\mechanism_degradation_summary.csv`
- `experiment_mechanism_results\mechanism_degradation_auc.csv`
- `experiment_mechanism_results\mechanism_time_shuffle_results.csv`
- `experiment_mechanism_results\mechanism_clean_class_f1.csv`
- `experiment_mechanism_results\mechanism_time_shuffle_class_f1.csv`
