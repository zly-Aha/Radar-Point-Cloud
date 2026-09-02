# run_loso_7fold.py 使用说明

## 默认正式运行：Radar-STNet 七折 clean + 点缺失/坐标噪声 AUC

如果当前命令行前面显示 `(.venv)`，而 `.venv` 里没有安装 `numpy/torch/scikit-learn`，会出现 `ModuleNotFoundError: No module named 'numpy'`。本项目当前可用依赖在系统 Python 中，建议先退出虚拟环境：

```powershell
deactivate
```

然后运行：

```powershell
python 02_experiments\code\run_loso_7fold.py --output_dir experiment_loso_7fold_results --methods radar_stnet --device cuda --num_workers 8
```

如果不想退出虚拟环境，也可以直接指定当前机器上已安装依赖的 Python：

```powershell
& 'D:\Program Files (x86)\Python\python.exe' 02_experiments\code\run_loso_7fold.py --output_dir experiment_loso_7fold_results --methods radar_stnet --device cuda --num_workers 8
```

从任何目录启动也可以使用绝对路径；脚本会自动把相对数据路径解析到项目根目录：

```powershell
& 'D:\Program Files (x86)\Python\python.exe' 'C:\Users\Qy\Desktop\Radar\02_experiments\code\run_loso_7fold.py' --output_dir experiment_loso_7fold_results --methods radar_stnet --device cuda --num_workers 8
```

默认参数：

- 7 折 LOSO：每折 1 人测试、下一编号 1 人验证、其余 5 人训练。
- 每折重新训练 30 epoch。
- Clean 指标：样本级 Acc、样本级 Macro-F1、试次级 Acc、试次级 Macro-F1。
- 退化指标：默认跑点缺失和坐标噪声；非零退化等级用 5 个随机种子。
- AUC 口径默认与现有代码一致：`trapz(Macro-F1, degradation_level) / degradation_range`，未再除以 clean 性能 `P0`。

## 只跑七折 clean，不跑退化

```powershell
python 02_experiments\code\run_loso_7fold.py --output_dir experiment_loso_7fold_clean --methods radar_stnet --skip_degradation --device cuda --num_workers 8
```

## 同时跑强合成杂波和结构化真实杂波

```powershell
python 02_experiments\code\run_loso_7fold.py --output_dir experiment_loso_7fold_full_degradation --methods radar_stnet --run_clutter --run_realistic_clutter --device cuda --num_workers 8
```

## 跑所有模型

```powershell
python 02_experiments\code\run_loso_7fold.py --output_dir experiment_loso_7fold_all_models --methods all --device cuda --num_workers 8
```

## 如果正文公式坚持除以 P0

若正文采用 `AUC / (dmax * P0)` 口径，加上：

```powershell
--divide_auc_by_clean
```

否则建议保持默认，并把正文公式写成“曲线面积除以退化强度区间长度”。

## 输出文件

运行结束后看输出目录：

- `LOSO_七折可粘贴结果.txt`：中文可直接复制粘贴结果。
- `loso_clean_mean_std.csv`：clean 七折均值 ± 标准差。
- `loso_clean_fold_results.csv`：每折 clean 结果。
- `loso_degradation_auc_mean_std.csv`：退化 AUC 七折均值 ± 标准差。
- `loso_degradation_auc_by_fold.csv`：每折退化 AUC。
- `loso_degradation_level_results.csv`：每折每个退化等级的指标。
- `checkpoints/`：每折训练得到的模型 checkpoint。
