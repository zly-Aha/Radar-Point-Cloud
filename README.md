# 服务器训练包

## 内容
- `code/`: 训练与评估脚本
- `Processed_Dataset_NPY/`: 预处理后的数据
- `requirements.txt`: 依赖版本
- `run_on_server.ps1`: Windows 服务器启动示例
- `run_on_server.sh`: Linux 服务器启动示例

## 训练入口
优先使用 7 折 LOSO：
```powershell
python code\run_loso_7fold.py --output_dir experiment_loso_7fold_results --methods radar_stnet --device cuda --num_workers 8
```

如果只想先跑单次训练与测试：
```powershell
python code\experiment1_action_classification.py --dataset_path Processed_Dataset_NPY --output_dir experiment1_results --methods radar_stnet --device cuda --num_workers 8
```

## 说明
- 代码会从 `Processed_Dataset_NPY` 读取数据。
- 如果服务器没有 CUDA，请把 `--device cuda` 改成 `--device cpu`。
- 若只验证代码能否启动，可先加 `--skip_degradation`。

## 修订稿缺失实验
详细入口、参数和输出文件见 `MISSING_EXPERIMENTS_README.md`。一键执行可用：
```bash
bash run_missing_experiments.sh cuda 8 missing_experiments_results
```
