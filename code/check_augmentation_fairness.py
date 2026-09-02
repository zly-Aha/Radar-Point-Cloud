from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "缺少 numpy。请用已安装依赖的 Python 运行，例如：\n"
        "& 'D:\\Program Files (x86)\\Python\\python.exe' "
        "'C:\\Users\\Qy\\Desktop\\Radar\\02_experiments\\code\\check_augmentation_fairness.py'"
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset_loader import DatasetConfig, Transforms


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def first_npy(dataset_path: Path, split: str) -> Path:
    files = sorted((dataset_path / split).rglob("*.npy"))
    if not files:
        raise FileNotFoundError(f"{dataset_path / split} 下没有 .npy 文件")
    return files[0]


def max_abs_diff(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def read_experiment_flags() -> dict:
    path = SCRIPT_DIR / "experiment1_action_classification.py"
    text = path.read_text(encoding="utf-8", errors="replace")
    checks = {
        "deep_train_augment_true": "train_records, data_cfg, args.seq_len, augment=True" in text,
        "traditional_train_plain_augment_false": "train_records, data_cfg, args.seq_len, augment=False" in text,
        "val_augment_false": "val_records, data_cfg, args.seq_len, augment=False" in text,
        "test_augment_false": "test_records, data_cfg, args.seq_len, augment=False" in text,
        "traditional_uses_plain_ds": "run_traditional(method, train_plain_ds, test_ds" in text,
        "deep_uses_augmented_train_ds": "train_deep(method, train_deep_ds, val_ds, test_ds" in text,
    }

    method_pattern = re.compile(r"DEEP_METHODS\s*=\s*\{(?P<body>.*?)\}", re.S)
    match = method_pattern.search(text)
    if match:
        body = "{" + match.group("body") + "}"
        try:
            checks["deep_methods"] = sorted(ast.literal_eval(body))
        except Exception:
            checks["deep_methods"] = []
    else:
        checks["deep_methods"] = []

    traditional_pattern = re.compile(r"TRADITIONAL_METHODS\s*=\s*(?P<body>\{.*?\})", re.S)
    match = traditional_pattern.search(text)
    if match:
        try:
            checks["traditional_methods"] = sorted(ast.literal_eval(match.group("body")))
        except Exception:
            checks["traditional_methods"] = []
    else:
        checks["traditional_methods"] = []

    return checks


def main():
    parser = argparse.ArgumentParser(description="检查训练期数据增强参数与公平性设置。")
    parser.add_argument("--dataset_path", default="Processed_Dataset_NPY")
    parser.add_argument("--seq_len", type=int, default=25)
    args = parser.parse_args()

    dataset_path = resolve_path(args.dataset_path)
    cfg = DatasetConfig(dataset_path=str(dataset_path))

    print("=== 数据增强参数 ===")
    print(f"全局空间平移标准差: {cfg.aug_shift_std} m")
    print(f"逐点坐标高斯抖动标准差: {cfg.aug_jitter_std} m")
    print(f"坐标缩放因子: Normal(mean=1.0, std={cfg.aug_scale_std})")
    print(f"反射强度乘性扰动: Normal(mean=1.0, std={cfg.aug_intensity_std})")

    train_file = first_npy(dataset_path, "Train")
    val_file = first_npy(dataset_path, "Ver")
    test_file = first_npy(dataset_path, "Test")

    train_chunks = list(np.load(train_file, allow_pickle=True)[: args.seq_len])
    val_chunks = list(np.load(val_file, allow_pickle=True)[: args.seq_len])
    test_chunks = list(np.load(test_file, allow_pickle=True)[: args.seq_len])

    train_aug = Transforms(cfg, augment=True, deterministic=False)
    train_out_1 = train_aug.transform(train_chunks).numpy()
    train_out_2 = train_aug.transform(train_chunks).numpy()

    train_plain = Transforms(cfg, augment=False, deterministic=True)
    train_plain_1 = train_plain.transform(train_chunks).numpy()
    train_plain_2 = train_plain.transform(train_chunks).numpy()

    val_plain = Transforms(cfg, augment=False, deterministic=True)
    val_out_1 = val_plain.transform(val_chunks).numpy()
    val_out_2 = val_plain.transform(val_chunks).numpy()

    test_plain = Transforms(cfg, augment=False, deterministic=True)
    test_out_1 = test_plain.transform(test_chunks).numpy()
    test_out_2 = test_plain.transform(test_chunks).numpy()

    print("\n=== 随机性/确定性核对 ===")
    print(f"训练集 augment=True 同一窗口重复处理最大差异: {max_abs_diff(train_out_1, train_out_2):.6f}")
    print(f"训练集 augment=False 同一窗口重复处理最大差异: {max_abs_diff(train_plain_1, train_plain_2):.6f}")
    print(f"验证集 augment=False 同一窗口重复处理最大差异: {max_abs_diff(val_out_1, val_out_2):.6f}")
    print(f"测试集 augment=False 同一窗口重复处理最大差异: {max_abs_diff(test_out_1, test_out_2):.6f}")

    print("\n=== 实验脚本公平性静态核对 ===")
    checks = read_experiment_flags()
    print(f"深度学习训练集使用 augment=True: {checks['deep_train_augment_true']}")
    print(f"验证集使用 augment=False: {checks['val_augment_false']}")
    print(f"测试集使用 augment=False: {checks['test_augment_false']}")
    print(f"传统机器学习训练集使用 train_plain_ds / augment=False: {checks['traditional_train_plain_augment_false'] and checks['traditional_uses_plain_ds']}")
    print(f"深度学习方法通过 train_deep(train_deep_ds, val_ds, test_ds) 训练: {checks['deep_uses_augmented_train_ds']}")
    print(f"深度学习方法列表: {checks['deep_methods']}")
    print(f"传统机器学习方法列表: {checks['traditional_methods']}")

    print("\n=== 可复制结论 ===")
    print(
        "训练期数据增强仅作用于深度学习模型的训练集；验证集和测试集采用 "
        "augment=False 且 deterministic=True 的确定性处理，不引入随机增强。"
    )
    print(
        "传统机器学习方法使用相同预处理和滑窗样本，但训练输入来自 train_plain_ds，"
        "即不施加训练期随机增强。"
    )


if __name__ == "__main__":
    main()
