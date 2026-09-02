from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "缺少 numpy。请用已安装依赖的 Python 运行，例如：\n"
        "& 'D:\\Program Files (x86)\\Python\\python.exe' "
        "'C:\\Users\\Qy\\Desktop\\Radar\\02_experiments\\code\\check_window_trial_aggregation.py'"
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_subject_id(path: Path) -> int:
    token = path.stem.split("_")[0]
    digits = "".join(ch for ch in token if ch.isdigit())
    return int(digits) if digits else -1


def window_count(num_steps: int, seq_len: int, stride: int) -> int:
    if num_steps < seq_len:
        return 0
    return (num_steps - seq_len) // stride + 1


def macro_f1_score(y_true, y_pred, labels):
    scores = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        scores.append(f1)
    return sum(scores) / len(scores) if scores else 0.0


def majority_vote_trial_metrics(y_true, y_pred, file_ids, labels):
    grouped_true = defaultdict(list)
    grouped_pred = defaultdict(list)

    for true_label, pred_label, file_id in zip(y_true, y_pred, file_ids):
        grouped_true[file_id].append(int(true_label))
        grouped_pred[file_id].append(int(pred_label))

    trial_true, trial_pred = [], []
    for file_id in sorted(grouped_true):
        trial_true.append(Counter(grouped_true[file_id]).most_common(1)[0][0])
        trial_pred.append(Counter(grouped_pred[file_id]).most_common(1)[0][0])

    correct = sum(t == p for t, p in zip(trial_true, trial_pred))
    trial_accuracy = correct / len(trial_true) if trial_true else 0.0
    trial_macro_f1 = macro_f1_score(trial_true, trial_pred, labels)
    return {
        "trial_count": len(trial_true),
        "trial_accuracy": trial_accuracy,
        "trial_macro_f1": trial_macro_f1,
    }


def collect_split_stats(dataset_path: Path, split: str, seq_len: int, stride: int):
    split_dir = dataset_path / split
    files = sorted(split_dir.rglob("*.npy"))
    stats = {
        "split": split,
        "trial_files": len(files),
        "total_time_steps": 0,
        "total_windows": 0,
        "min_steps": None,
        "max_steps": None,
        "min_windows": None,
        "max_windows": None,
        "per_class_trials": Counter(),
        "per_class_windows": Counter(),
        "subjects": Counter(),
    }

    rows = []
    for path in files:
        chunks = np.load(path, allow_pickle=True)
        steps = len(chunks)
        windows = window_count(steps, seq_len, stride)
        class_name = path.parent.name
        subject = parse_subject_id(path)

        stats["total_time_steps"] += steps
        stats["total_windows"] += windows
        stats["min_steps"] = steps if stats["min_steps"] is None else min(stats["min_steps"], steps)
        stats["max_steps"] = steps if stats["max_steps"] is None else max(stats["max_steps"], steps)
        stats["min_windows"] = windows if stats["min_windows"] is None else min(stats["min_windows"], windows)
        stats["max_windows"] = windows if stats["max_windows"] is None else max(stats["max_windows"], windows)
        stats["per_class_trials"][class_name] += 1
        stats["per_class_windows"][class_name] += windows
        stats["subjects"][subject] += 1
        rows.append((path, class_name, subject, steps, windows))

    return stats, rows


def build_perfect_window_predictions(all_rows, label2idx, seq_len, stride):
    y_true, y_pred, file_ids = [], [], []
    for path, class_name, _subject, steps, _windows in all_rows:
        label = label2idx[class_name]
        for start in range(0, steps - seq_len + 1, stride):
            y_true.append(label)
            y_pred.append(label)
            file_ids.append(str(path))
    return y_true, y_pred, file_ids


def main():
    parser = argparse.ArgumentParser(description="检查动作试次切段、滑窗数量和试次级多数投票。")
    parser.add_argument("--dataset_path", default="Processed_Dataset_NPY")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--frames_per_chunk", type=int, default=6)
    parser.add_argument("--seq_len", type=int, default=25)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    dataset_path = resolve_path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"找不到数据目录: {dataset_path}")

    chunk_seconds = args.frames_per_chunk / args.fps
    window_seconds = args.seq_len * chunk_seconds
    stride_seconds = args.stride * chunk_seconds

    print("=== 滑窗参数 ===")
    print(f"数据目录: {dataset_path}")
    print(f"采样频率: {args.fps:g} 帧/s")
    print(f"每个时间步融合帧数: {args.frames_per_chunk}")
    print(f"每个时间步时长: {chunk_seconds:.3f} s")
    print(f"模型输入长度 T: {args.seq_len} 个时间步")
    print(f"单个窗口覆盖时长: {window_seconds:.3f} s")
    print(f"滑窗步长: {args.stride} 个时间步 = {stride_seconds:.3f} s")

    label_names = sorted(
        {p.parent.name for split in ["Train", "Ver", "Test"] for p in (dataset_path / split).rglob("*.npy")}
    )
    label2idx = {name: idx for idx, name in enumerate(label_names)}
    labels = list(label2idx.values())

    print("\n=== 数据集窗口统计 ===")
    all_rows = []
    for split in ["Train", "Ver", "Test"]:
        stats, rows = collect_split_stats(dataset_path, split, args.seq_len, args.stride)
        all_rows.extend(rows)
        print(
            f"{split}: 试次文件 {stats['trial_files']} 个，"
            f"时间步总数 {stats['total_time_steps']}，"
            f"窗口总数 {stats['total_windows']}，"
            f"每试次时间步范围 {stats['min_steps']}–{stats['max_steps']}，"
            f"每试次窗口范围 {stats['min_windows']}–{stats['max_windows']}"
        )
        print(f"  被试文件数: {dict(sorted(stats['subjects'].items()))}")
        print(f"  各类试次数: {dict(sorted(stats['per_class_trials'].items()))}")

    print("\n=== 公式核对 ===")
    example_steps = 900
    example_windows = window_count(example_steps, args.seq_len, args.stride)
    print(f"若每个试次包含 {example_steps} 个时间步，则窗口数 = ({example_steps} - {args.seq_len}) / {args.stride} + 1 = {example_windows}")

    print("\n=== 试次级多数投票核对 ===")
    y_true, y_pred, file_ids = build_perfect_window_predictions(all_rows, label2idx, args.seq_len, args.stride)
    metrics = majority_vote_trial_metrics(y_true, y_pred, file_ids, labels)
    print("用真实标签模拟完美窗口预测时：")
    print(f"  试次数: {metrics['trial_count']}")
    print(f"  试次级 Acc: {metrics['trial_accuracy']:.4f}")
    print(f"  试次级 Macro-F1: {metrics['trial_macro_f1']:.4f}")

    print("\n多数投票函数示例：")
    print("  某一试次窗口预测为 [3, 3, 3, 5, 5]，众数为 3，因此该试次最终预测类别为 3。")


if __name__ == "__main__":
    main()

