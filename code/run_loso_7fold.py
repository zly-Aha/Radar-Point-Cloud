from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset_loader import DatasetConfig
from experiment1_action_classification import (
    ALL_METHODS,
    DEEP_METHODS,
    RadarWindowDataset,
    collect_records,
    run_traditional,
    set_seed,
    train_deep,
)
from experiment2_degradation_robustness import (
    DegradeConfig,
    DegradedWindowDataset,
    evaluate_model as evaluate_degraded_model,
    load_model as load_degradation_model,
)
from experiment2_realistic_clutter import (
    ClutterConfig,
    RealisticClutterDataset,
    evaluate_model as evaluate_realistic_model,
)


METHOD_LABELS = {
    "stats_knn": "Stats-KNN",
    "stats_rf": "Stats-RF",
    "stats_svm": "Stats-SVM",
    "stats_lr": "Stats-LR",
    "pointnet_gru": "PointNet-GRU",
    "robhar_like": "RobHAR-like",
    "pointnet_tcn": "PointNet-TCN",
    "pct_gru": "PCT-GRU",
    "dgcnn_gru": "DGCNN-GRU",
    "radar_stnet": "Radar-STNet",
}


def parse_float_list(values):
    return [float(v) for v in values]


def parse_int_list(values):
    return [int(v) for v in values]


def get_device(device_arg: str):
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def mean_std(values, ddof=1):
    vals = np.asarray(values, dtype=np.float64)
    if len(vals) == 0:
        return math.nan, math.nan
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(vals.mean()), float(vals.std(ddof=ddof))


def fmt_percent(mean, std):
    return f"{mean * 100:.2f} ± {std * 100:.2f}%"


def fmt_decimal(mean, std):
    return f"{mean:.4f} ± {std:.4f}"


def auc_mean_over_range(levels, values, divide_by_clean=False):
    x = np.asarray(levels, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if len(x) == 0:
        return math.nan
    if len(x) == 1 or x[-1] == x[0]:
        auc = float(y[0])
    else:
        auc = float(np.trapezoid(y, x) / (x[-1] - x[0]))
    if divide_by_clean:
        clean = float(y[0])
        if clean != 0:
            auc /= clean
    return auc


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_path(output_dir: Path, fold_name: str, method: str) -> Path:
    return output_dir / "checkpoints" / f"{fold_name}_{method}.pth"


def run_clean_fold(
    records,
    class_names,
    data_cfg,
    train_ids,
    val_ids,
    test_ids,
    method,
    train_args,
):
    train_records = [r for r in records if r.subject in train_ids]
    val_records = [r for r in records if r.subject in val_ids]
    test_records = [r for r in records if r.subject in test_ids]

    train_deep_ds = RadarWindowDataset(
        train_records,
        data_cfg,
        train_args.seq_len,
        augment=True,
        window_stride=train_args.window_stride,
        max_windows_per_file=train_args.max_windows_per_file,
    )
    train_plain_ds = RadarWindowDataset(
        train_records,
        data_cfg,
        train_args.seq_len,
        augment=False,
        window_stride=train_args.window_stride,
        max_windows_per_file=train_args.max_windows_per_file,
    )
    val_ds = RadarWindowDataset(
        val_records,
        data_cfg,
        train_args.seq_len,
        augment=False,
        window_stride=train_args.window_stride,
        max_windows_per_file=train_args.max_windows_per_file,
    )
    test_ds = RadarWindowDataset(
        test_records,
        data_cfg,
        train_args.seq_len,
        augment=False,
        window_stride=train_args.window_stride,
        max_windows_per_file=train_args.max_windows_per_file,
    )

    if method in DEEP_METHODS:
        metrics, params = train_deep(
            method,
            train_deep_ds,
            val_ds,
            test_ds,
            class_names,
            train_args,
            train_args.fold_name,
        )
    else:
        metrics, params = run_traditional(
            method,
            train_plain_ds,
            test_ds,
            class_names,
            train_args.seed,
        )
    return metrics, params


def evaluate_synthetic_degradation(
    method,
    ckpt_path,
    records,
    class_names,
    data_cfg,
    test_ids,
    levels,
    kind,
    seeds,
    args,
):
    device = get_device(args.device)
    model = load_degradation_model(method, ckpt_path, class_names, args.seq_len, device)
    test_records = [r for r in records if r.subject in test_ids]
    level_rows = []

    for level in levels:
        level_seeds = [seeds[0]] if float(level) == 0.0 else seeds
        metrics_for_level = []
        for seed in level_seeds:
            dataset = DegradedWindowDataset(
                records=test_records,
                data_cfg=data_cfg,
                seq_len=args.seq_len,
                degrade=DegradeConfig(kind=kind, level=float(level), seed=int(seed)),
                window_stride=args.window_stride,
                max_windows_per_file=args.max_windows_per_file,
            )
            loader = DataLoader(
                dataset,
                batch_size=args.degradation_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
            metrics_for_level.append(evaluate_degraded_model(model, loader, device))

        row = {"level": float(level), "runs": len(metrics_for_level)}
        for key in ["sample_accuracy", "sample_macro_f1", "trial_accuracy", "trial_macro_f1"]:
            vals = [m[key] for m in metrics_for_level]
            row[f"{key}_mean"], row[f"{key}_std"] = mean_std(vals, ddof=0)
        level_rows.append(row)

    return level_rows


def evaluate_realistic_clutter(
    method,
    ckpt_path,
    records,
    class_names,
    data_cfg,
    test_ids,
    levels,
    seeds,
    args,
):
    device = get_device(args.device)
    model = load_degradation_model(method, ckpt_path, class_names, args.seq_len, device)
    test_records = [r for r in records if r.subject in test_ids]
    level_rows = []

    for level in levels:
        level_seeds = [seeds[0]] if float(level) == 0.0 else seeds
        metrics_for_level = []
        for seed in level_seeds:
            dataset = RealisticClutterDataset(
                records=test_records,
                data_cfg=data_cfg,
                seq_len=args.seq_len,
                clutter_cfg=ClutterConfig(level=float(level), seed=int(seed), mode=args.realistic_mode),
                window_stride=args.window_stride,
                max_windows_per_file=args.max_windows_per_file,
            )
            loader = DataLoader(
                dataset,
                batch_size=args.degradation_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
            metrics_for_level.append(evaluate_realistic_model(model, loader, device))

        row = {"level": float(level), "runs": len(metrics_for_level)}
        for key in ["sample_accuracy", "sample_macro_f1", "trial_accuracy", "trial_macro_f1"]:
            vals = [m[key] for m in metrics_for_level]
            row[f"{key}_mean"], row[f"{key}_std"] = mean_std(vals, ddof=0)
        level_rows.append(row)

    return level_rows


def run_degradation_fold(method, ckpt, records, class_names, data_cfg, test_ids, args):
    seeds = parse_int_list(args.degradation_seeds)
    plans = []
    if args.run_missing:
        plans.append(("missing", "point_missing", parse_float_list(args.missing_levels)))
    if args.run_noise:
        plans.append(("noise", "coordinate_noise", parse_float_list(args.noise_levels)))
    if args.run_clutter:
        plans.append(("clutter", "strong_synthetic_clutter", parse_float_list(args.clutter_levels)))

    results = []
    for kind, label, levels in plans:
        rows = evaluate_synthetic_degradation(
            method=method,
            ckpt_path=ckpt,
            records=records,
            class_names=class_names,
            data_cfg=data_cfg,
            test_ids=test_ids,
            levels=levels,
            kind=kind,
            seeds=seeds,
            args=args,
        )
        auc = auc_mean_over_range(
            [r["level"] for r in rows],
            [r["sample_macro_f1_mean"] for r in rows],
            divide_by_clean=args.divide_auc_by_clean,
        )
        results.append((label, auc, rows))

    if args.run_realistic_clutter:
        levels = parse_float_list(args.realistic_clutter_levels)
        rows = evaluate_realistic_clutter(
            method=method,
            ckpt_path=ckpt,
            records=records,
            class_names=class_names,
            data_cfg=data_cfg,
            test_ids=test_ids,
            levels=levels,
            seeds=seeds,
            args=args,
        )
        auc = auc_mean_over_range(
            [r["level"] for r in rows],
            [r["sample_macro_f1_mean"] for r in rows],
            divide_by_clean=args.divide_auc_by_clean,
        )
        results.append(("structured_realistic_clutter", auc, rows))

    return results


def summarize_clean(clean_rows):
    summary_rows = []
    for method in sorted({r["method"] for r in clean_rows}):
        subset = [r for r in clean_rows if r["method"] == method]
        row = {"method": method, "folds": len(subset)}
        for key in [
            "sample_accuracy",
            "sample_macro_f1",
            "sample_weighted_f1",
            "trial_accuracy",
            "trial_macro_f1",
            "trial_weighted_f1",
        ]:
            row[f"{key}_mean"], row[f"{key}_std"] = mean_std([r[key] for r in subset], ddof=1)
        summary_rows.append(row)
    return summary_rows


def summarize_degradation(degradation_rows):
    summary_rows = []
    for method in sorted({r["method"] for r in degradation_rows}):
        for degradation in sorted({r["degradation"] for r in degradation_rows if r["method"] == method}):
            subset = [r for r in degradation_rows if r["method"] == method and r["degradation"] == degradation]
            mean, std = mean_std([r["auc"] for r in subset], ddof=1)
            summary_rows.append({
                "method": method,
                "degradation": degradation,
                "folds": len(subset),
                "auc_mean": mean,
                "auc_std": std,
            })
    return summary_rows


def write_copy_text(output_dir, clean_summary, degradation_summary):
    lines = []
    lines.append("LOSO 七折结果（可复制粘贴）")
    lines.append("")
    for row in clean_summary:
        label = METHOD_LABELS.get(row["method"], row["method"])
        lines.append(
            f"{label} 在 7 折 LOSO 协议下取得样本级 Acc "
            f"{fmt_percent(row['sample_accuracy_mean'], row['sample_accuracy_std'])}，"
            f"样本级 Macro-F1 {fmt_percent(row['sample_macro_f1_mean'], row['sample_macro_f1_std'])}；"
            f"试次级 Acc {fmt_percent(row['trial_accuracy_mean'], row['trial_accuracy_std'])}，"
            f"试次级 Macro-F1 {fmt_percent(row['trial_macro_f1_mean'], row['trial_macro_f1_std'])}。"
        )
    if degradation_summary:
        lines.append("")
        for row in degradation_summary:
            label = METHOD_LABELS.get(row["method"], row["method"])
            deg_name = {
                "point_missing": "点缺失",
                "coordinate_noise": "坐标噪声",
                "strong_synthetic_clutter": "强合成杂波",
                "structured_realistic_clutter": "结构化真实杂波",
            }.get(row["degradation"], row["degradation"])
            lines.append(
                f"{label} 在 {deg_name} 条件下的 AUC 为 "
                f"{fmt_decimal(row['auc_mean'], row['auc_std'])}。"
            )
    text = "\n".join(lines) + "\n"
    (output_dir / "LOSO_七折可粘贴结果.txt").write_text(text, encoding="utf-8")
    return text


def parse_args():
    parser = argparse.ArgumentParser(description="Run 7-fold LOSO clean and degradation evaluation.")
    parser.add_argument("--dataset_path", default="Processed_Dataset_NPY")
    parser.add_argument("--output_dir", default="experiment_loso_7fold_results")
    parser.add_argument("--methods", nargs="+", default=["radar_stnet"])
    parser.add_argument("--seq_len", type=int, default=25)
    parser.add_argument("--n_points", type=int, default=128)
    parser.add_argument("--window_stride", type=int, default=1)
    parser.add_argument("--max_windows_per_file", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--degradation_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--label_smoothing", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip_clean_train", action="store_true", help="Use existing fold checkpoints when present.")
    parser.add_argument("--skip_degradation", action="store_true")
    parser.add_argument("--run_missing", action="store_true", default=True)
    parser.add_argument("--run_noise", action="store_true", default=True)
    parser.add_argument("--run_clutter", action="store_true", default=False)
    parser.add_argument("--run_realistic_clutter", action="store_true", default=False)
    parser.add_argument("--missing_levels", nargs="+", default=["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7"])
    parser.add_argument("--noise_levels", nargs="+", default=["0", "0.01", "0.02", "0.05", "0.08", "0.10"])
    parser.add_argument("--clutter_levels", nargs="+", default=["0", "0.1", "0.2", "0.4", "0.6", "0.8"])
    parser.add_argument("--realistic_clutter_levels", nargs="+", default=["0", "0.02", "0.05", "0.10", "0.15", "0.20", "0.30"])
    parser.add_argument("--degradation_seeds", nargs="+", default=["0", "1", "2", "3", "4"])
    parser.add_argument("--realistic_mode", choices=["mixed", "static", "aid"], default="mixed")
    parser.add_argument("--divide_auc_by_clean", action="store_true", help="Match formula AUC/(dmax*P0). Default matches existing code output.")
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.methods) == 1 and args.methods[0] == "all":
        args.methods = ALL_METHODS

    dataset_path = Path(args.dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    args.dataset_path = str(dataset_path)
    args.output_dir = str(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    records, class_names = collect_records(args.dataset_path)
    subjects = sorted({r.subject for r in records if r.subject > 0})
    data_cfg = DatasetConfig(
        dataset_path=args.dataset_path,
        n_chunk_per_data=args.seq_len,
        n_sample_per_chunk=args.n_points,
    )

    clean_rows = []
    degradation_rows = []
    level_rows = []

    run_config = vars(args).copy()
    run_config["subjects"] = subjects
    run_config["class_names"] = class_names
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for test_id in subjects:
        test_index = subjects.index(test_id)
        val_id = subjects[(test_index + 1) % len(subjects)]
        train_ids = [sid for sid in subjects if sid not in {test_id, val_id}]
        fold_name = f"loso_test_p{test_id}_val_p{val_id}"

        for method in args.methods:
            start = time.time()
            ckpt = checkpoint_path(output_dir, fold_name, method)
            print(f"\n=== {fold_name} | {method} ===")
            print(f"train={train_ids}, val={[val_id]}, test={[test_id]}")

            if args.skip_clean_train:
                if not ckpt.exists():
                    print(f"[skip] checkpoint not found: {ckpt}")
                    continue
                print(f"[reuse] {ckpt}")
                # Reuse the checkpoint and continue directly to degradation
                # evaluation below.  Previously this branch skipped the entire
                # fold, producing no robustness CSV when checkpoints existed.
                if args.skip_degradation or method not in DEEP_METHODS:
                    continue
                degradation_results = run_degradation_fold(
                    method=method, ckpt=ckpt, records=records,
                    class_names=class_names, data_cfg=data_cfg,
                    test_ids=[test_id], args=args,
                )
                for degradation, auc, rows in degradation_results:
                    degradation_rows.append({"fold": fold_name, "method": method, "degradation": degradation, "auc": auc, "test_subjects": f"p{test_id}"})
                    for row in rows:
                        level_row = {"fold": fold_name, "method": method, "degradation": degradation, "test_subjects": f"p{test_id}"}; level_row.update(row); level_rows.append(level_row)
                write_csv(output_dir / "loso_degradation_auc_by_fold.csv", degradation_rows)
                write_csv(output_dir / "loso_degradation_level_results.csv", level_rows)
                continue

            train_args = SimpleNamespace(
                dataset_path=args.dataset_path,
                output_dir=str(output_dir),
                seq_len=args.seq_len,
                n_points=args.n_points,
                window_stride=args.window_stride,
                max_windows_per_file=args.max_windows_per_file,
                epochs=args.epochs,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                lr=args.lr,
                weight_decay=args.weight_decay,
                label_smoothing=args.label_smoothing,
                seed=args.seed,
                device=args.device,
                save_checkpoints=True,
                fold_name=fold_name,
            )
            metrics, params = run_clean_fold(
                records=records,
                class_names=class_names,
                data_cfg=data_cfg,
                train_ids=train_ids,
                val_ids=[val_id],
                test_ids=[test_id],
                method=method,
                train_args=train_args,
            )
            clean_row = {
                "fold": fold_name,
                "method": method,
                "train_subjects": ",".join(f"p{i}" for i in train_ids),
                "val_subjects": f"p{val_id}",
                "test_subjects": f"p{test_id}",
                "params": params,
                "seconds": round(time.time() - start, 2),
                "sample_accuracy": metrics["sample_accuracy"],
                "sample_macro_f1": metrics["sample_macro_f1"],
                "sample_weighted_f1": metrics["sample_weighted_f1"],
                "sample_count": metrics["sample_count"],
                "trial_accuracy": metrics["trial_accuracy"],
                "trial_macro_f1": metrics["trial_macro_f1"],
                "trial_weighted_f1": metrics["trial_weighted_f1"],
                "trial_count": metrics["trial_count"],
            }
            clean_rows.append(clean_row)
            write_csv(output_dir / "loso_clean_fold_results.csv", clean_rows)

            if args.skip_degradation or method not in DEEP_METHODS:
                continue

            if not ckpt.exists():
                print(f"[warn] checkpoint missing after train, skip degradation: {ckpt}")
                continue

            degradation_results = run_degradation_fold(
                method=method,
                ckpt=ckpt,
                records=records,
                class_names=class_names,
                data_cfg=data_cfg,
                test_ids=[test_id],
                args=args,
            )
            for degradation, auc, rows in degradation_results:
                degradation_rows.append({
                    "fold": fold_name,
                    "method": method,
                    "degradation": degradation,
                    "auc": auc,
                    "test_subjects": f"p{test_id}",
                })
                for row in rows:
                    level_row = {
                        "fold": fold_name,
                        "method": method,
                        "degradation": degradation,
                        "test_subjects": f"p{test_id}",
                    }
                    level_row.update(row)
                    level_rows.append(level_row)

            write_csv(output_dir / "loso_degradation_auc_by_fold.csv", degradation_rows)
            write_csv(output_dir / "loso_degradation_level_results.csv", level_rows)

    clean_summary = summarize_clean(clean_rows)
    degradation_summary = summarize_degradation(degradation_rows)
    write_csv(output_dir / "loso_clean_mean_std.csv", clean_summary)
    write_csv(output_dir / "loso_degradation_auc_mean_std.csv", degradation_summary)
    copy_text = write_copy_text(output_dir, clean_summary, degradation_summary)

    print("\n" + copy_text)
    print(f"结果已保存到: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
