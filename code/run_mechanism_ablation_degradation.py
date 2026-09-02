from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset

from dataset_loader import DatasetConfig, Transforms
from experiment1_action_classification import collect_records
from experiment2_degradation_robustness import DegradeConfig, DegradedWindowDataset
from experiment3_ablation_tsne_v2 import build_ablation_model, load_original_radar_stnet


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]


VARIANTS = [
    "full",
    "no_motion",
    "no_tcn",
    "no_transformer",
    "no_attention_pool",
    "no_raw_stats",
    "max_only",
    "first_order_motion",
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str):
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_float_list(values):
    return [float(v) for v in values]


def parse_int_list(values):
    return [int(v) for v in values]


def safe_torch_load(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def resolve_path(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def find_full_checkpoint(root: Path):
    candidates = sorted(root.rglob("*radar_stnet*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    fixed = [p for p in candidates if "train_p1-p5_val_p6_test_p7" in p.name]
    return fixed[0] if fixed else candidates[0]


def find_variant_checkpoint(root: Path, variant: str):
    direct_candidates = [
        root / f"{variant}_best.pth",
        root / "checkpoints" / f"{variant}_best.pth",
    ]
    for direct in direct_candidates:
        if direct.exists():
            return direct

    candidates = sorted(root.rglob(f"*{variant}_best.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]

    candidates = sorted(root.rglob(f"*{variant}.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_model_for_variant(variant: str, checkpoint_path: Path, class_names, seq_len: int, device):
    if variant == "full":
        return load_original_radar_stnet(checkpoint_path, class_names, seq_len, device)

    checkpoint = safe_torch_load(checkpoint_path, device)
    model = build_ablation_model(variant, len(class_names), seq_len).to(device)
    state = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    model.eval()
    return model


def majority_vote_metrics(y_true, y_pred, file_ids):
    grouped_true = defaultdict(list)
    grouped_pred = defaultdict(list)

    for t, p, fid in zip(y_true, y_pred, file_ids):
        grouped_true[fid].append(int(t))
        grouped_pred[fid].append(int(p))

    trial_true, trial_pred = [], []
    for fid in sorted(grouped_true):
        trial_true.append(Counter(grouped_true[fid]).most_common(1)[0][0])
        trial_pred.append(Counter(grouped_pred[fid]).most_common(1)[0][0])

    return {
        "trial_accuracy": float(accuracy_score(trial_true, trial_pred)),
        "trial_macro_f1": float(f1_score(trial_true, trial_pred, average="macro", zero_division=0)),
        "trial_count": len(trial_true),
    }


@torch.no_grad()
def predict_with_report(model, loader, device, class_names):
    model.eval()
    y_true, y_pred, file_ids = [], [], []
    dataset = loader.dataset

    for x, y, file_idx in loader:
        x = x.to(device, non_blocking=True)
        pred = model(x).argmax(dim=1).cpu().numpy()

        y_true.extend(y.numpy().tolist())
        y_pred.extend(pred.tolist())
        file_ids.extend([dataset.file_keys[int(i)] for i in file_idx.numpy().tolist()])

    labels = np.arange(len(class_names))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    metrics = {
        "sample_accuracy": float(accuracy_score(y_true, y_pred)),
        "sample_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "sample_weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "sample_count": int(len(y_true)),
    }
    metrics.update(majority_vote_metrics(y_true, y_pred, file_ids))
    return metrics, report


class ShuffledWindowDataset(DegradedWindowDataset):
    def __init__(
        self,
        records,
        data_cfg: DatasetConfig,
        seq_len: int,
        degrade: DegradeConfig,
        shuffle_seed: int = 2026,
        window_stride: int = 1,
        max_windows_per_file: int = 0,
    ):
        super().__init__(
            records=records,
            data_cfg=data_cfg,
            seq_len=seq_len,
            degrade=degrade,
            window_stride=window_stride,
            max_windows_per_file=max_windows_per_file,
        )
        self.shuffle_seed = shuffle_seed

    def __getitem__(self, idx):
        file_idx, start, label = self.items[idx]
        chunks = self.chunks_by_file[file_idx]
        window = list(chunks[start:start + self.seq_len])

        rng = np.random.default_rng(self.shuffle_seed + idx * 1009)
        perm = rng.permutation(len(window))
        window = [window[i] for i in perm]

        processed = []
        for local_i, pts in enumerate(window):
            frame_rng = np.random.default_rng(self.degrade.seed + idx * 1009 + local_i * 9173)
            pts = self.degrade_points(pts, frame_rng)
            pts = self.transformer.resampler.resample(pts)
            pts = self.transformer.normalize(pts)
            processed.append(pts)

        x = torch.tensor(np.stack(processed, axis=0), dtype=torch.float32)
        return x, torch.tensor(label, dtype=torch.long), torch.tensor(file_idx, dtype=torch.long)


def make_loader(dataset, batch_size: int, num_workers: int, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def summarize_degradation(raw_df: pd.DataFrame):
    rows = []
    for (variant, degradation), g in raw_df.groupby(["variant", "degradation"]):
        g = g.sort_values("level")
        x = g["level"].to_numpy(dtype=np.float64)
        y = g["sample_macro_f1_mean"].to_numpy(dtype=np.float64)

        clean = float(y[0]) if len(y) else math.nan
        if len(x) > 1 and x[-1] > x[0]:
            auc_range = float(np.trapz(y, x) / (x[-1] - x[0]))
        else:
            auc_range = clean

        nauc = float(auc_range / clean) if clean and clean > 0 else math.nan
        nonzero = y[x > 0]
        avg_drop = float(clean - nonzero.mean()) if len(nonzero) else 0.0
        final_drop = float(clean - y[-1]) if len(y) else math.nan

        rows.append({
            "variant": variant,
            "degradation": degradation,
            "clean_macro_f1": clean,
            "auc_range_norm": auc_range,
            "nauc": nauc,
            "avg_drop": avg_drop,
            "final_drop": final_drop,
        })

    return pd.DataFrame(rows)


def degradation_clean_dataset(records, data_cfg, seq_len, window_stride, max_windows_per_file):
    return DegradedWindowDataset(
        records=records,
        data_cfg=data_cfg,
        seq_len=seq_len,
        degrade=DegradeConfig(kind="clean", level=0.0, seed=0),
        window_stride=window_stride,
        max_windows_per_file=max_windows_per_file,
    )


def evaluate_variant_clean(
    variant: str,
    model,
    test_records,
    data_cfg,
    seq_len,
    window_stride,
    max_windows_per_file,
    batch_size,
    num_workers,
    device,
    class_names,
):
    dataset = degradation_clean_dataset(test_records, data_cfg, seq_len, window_stride, max_windows_per_file)
    loader = make_loader(dataset, batch_size, num_workers, device)
    metrics, report = predict_with_report(model, loader, device, class_names)
    return metrics, report


def evaluate_variant_degradation(
    variant: str,
    model,
    test_records,
    data_cfg,
    seq_len,
    window_stride,
    max_windows_per_file,
    batch_size,
    num_workers,
    device,
    class_names,
    missing_levels,
    noise_levels,
    clutter_levels,
    seeds,
):
    raw_rows = []
    plan = [
        ("missing", missing_levels),
        ("noise", noise_levels),
        ("clutter", clutter_levels),
    ]

    for degradation, levels in plan:
        for level in levels:
            level_seeds = [seeds[0]] if float(level) == 0.0 else seeds
            for seed in level_seeds:
                dataset = DegradedWindowDataset(
                    records=test_records,
                    data_cfg=data_cfg,
                    seq_len=seq_len,
                    degrade=DegradeConfig(degradation, float(level), int(seed)),
                    window_stride=window_stride,
                    max_windows_per_file=max_windows_per_file,
                )
                loader = make_loader(dataset, batch_size, num_workers, device)
                metrics, _ = predict_with_report(model, loader, device, class_names)
                raw_rows.append({
                    "variant": variant,
                    "degradation": degradation,
                    "level": float(level),
                    "seed": int(seed),
                    **metrics,
                })
    return raw_rows


def evaluate_time_shuffle(
    variant: str,
    model,
    test_records,
    data_cfg,
    seq_len,
    window_stride,
    max_windows_per_file,
    batch_size,
    num_workers,
    device,
    class_names,
    shuffle_seed: int,
):
    dataset = ShuffledWindowDataset(
        records=test_records,
        data_cfg=data_cfg,
        seq_len=seq_len,
        degrade=DegradeConfig("clean", 0.0, 0),
        shuffle_seed=shuffle_seed,
        window_stride=window_stride,
        max_windows_per_file=max_windows_per_file,
    )
    loader = make_loader(dataset, batch_size, num_workers, device)
    metrics, report = predict_with_report(model, loader, device, class_names)
    return metrics, report


def extract_class_f1(report: dict, class_names, variant: str, setting: str):
    rows = []
    for class_name in class_names:
        if class_name not in report:
            continue
        rows.append({
            "variant": variant,
            "setting": setting,
            "class_name": class_name,
            "precision": float(report[class_name]["precision"]),
            "recall": float(report[class_name]["recall"]),
            "f1": float(report[class_name]["f1-score"]),
            "support": int(report[class_name]["support"]),
        })
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Mechanism degradation and negative-control evaluation for Radar-STNet.")
    parser.add_argument("--dataset_path", default="Processed_Dataset_NPY")
    parser.add_argument("--checkpoint_root", default="experiment1_results")
    parser.add_argument("--ablation_output_dir", default="experiment3_results/mechanism")
    parser.add_argument("--output_dir", default="experiment_mechanism_results")
    parser.add_argument("--variants", nargs="+", default=VARIANTS)
    parser.add_argument("--test_subjects", nargs="+", type=int, default=[7])
    parser.add_argument("--seq_len", type=int, default=25)
    parser.add_argument("--n_points", type=int, default=128)
    parser.add_argument("--window_stride", type=int, default=1)
    parser.add_argument("--max_windows_per_file", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", default=["0", "1", "2"])
    parser.add_argument("--missing_levels", nargs="+", default=["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7"])
    parser.add_argument("--noise_levels", nargs="+", default=["0", "0.01", "0.02", "0.05", "0.08", "0.10"])
    parser.add_argument("--clutter_levels", nargs="+", default=["0", "0.1", "0.2", "0.4", "0.6", "0.8"])
    parser.add_argument("--eval_time_shuffle", action="store_true")
    parser.add_argument("--shuffle_variant", default="full")
    parser.add_argument("--shuffle_seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    dataset_path = resolve_path(args.dataset_path)
    checkpoint_root = resolve_path(args.checkpoint_root)
    ablation_output_dir = resolve_path(args.ablation_output_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_int_list(args.seeds)
    missing_levels = parse_float_list(args.missing_levels)
    noise_levels = parse_float_list(args.noise_levels)
    clutter_levels = parse_float_list(args.clutter_levels)

    records, class_names = collect_records(str(dataset_path))
    test_records = [r for r in records if r.subject in args.test_subjects]
    data_cfg = DatasetConfig(
        dataset_path=str(dataset_path),
        n_chunk_per_data=args.seq_len,
        n_sample_per_chunk=args.n_points,
    )

    clean_rows = []
    degradation_raw_rows = []
    class_f1_rows = []
    shuffle_rows = []
    shuffle_class_rows = []

    run_config = vars(args).copy()
    run_config["dataset_path"] = str(dataset_path)
    run_config["checkpoint_root"] = str(checkpoint_root)
    run_config["ablation_output_dir"] = str(ablation_output_dir)
    run_config["output_dir"] = str(output_dir)
    run_config["class_names"] = class_names
    (output_dir / "mechanism_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for variant in args.variants:
        print(f"\n========== Variant: {variant} ==========")

        if variant == "full":
            ckpt_path = find_full_checkpoint(checkpoint_root)
            if ckpt_path is None:
                raise FileNotFoundError(f"Missing full checkpoint under: {checkpoint_root}")
        else:
            ckpt_path = find_variant_checkpoint(ablation_output_dir, variant)
            if ckpt_path is None:
                raise FileNotFoundError(f"Missing checkpoint for {variant} under: {ablation_output_dir}")

        print(f"[ckpt] {ckpt_path}")
        model = load_model_for_variant(variant, ckpt_path, class_names, args.seq_len, get_device(args.device))
        device = next(model.parameters()).device

        clean_metrics, clean_report = evaluate_variant_clean(
            variant,
            model,
            test_records,
            data_cfg,
            args.seq_len,
            args.window_stride,
            args.max_windows_per_file,
            args.batch_size,
            args.num_workers,
            device,
            class_names,
        )
        clean_rows.append({
            "variant": variant,
            "setting": "clean",
            "checkpoint": str(ckpt_path),
            **clean_metrics,
        })
        class_f1_rows.extend(extract_class_f1(clean_report, class_names, variant, "clean"))

        deg_rows = evaluate_variant_degradation(
            variant,
            model,
            test_records,
            data_cfg,
            args.seq_len,
            args.window_stride,
            args.max_windows_per_file,
            args.batch_size,
            args.num_workers,
            device,
            class_names,
            missing_levels,
            noise_levels,
            clutter_levels,
            seeds,
        )
        degradation_raw_rows.extend(deg_rows)

        if args.eval_time_shuffle and variant == args.shuffle_variant:
            shuffle_metrics, shuffle_report = evaluate_time_shuffle(
                variant,
                model,
                test_records,
                data_cfg,
                args.seq_len,
                args.window_stride,
                args.max_windows_per_file,
                args.batch_size,
                args.num_workers,
                device,
                class_names,
                args.shuffle_seed,
            )
            shuffle_rows.append({
                "variant": variant,
                "setting": "time_shuffle",
                "shuffle_seed": args.shuffle_seed,
                **shuffle_metrics,
            })
            shuffle_class_rows.extend(extract_class_f1(shuffle_report, class_names, variant, "time_shuffle"))

    clean_df = pd.DataFrame(clean_rows)
    raw_df = pd.DataFrame(degradation_raw_rows)
    class_f1_df = pd.DataFrame(class_f1_rows)
    shuffle_df = pd.DataFrame(shuffle_rows)
    shuffle_class_df = pd.DataFrame(shuffle_class_rows)

    clean_df.to_csv(output_dir / "mechanism_clean_results.csv", index=False, encoding="utf-8-sig")
    class_f1_df.to_csv(output_dir / "mechanism_clean_class_f1.csv", index=False, encoding="utf-8-sig")

    if not raw_df.empty:
        raw_df.to_csv(output_dir / "mechanism_degradation_raw.csv", index=False, encoding="utf-8-sig")
        summary_rows = []
        for (variant, degradation, level), g in raw_df.groupby(["variant", "degradation", "level"]):
            summary_rows.append({
                "variant": variant,
                "degradation": degradation,
                "level": float(level),
                "sample_accuracy_mean": float(g["sample_accuracy"].mean()),
                "sample_accuracy_std": float(g["sample_accuracy"].std(ddof=0)),
                "sample_macro_f1_mean": float(g["sample_macro_f1"].mean()),
                "sample_macro_f1_std": float(g["sample_macro_f1"].std(ddof=0)),
                "trial_accuracy_mean": float(g["trial_accuracy"].mean()),
                "trial_accuracy_std": float(g["trial_accuracy"].std(ddof=0)),
                "trial_macro_f1_mean": float(g["trial_macro_f1"].mean()),
                "trial_macro_f1_std": float(g["trial_macro_f1"].std(ddof=0)),
                "runs": int(len(g)),
            })
        summary_df = pd.DataFrame(summary_rows).sort_values(["variant", "degradation", "level"])
        summary_df.to_csv(output_dir / "mechanism_degradation_summary.csv", index=False, encoding="utf-8-sig")
        auc_df = summarize_degradation(summary_df)
        auc_df.to_csv(output_dir / "mechanism_degradation_auc.csv", index=False, encoding="utf-8-sig")
    else:
        summary_df = pd.DataFrame()
        auc_df = pd.DataFrame()

    if not shuffle_df.empty:
        shuffle_df.to_csv(output_dir / "mechanism_time_shuffle_results.csv", index=False, encoding="utf-8-sig")
        shuffle_class_df.to_csv(output_dir / "mechanism_time_shuffle_class_f1.csv", index=False, encoding="utf-8-sig")

    print(f"\nSaved results to: {output_dir.resolve()}")
    if not clean_df.empty:
        print("\nClean summary:")
        print(clean_df[["variant", "sample_accuracy", "sample_macro_f1", "trial_accuracy", "trial_macro_f1"]].to_string(index=False))
    if not auc_df.empty:
        print("\nDegradation AUC summary:")
        print(auc_df[["variant", "degradation", "clean_macro_f1", "auc_range_norm", "nauc", "avg_drop", "final_drop"]].to_string(index=False))
    if not shuffle_df.empty:
        print("\nTime-shuffle summary:")
        print(shuffle_df[["variant", "sample_accuracy", "sample_macro_f1", "trial_accuracy", "trial_macro_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
