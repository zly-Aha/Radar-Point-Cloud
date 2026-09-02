from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

from dataset_loader import DatasetConfig, Transforms
from experiment1_action_classification import build_deep_model, collect_records


METHOD_LABELS = {
    "pointnet_gru": "PointNet-GRU",
    "robhar_like": "RobHAR-like",
    "pointnet_tcn": "PointNet-TCN",
    "pct_gru": "PCT-GRU",
    "dgcnn_gru": "DGCNN-GRU",
    "radar_stnet": "Radar-STNet",
}

METHOD_ORDER = ["pointnet_gru", "robhar_like", "pointnet_tcn", "pct_gru", "dgcnn_gru", "radar_stnet"]


@dataclass
class DegradeConfig:
    kind: str
    level: float
    seed: int


def get_device(device_arg: str):
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def safe_torch_load(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def find_checkpoint(root: Path, method: str):
    candidates = sorted(root.rglob(f"*{method}.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None

    fixed = [p for p in candidates if "train_p1-p5_val_p6_test_p7" in p.name]
    return fixed[0] if fixed else candidates[0]


def load_model(method: str, checkpoint_path: Path, class_names, seq_len: int, device):
    checkpoint = safe_torch_load(checkpoint_path, device)
    model = build_deep_model(method, len(class_names), seq_len).to(device)

    state = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()
    return model


def clean_points(points):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 4:
        return np.empty((0, 4), dtype=np.float32)
    return points[:, :4].astype(np.float32, copy=True)


def apply_missing(points, ratio, rng):
    n = len(points)
    if n == 0 or ratio <= 0:
        return points
    keep = max(1, int(round(n * (1.0 - ratio))))
    keep = min(keep, n)
    idx = rng.choice(n, size=keep, replace=False)
    return points[idx]


def apply_noise(points, std, rng):
    if len(points) == 0 or std <= 0:
        return points
    out = points.copy()
    out[:, :3] += rng.normal(0.0, std, size=out[:, :3].shape).astype(np.float32)
    return out


def make_clutter(points, ratio, rng, cfg: DatasetConfig):
    n = len(points)
    if ratio <= 0:
        return points

    base_n = max(n, 32)
    extra_n = int(round(base_n * ratio))
    if extra_n <= 0:
        return points

    extra = np.zeros((extra_n, 4), dtype=np.float32)
    mode = rng.random(extra_n)

    wall_mask = mode < 0.55
    local_mask = ~wall_mask

    nw = int(wall_mask.sum())
    if nw > 0:
        extra[wall_mask, 0] = rng.uniform(cfg.min_x, cfg.max_x, size=nw)
        side = rng.choice([cfg.min_y, cfg.max_y], size=nw)
        extra[wall_mask, 1] = side + rng.normal(0.0, 0.05, size=nw)
        extra[wall_mask, 2] = rng.uniform(cfg.min_z, cfg.max_z, size=nw)
        extra[wall_mask, 3] = rng.uniform(3.0, 35.0, size=nw)

    nl = int(local_mask.sum())
    if nl > 0:
        if n > 0:
            center = points[:, :3].mean(axis=0)
        else:
            center = np.array([
                (cfg.min_x + cfg.max_x) / 2,
                (cfg.min_y + cfg.max_y) / 2,
                (cfg.min_z + cfg.max_z) / 2,
            ], dtype=np.float32)

        extra[local_mask, :3] = center + rng.normal(
            0.0, [0.28, 0.28, 0.20], size=(nl, 3)
        ).astype(np.float32)
        extra[local_mask, 3] = rng.uniform(5.0, 45.0, size=nl)

    extra[:, 0] = np.clip(extra[:, 0], cfg.min_x, cfg.max_x)
    extra[:, 1] = np.clip(extra[:, 1], cfg.min_y, cfg.max_y)
    extra[:, 2] = np.clip(extra[:, 2], cfg.min_z, cfg.max_z)

    if n == 0:
        return extra
    return np.concatenate([points, extra], axis=0).astype(np.float32)


class DegradedWindowDataset(Dataset):
    def __init__(
        self,
        records,
        data_cfg: DatasetConfig,
        seq_len: int,
        degrade: DegradeConfig,
        window_stride: int = 1,
        max_windows_per_file: int = 0,
    ):
        self.data_cfg = data_cfg
        self.seq_len = seq_len
        self.degrade = degrade
        self.transformer = Transforms(data_cfg, augment=False, deterministic=True)
        self.chunks_by_file = []
        self.file_keys = []
        self.items = []

        for rec in records:
            chunks = np.load(rec.path, allow_pickle=True)
            if len(chunks) < seq_len:
                continue

            file_idx = len(self.chunks_by_file)
            self.chunks_by_file.append(chunks)
            self.file_keys.append(f"p{rec.subject}:{rec.label_name}:{rec.path.name}")

            starts = list(range(0, len(chunks) - seq_len + 1, window_stride))
            if max_windows_per_file > 0 and len(starts) > max_windows_per_file:
                rng = np.random.default_rng(2026 + file_idx)
                starts = sorted(rng.choice(starts, size=max_windows_per_file, replace=False).tolist())

            for start in starts:
                self.items.append((file_idx, start, rec.label))

        if not self.items:
            raise RuntimeError("当前退化评估没有可用滑窗样本。")

    def __len__(self):
        return len(self.items)

    def degrade_points(self, points, rng):
        pts = clean_points(points)
        if self.degrade.kind == "missing":
            pts = apply_missing(pts, self.degrade.level, rng)
        elif self.degrade.kind == "noise":
            pts = apply_noise(pts, self.degrade.level, rng)
        elif self.degrade.kind == "clutter":
            pts = make_clutter(pts, self.degrade.level, rng, self.data_cfg)
        elif self.degrade.kind == "clean":
            pass
        else:
            raise ValueError(f"Unknown degradation kind: {self.degrade.kind}")
        return pts

    def __getitem__(self, idx):
        file_idx, start, label = self.items[idx]
        chunks = self.chunks_by_file[file_idx]
        window = chunks[start:start + self.seq_len]

        processed = []
        for local_i, pts in enumerate(window):
            rng = np.random.default_rng(self.degrade.seed + idx * 1009 + local_i * 9173)
            pts = self.degrade_points(pts, rng)
            pts = self.transformer.resampler.resample(pts)
            pts = self.transformer.normalize(pts)
            processed.append(pts)

        x = torch.tensor(np.stack(processed, axis=0), dtype=torch.float32)
        return x, torch.tensor(label, dtype=torch.long), torch.tensor(file_idx, dtype=torch.long)


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
def evaluate_model(model, loader, device):
    y_true, y_pred, file_ids = [], [], []
    dataset = loader.dataset

    model.eval()
    for x, y, file_idx in loader:
        x = x.to(device, non_blocking=True)
        pred = model(x).argmax(dim=1).cpu().numpy()

        y_true.extend(y.numpy().tolist())
        y_pred.extend(pred.tolist())
        file_ids.extend([dataset.file_keys[int(i)] for i in file_idx.numpy().tolist()])

    metrics = {
        "sample_accuracy": float(accuracy_score(y_true, y_pred)),
        "sample_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "sample_count": len(y_true),
    }
    metrics.update(majority_vote_metrics(y_true, y_pred, file_ids))
    return metrics


def summarize(raw_df):
    group_cols = ["method", "degradation", "level"]
    metric_cols = ["sample_accuracy", "sample_macro_f1", "trial_accuracy", "trial_macro_f1"]

    rows = []
    for keys, g in raw_df.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        for col in metric_cols:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = float(g[col].std(ddof=0))
        row["runs"] = int(len(g))
        rows.append(row)

    return pd.DataFrame(rows)


def robustness_table(summary_df):
    rows = []
    for (method, degradation), g in summary_df.groupby(["method", "degradation"]):
        g = g.sort_values("level")
        x = g["level"].to_numpy(dtype=np.float64)
        y = g["sample_macro_f1_mean"].to_numpy(dtype=np.float64)

        clean = float(y[0]) if len(y) else math.nan
        if len(x) > 1 and x[-1] > x[0]:
            auc = float(np.trapz(y, x) / (x[-1] - x[0]))
        else:
            auc = clean

        nonzero = y[x > 0]
        avg_drop = float(clean - nonzero.mean()) if len(nonzero) else 0.0
        final_drop = float(clean - y[-1]) if len(y) else math.nan

        rows.append({
            "method": method,
            "degradation": degradation,
            "clean_macro_f1": clean,
            "normalized_auc": auc,
            "avg_drop": avg_drop,
            "final_drop": final_drop,
        })

    return pd.DataFrame(rows)


def plot_curves(summary_df, out_dir: Path):
    colors = {
        "pointnet_gru": "#4C78A8",
        "robhar_like": "#F58518",
        "pointnet_tcn": "#54A24B",
        "pct_gru": "#B279A2",
        "dgcnn_gru": "#72B7B2",
        "radar_stnet": "#D62728",
    }

    xlabels = {
        "missing": "Missing Rate (%)",
        "noise": "Gaussian Noise Std. (m)",
        "clutter": "Clutter Ratio (%)",
    }

    titles = {
        "missing": "Robustness Under Point Missing",
        "noise": "Robustness Under Coordinate Noise",
        "clutter": "Robustness Under Clutter Interference",
    }

    for degradation in ["missing", "noise", "clutter"]:
        df = summary_df[summary_df["degradation"] == degradation].copy()
        if df.empty:
            continue

        fig, ax = plt.subplots(figsize=(8.5, 5.2))

        for method in METHOD_ORDER:
            g = df[df["method"] == method].sort_values("level")
            if g.empty:
                continue

            x = g["level"].to_numpy(dtype=np.float64)
            if degradation in {"missing", "clutter"}:
                x = x * 100.0

            y = g["sample_macro_f1_mean"].to_numpy(dtype=np.float64) * 100.0
            std = g["sample_macro_f1_std"].fillna(0).to_numpy(dtype=np.float64) * 100.0

            ax.plot(
                x, y,
                marker="o",
                linewidth=2.2 if method == "radar_stnet" else 1.8,
                markersize=6 if method == "radar_stnet" else 5,
                color=colors.get(method, None),
                label=METHOD_LABELS.get(method, method),
            )
            if len(g) > 1:
                ax.fill_between(x, y - std, y + std, color=colors.get(method, None), alpha=0.12)

        ax.set_title(titles[degradation])
        ax.set_xlabel(xlabels[degradation])
        ax.set_ylabel("Sample Macro-F1 (%)")
        ax.set_ylim(0, 101)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(ncol=2, fontsize=9)

        fig.tight_layout()
        fig.savefig(out_dir / f"fig_{degradation}_macro_f1.png", dpi=300)
        fig.savefig(out_dir / f"fig_{degradation}_macro_f1.pdf")
        plt.close(fig)


def parse_float_list(values):
    return [float(v) for v in values]


def parse_int_list(values):
    return [int(v) for v in values]


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 2: synthetic point-cloud degradation robustness.")
    parser.add_argument("--dataset_path", default="Processed_Dataset_NPY")
    parser.add_argument("--checkpoint_root", default="experiment1_results")
    parser.add_argument("--output_dir", default="experiment2_results")
    parser.add_argument("--methods", nargs="+", default=METHOD_ORDER)
    parser.add_argument("--test_subjects", nargs="+", type=int, default=[7])
    parser.add_argument("--seq_len", type=int, default=25)
    parser.add_argument("--n_points", type=int, default=128)
    parser.add_argument("--window_stride", type=int, default=1)
    parser.add_argument("--max_windows_per_file", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", nargs="+", default=["0", "1", "2"])
    parser.add_argument("--missing_levels", nargs="+", default=["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7"])
    parser.add_argument("--noise_levels", nargs="+", default=["0", "0.01", "0.02", "0.05", "0.08", "0.10"])
    parser.add_argument("--clutter_levels", nargs="+", default=["0", "0.1", "0.2", "0.4", "0.6", "0.8"])
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_int_list(args.seeds)
    missing_levels = parse_float_list(args.missing_levels)
    noise_levels = parse_float_list(args.noise_levels)
    clutter_levels = parse_float_list(args.clutter_levels)

    records, class_names = collect_records(args.dataset_path)
    test_records = [r for r in records if r.subject in args.test_subjects]

    data_cfg = DatasetConfig(
        dataset_path=args.dataset_path,
        n_chunk_per_data=args.seq_len,
        n_sample_per_chunk=args.n_points,
    )

    checkpoint_root = Path(args.checkpoint_root)
    checkpoint_map = {}
    for method in args.methods:
        ckpt = find_checkpoint(checkpoint_root, method)
        if ckpt is None:
            print(f"[skip] missing checkpoint for {method}")
            continue
        checkpoint_map[method] = ckpt
        print(f"[ckpt] {method}: {ckpt}")

    raw_rows = []

    degradation_plan = [
        ("missing", missing_levels),
        ("noise", noise_levels),
        ("clutter", clutter_levels),
    ]

    for method, ckpt in checkpoint_map.items():
        print(f"\n========== Method: {method} ==========")
        model = load_model(method, ckpt, class_names, args.seq_len, device)

        for degradation, levels in degradation_plan:
            for level in levels:
                level_seeds = [seeds[0]] if float(level) == 0.0 else seeds

                for seed in level_seeds:
                    print(f"[eval] method={method}, degradation={degradation}, level={level}, seed={seed}")

                    dataset = DegradedWindowDataset(
                        records=test_records,
                        data_cfg=data_cfg,
                        seq_len=args.seq_len,
                        degrade=DegradeConfig(degradation, float(level), int(seed)),
                        window_stride=args.window_stride,
                        max_windows_per_file=args.max_windows_per_file,
                    )

                    loader = DataLoader(
                        dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=device.type == "cuda",
                    )

                    metrics = evaluate_model(model, loader, device)
                    raw_rows.append({
                        "method": method,
                        "degradation": degradation,
                        "level": float(level),
                        "seed": int(seed),
                        **metrics,
                    })

                    pd.DataFrame(raw_rows).to_csv(out_dir / "degradation_raw.csv", index=False, encoding="utf-8-sig")

    raw_df = pd.DataFrame(raw_rows)
    summary_df = summarize(raw_df)
    auc_df = robustness_table(summary_df)

    raw_df.to_csv(out_dir / "degradation_raw.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "degradation_summary.csv", index=False, encoding="utf-8-sig")
    auc_df.to_csv(out_dir / "robustness_auc_summary.csv", index=False, encoding="utf-8-sig")

    plot_curves(summary_df, out_dir)

    config = vars(args)
    config["device"] = str(device)
    config["class_names"] = class_names
    config["checkpoint_map"] = {k: str(v) for k, v in checkpoint_map.items()}
    (out_dir / "experiment2_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nDone. Results saved to: {out_dir.resolve()}")
    print(auc_df)


if __name__ == "__main__":
    main()