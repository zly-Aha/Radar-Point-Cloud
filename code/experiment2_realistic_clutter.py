from __future__ import annotations

import argparse
import json
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


METHOD_ORDER = ["pointnet_gru", "robhar_like", "pointnet_tcn", "pct_gru", "dgcnn_gru", "radar_stnet"]
METHOD_LABELS = {
    "pointnet_gru": "PointNet-GRU",
    "robhar_like": "RobHAR-like",
    "pointnet_tcn": "PointNet-TCN",
    "pct_gru": "PCT-GRU",
    "dgcnn_gru": "DGCNN-GRU",
    "radar_stnet": "Radar-STNet",
}


@dataclass
class ClutterConfig:
    level: float
    seed: int
    mode: str = "mixed"


def get_device(device_arg: str):
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def clip_roi(xyz, cfg: DatasetConfig):
    xyz[:, 0] = np.clip(xyz[:, 0], cfg.min_x, cfg.max_x)
    xyz[:, 1] = np.clip(xyz[:, 1], cfg.min_y, cfg.max_y)
    xyz[:, 2] = np.clip(xyz[:, 2], cfg.min_z, cfg.max_z)
    return xyz


def make_trial_profile(file_idx: int, seed: int, cfg: DatasetConfig):
    rng = np.random.default_rng(seed + file_idx * 7919)

    n_static = 160
    static_xyz = np.zeros((n_static, 3), dtype=np.float32)

    wall_side = rng.choice([cfg.min_y, cfg.max_y], size=n_static)
    floor_mask = rng.random(n_static) < 0.35

    static_xyz[:, 0] = rng.uniform(cfg.min_x, cfg.max_x, size=n_static)
    static_xyz[:, 1] = wall_side + rng.normal(0.0, 0.035, size=n_static)
    static_xyz[:, 2] = rng.uniform(cfg.min_z, cfg.max_z, size=n_static)

    static_xyz[floor_mask, 1] = rng.uniform(cfg.min_y, cfg.max_y, size=floor_mask.sum())
    static_xyz[floor_mask, 2] = cfg.min_z + rng.normal(0.10, 0.035, size=floor_mask.sum())
    static_xyz = clip_roi(static_xyz, cfg)

    aid_offsets = rng.normal(0.0, [0.15, 0.18, 0.10], size=(96, 3)).astype(np.float32)
    side = rng.choice([-1.0, 1.0])
    aid_offsets[:, 0] += rng.uniform(-0.10, 0.15, size=96)
    aid_offsets[:, 1] += side * rng.uniform(0.25, 0.55, size=96)
    aid_offsets[:, 2] += rng.uniform(-0.45, -0.10, size=96)

    return {
        "static_xyz": static_xyz,
        "static_i": rng.uniform(4.0, 32.0, size=n_static).astype(np.float32),
        "aid_offsets": aid_offsets,
        "aid_i": rng.uniform(8.0, 45.0, size=96).astype(np.float32),
        "phase": float(rng.uniform(0, 2 * np.pi)),
    }


def apply_realistic_clutter(points, level: float, rng, profile, global_t: int, cfg: DatasetConfig, mode: str):
    pts = clean_points(points)
    if level <= 0:
        return pts

    base_n = max(len(pts), 32)
    extra_n = max(1, int(round(base_n * level)))

    if mode == "static":
        n_static, n_aid, n_ghost = extra_n, 0, 0
    elif mode == "aid":
        n_static, n_aid, n_ghost = 0, int(extra_n * 0.75), extra_n - int(extra_n * 0.75)
    else:
        n_static = int(extra_n * 0.45)
        n_aid = int(extra_n * 0.35)
        n_ghost = extra_n - n_static - n_aid

    parts = []

    if n_static > 0:
        idx = np.arange(n_static) % len(profile["static_xyz"])
        xyz = profile["static_xyz"][idx].copy()
        xyz += rng.normal(0.0, [0.006, 0.006, 0.004], size=xyz.shape).astype(np.float32)
        xyz = clip_roi(xyz, cfg)

        intensity = profile["static_i"][idx].copy()
        intensity *= 0.90 + 0.10 * np.sin(0.05 * global_t + profile["phase"])
        parts.append(np.column_stack([xyz, intensity]).astype(np.float32))

    if n_aid > 0:
        if len(pts) > 0:
            center = np.median(pts[:, :3], axis=0)
            center[2] = np.percentile(pts[:, 2], 25)
        else:
            center = np.array([
                (cfg.min_x + cfg.max_x) / 2,
                (cfg.min_y + cfg.max_y) / 2,
                (cfg.min_z + cfg.max_z) / 2,
            ], dtype=np.float32)

        idx = np.arange(n_aid) % len(profile["aid_offsets"])
        drift = np.array([
            0.025 * np.sin(0.08 * global_t + profile["phase"]),
            0.020 * np.cos(0.07 * global_t + profile["phase"]),
            0.010 * np.sin(0.05 * global_t + profile["phase"]),
        ], dtype=np.float32)

        xyz = center + profile["aid_offsets"][idx] + drift
        xyz += rng.normal(0.0, [0.018, 0.018, 0.012], size=xyz.shape).astype(np.float32)
        xyz = clip_roi(xyz, cfg)

        intensity = profile["aid_i"][idx].copy()
        intensity *= 0.85 + 0.15 * np.sin(0.10 * global_t + profile["phase"])
        parts.append(np.column_stack([xyz, intensity]).astype(np.float32))

    if n_ghost > 0 and len(pts) > 0:
        pick = rng.choice(len(pts), size=n_ghost, replace=len(pts) < n_ghost)
        ghost = pts[pick].copy()
        ghost[:, 0] += rng.normal(0.08, 0.04, size=n_ghost)
        ghost[:, 1] += rng.choice([-1.0, 1.0], size=n_ghost) * rng.uniform(0.18, 0.42, size=n_ghost)
        ghost[:, 2] += rng.normal(0.0, 0.05, size=n_ghost)
        ghost[:, :3] = clip_roi(ghost[:, :3], cfg)
        ghost[:, 3] *= rng.uniform(0.25, 0.65, size=n_ghost)
        parts.append(ghost.astype(np.float32))

    if not parts:
        return pts

    clutter = np.concatenate(parts, axis=0)
    if len(pts) == 0:
        return clutter
    return np.concatenate([pts, clutter], axis=0).astype(np.float32)


class RealisticClutterDataset(Dataset):
    def __init__(self, records, data_cfg: DatasetConfig, seq_len: int, clutter_cfg: ClutterConfig,
                 window_stride: int = 1, max_windows_per_file: int = 0):
        self.data_cfg = data_cfg
        self.seq_len = seq_len
        self.clutter_cfg = clutter_cfg
        self.transformer = Transforms(data_cfg, augment=False, deterministic=True)

        self.chunks_by_file = []
        self.file_keys = []
        self.profiles = []
        self.items = []

        for rec in records:
            chunks = np.load(rec.path, allow_pickle=True)
            if len(chunks) < seq_len:
                continue

            file_idx = len(self.chunks_by_file)
            self.chunks_by_file.append(chunks)
            self.file_keys.append(f"p{rec.subject}:{rec.label_name}:{rec.path.name}")
            self.profiles.append(make_trial_profile(file_idx, clutter_cfg.seed, data_cfg))

            starts = list(range(0, len(chunks) - seq_len + 1, window_stride))
            if max_windows_per_file > 0 and len(starts) > max_windows_per_file:
                rng = np.random.default_rng(2026 + file_idx)
                starts = sorted(rng.choice(starts, size=max_windows_per_file, replace=False).tolist())

            for start in starts:
                self.items.append((file_idx, start, rec.label))

        if not self.items:
            raise RuntimeError("没有可用测试滑窗样本。")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        file_idx, start, label = self.items[idx]
        chunks = self.chunks_by_file[file_idx]
        profile = self.profiles[file_idx]

        processed = []
        for local_i, pts in enumerate(chunks[start:start + self.seq_len]):
            global_t = start + local_i
            rng = np.random.default_rng(self.clutter_cfg.seed + file_idx * 10007 + global_t * 9176)
            pts = apply_realistic_clutter(
                pts,
                self.clutter_cfg.level,
                rng,
                profile,
                global_t,
                self.data_cfg,
                self.clutter_cfg.mode,
            )
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
    rows = []
    for (method, level), g in raw_df.groupby(["method", "level"]):
        row = {"method": method, "level": level}
        for col in ["sample_accuracy", "sample_macro_f1", "trial_accuracy", "trial_macro_f1"]:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = float(g[col].std(ddof=0))
        row["runs"] = int(len(g))
        rows.append(row)
    return pd.DataFrame(rows)


def auc_summary(summary_df):
    rows = []
    for method, g in summary_df.groupby("method"):
        g = g.sort_values("level")
        x = g["level"].to_numpy(dtype=np.float64)
        y = g["sample_macro_f1_mean"].to_numpy(dtype=np.float64)

        clean = float(y[0])
        auc = float(np.trapz(y, x) / (x[-1] - x[0])) if x[-1] > x[0] else clean
        nonzero = y[x > 0]
        rows.append({
            "method": method,
            "clean_macro_f1": clean,
            "normalized_auc": auc,
            "avg_drop": float(clean - nonzero.mean()) if len(nonzero) else 0.0,
            "final_drop": float(clean - y[-1]),
        })
    return pd.DataFrame(rows)


def plot_curve(summary_df, out_dir: Path):
    colors = {
        "pointnet_gru": "#4C78A8",
        "robhar_like": "#F58518",
        "pointnet_tcn": "#54A24B",
        "pct_gru": "#B279A2",
        "dgcnn_gru": "#72B7B2",
        "radar_stnet": "#D62728",
    }

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    for method in METHOD_ORDER:
        g = summary_df[summary_df["method"] == method].sort_values("level")
        if g.empty:
            continue

        x = g["level"].to_numpy(dtype=np.float64) * 100.0
        y = g["sample_macro_f1_mean"].to_numpy(dtype=np.float64) * 100.0
        std = g["sample_macro_f1_std"].fillna(0).to_numpy(dtype=np.float64) * 100.0

        ax.plot(
            x, y,
            marker="o",
            linewidth=2.4 if method == "radar_stnet" else 1.8,
            markersize=6 if method == "radar_stnet" else 5,
            color=colors.get(method),
            label=METHOD_LABELS.get(method, method),
        )
        ax.fill_between(x, y - std, y + std, color=colors.get(method), alpha=0.12)

    ax.set_title("Robustness Under Realistic Clutter Interference")
    ax.set_xlabel("Realistic Clutter Ratio (%)")
    ax.set_ylabel("Sample Macro-F1 (%)")
    ax.set_ylim(0, 101)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(ncol=2, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / "fig_realistic_clutter_macro_f1.png", dpi=300)
    fig.savefig(out_dir / "fig_realistic_clutter_macro_f1.pdf")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Realistic clutter robustness evaluation.")
    parser.add_argument("--dataset_path", default="Processed_Dataset_NPY")
    parser.add_argument("--checkpoint_root", default="experiment1_results")
    parser.add_argument("--output_dir", default="experiment2_results/realistic_clutter")
    parser.add_argument("--methods", nargs="+", default=METHOD_ORDER)
    parser.add_argument("--test_subjects", nargs="+", type=int, default=[7])
    parser.add_argument("--levels", nargs="+", default=["0", "0.02", "0.05", "0.1", "0.15", "0.2", "0.3"])
    parser.add_argument("--seeds", nargs="+", default=["0", "1", "2", "3", "4"])
    parser.add_argument("--mode", choices=["mixed", "static", "aid"], default="mixed")
    parser.add_argument("--seq_len", type=int, default=25)
    parser.add_argument("--n_points", type=int, default=128)
    parser.add_argument("--window_stride", type=int, default=1)
    parser.add_argument("--max_windows_per_file", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    levels = [float(v) for v in args.levels]
    seeds = [int(v) for v in args.seeds]

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

    for method, ckpt in checkpoint_map.items():
        print(f"\n========== Method: {method} ==========")
        model = load_model(method, ckpt, class_names, args.seq_len, device)

        for level in levels:
            level_seeds = [seeds[0]] if level == 0 else seeds
            for seed in level_seeds:
                print(f"[eval] method={method}, realistic_clutter={level}, seed={seed}, mode={args.mode}")

                dataset = RealisticClutterDataset(
                    records=test_records,
                    data_cfg=data_cfg,
                    seq_len=args.seq_len,
                    clutter_cfg=ClutterConfig(level=level, seed=seed, mode=args.mode),
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
                    "level": level,
                    "seed": seed,
                    "mode": args.mode,
                    **metrics,
                })

                pd.DataFrame(raw_rows).to_csv(out_dir / "realistic_clutter_raw.csv", index=False, encoding="utf-8-sig")

    raw_df = pd.DataFrame(raw_rows)
    summary_df = summarize(raw_df)
    auc_df = auc_summary(summary_df)

    raw_df.to_csv(out_dir / "realistic_clutter_raw.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "realistic_clutter_summary.csv", index=False, encoding="utf-8-sig")
    auc_df.to_csv(out_dir / "realistic_clutter_auc_summary.csv", index=False, encoding="utf-8-sig")
    plot_curve(summary_df, out_dir)

    config = vars(args)
    config["device"] = str(device)
    config["class_names"] = class_names
    config["checkpoint_map"] = {k: str(v) for k, v in checkpoint_map.items()}
    (out_dir / "realistic_clutter_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nDone. Results saved to: {out_dir.resolve()}")
    print(auc_df)


if __name__ == "__main__":
    main()