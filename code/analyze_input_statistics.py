"""Quantify input-statistics drift under Radar point-cloud degradations.

This script is training-free. It uses the same raw chunk files and degradation
constructors as experiment2, then reports per-channel mean/std/min/max shifts,
centroid displacement, and covariance-ellipse area change.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset_loader import DatasetConfig
from experiment1_action_classification import collect_records
from experiment2_degradation_robustness import DegradeConfig, DegradedWindowDataset
from experiment2_realistic_clutter import (
    ClutterConfig,
    apply_realistic_clutter,
    make_trial_profile,
)


CHANNELS = ["x", "y", "z", "I"]
STAT_NAMES = ["mean", "std", "min", "max"]


def parse_ints(values):
    return [int(v) for v in values]


def parse_floats(values):
    return [float(v) for v in values]


def stats(points):
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return {f"{s}_{c}": np.nan for s in STAT_NAMES for c in CHANNELS}
    out = {}
    for j, c in enumerate(CHANNELS):
        v = pts[:, j].astype(np.float64)
        out[f"mean_{c}"] = float(v.mean())
        out[f"std_{c}"] = float(v.std())
        out[f"min_{c}"] = float(v.min())
        out[f"max_{c}"] = float(v.max())
    return out


def centroid_and_area(points):
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or len(pts) < 2:
        return np.array([np.nan, np.nan, np.nan]), np.nan
    center = pts[:, :3].mean(axis=0)
    cov = np.cov(pts[:, :2].T)
    eig = np.maximum(np.linalg.eigvalsh(cov), 1e-12)
    # Area of the 2-sigma ellipse: pi * (2 sqrt(lambda1)) * (2 sqrt(lambda2)).
    area = float(4.0 * np.pi * np.sqrt(eig[0] * eig[1]))
    return center, area


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", default="Processed_Dataset_NPY")
    ap.add_argument("--output_dir", default="missing_input_statistics")
    ap.add_argument("--test_subjects", nargs="+", type=int, default=[7])
    ap.add_argument("--seq_len", type=int, default=25)
    ap.add_argument("--n_points", type=int, default=128)
    ap.add_argument("--window_stride", type=int, default=1)
    ap.add_argument("--max_windows_per_file", type=int, default=20)
    ap.add_argument("--seeds", nargs="+", default=["42", "43", "44", "45", "46"])
    ap.add_argument("--missing_levels", nargs="+", default=["0", "0.3", "0.7"])
    ap.add_argument("--noise_levels", nargs="+", default=["0", "0.05", "0.10"])
    ap.add_argument("--clutter_levels", nargs="+", default=["0", "0.15", "0.30"])
    ap.add_argument("--include_realistic", action="store_true")
    ap.add_argument("--realistic_mode", choices=["mixed", "static", "aid"], default="mixed")
    args = ap.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records, _ = collect_records(str(dataset_path))
    records = [r for r in records if r.subject in args.test_subjects]
    cfg = DatasetConfig(dataset_path=str(dataset_path), n_chunk_per_data=args.seq_len,
                        n_sample_per_chunk=args.n_points)
    base = DegradedWindowDataset(
        records, cfg, args.seq_len, DegradeConfig("clean", 0.0, 0),
        window_stride=args.window_stride, max_windows_per_file=args.max_windows_per_file,
    )
    seeds = parse_ints(args.seeds)
    plans = [("point_missing", "missing", parse_floats(args.missing_levels)),
             ("coordinate_noise", "noise", parse_floats(args.noise_levels)),
             ("synthetic_clutter", "clutter", parse_floats(args.clutter_levels))]

    raw_rows = []
    for item_idx, (file_idx, start, label) in enumerate(base.items):
        window = base.chunks_by_file[file_idx][start:start + args.seq_len]
        profile_cache = {}
        for degradation, kind, levels in plans:
            for level in levels:
                level_seeds = [seeds[0]] if level == 0 else seeds
                for seed in level_seeds:
                    for t, clean in enumerate(window):
                        clean = np.asarray(clean, dtype=np.float32)[:, :4]
                        clean_stats = stats(clean)
                        clean_center, clean_area = centroid_and_area(clean)
                        rng = np.random.default_rng(seed + item_idx * 1009 + t * 9173)
                        if level == 0:
                            degraded = clean
                        else:
                            # Use a dataset instance configured for the current
                            # degradation; the base dataset is intentionally clean.
                            if kind == "missing":
                                keep = max(1, int(round(len(clean) * (1.0 - level))))
                                degraded = clean[rng.choice(len(clean), size=keep, replace=False)] if len(clean) else clean
                            elif kind == "noise":
                                degraded = clean.copy()
                                if len(degraded):
                                    degraded[:, :3] += rng.normal(0.0, level, size=degraded[:, :3].shape).astype(np.float32)
                            elif kind == "clutter":
                                # Reuse the public degradation implementation.
                                from experiment2_degradation_robustness import make_clutter
                                degraded = make_clutter(clean, float(level), rng, cfg)
                            else:
                                degraded = clean
                        deg_stats = stats(degraded)
                        deg_center, deg_area = centroid_and_area(degraded)
                        row = {
                            "item": item_idx, "file": base.file_keys[file_idx], "label": int(label),
                            "time_step": t, "degradation": degradation, "level": level, "seed": seed,
                            "point_count_clean": len(clean), "point_count_degraded": len(degraded),
                            "centroid_shift": float(np.linalg.norm(deg_center - clean_center)),
                            "ellipse_area_clean": clean_area, "ellipse_area_degraded": deg_area,
                            "ellipse_area_change": (deg_area - clean_area) / max(abs(clean_area), 1e-12),
                        }
                        for s in STAT_NAMES:
                            for c in CHANNELS:
                                a, b = clean_stats[f"{s}_{c}"], deg_stats[f"{s}_{c}"]
                                row[f"{s}_{c}_relative_shift"] = (b - a) / max(abs(a), 1e-6)
                        raw_rows.append(row)

    # Structured clutter is generated by the same profile function, with static
    # points shared across time steps inside each trial.
    if args.include_realistic:
        for item_idx, (file_idx, start, label) in enumerate(base.items):
            window = base.chunks_by_file[file_idx][start:start + args.seq_len]
            for level in parse_floats(args.clutter_levels):
                for seed in seeds:
                    profile = make_trial_profile(file_idx, seed, cfg)
                    for t, clean in enumerate(window):
                        clean = np.asarray(clean, dtype=np.float32)[:, :4]
                        rng = np.random.default_rng(seed + item_idx * 1009 + t * 9173)
                        degraded = apply_realistic_clutter(
                            clean, level, rng, profile, start + t, cfg, args.realistic_mode
                        ) if level else clean
                        a = stats(clean)
                        b = stats(degraded)
                        cc, ca = centroid_and_area(clean)
                        dc, da = centroid_and_area(degraded)
                        row = {
                            "item": item_idx, "file": base.file_keys[file_idx], "label": int(label),
                            "time_step": t, "degradation": "structured_realistic_clutter", "level": level,
                            "seed": seed, "point_count_clean": len(clean), "point_count_degraded": len(degraded),
                            "centroid_shift": float(np.linalg.norm(dc - cc)), "ellipse_area_clean": ca,
                            "ellipse_area_degraded": da, "ellipse_area_change": (da - ca) / max(abs(ca), 1e-12),
                        }
                        for s in STAT_NAMES:
                            for c in CHANNELS:
                                row[f"{s}_{c}_relative_shift"] = (b[f"{s}_{c}"] - a[f"{s}_{c}"]) / max(abs(a[f"{s}_{c}"]), 1e-6)
                        raw_rows.append(row)

    write_csv(output_dir / "input_statistics_raw.csv", raw_rows)
    # Aggregate absolute relative shifts; this is the compact table used in the paper.
    grouped = {}
    for row in raw_rows:
        key = (row["degradation"], row["level"])
        grouped.setdefault(key, []).append(row)
    summary = []
    for (deg, level), rows in sorted(grouped.items()):
        out = {"degradation": deg, "level": level, "n": len(rows)}
        shifts = []
        for s in STAT_NAMES:
            for c in CHANNELS:
                vals = np.abs([r[f"{s}_{c}_relative_shift"] for r in rows if np.isfinite(r[f"{s}_{c}_relative_shift"])])
                out[f"abs_shift_{s}_{c}_mean"] = float(vals.mean()) if len(vals) else np.nan
                shifts.extend(vals.tolist())
        cent = np.asarray([r["centroid_shift"] for r in rows], dtype=float)
        area = np.asarray([abs(r["ellipse_area_change"]) for r in rows], dtype=float)
        out["centroid_shift_mean"] = float(np.nanmean(cent))
        out["ellipse_area_change_abs_mean"] = float(np.nanmean(area))
        out["overall_abs_stat_shift_mean"] = float(np.nanmean(shifts)) if shifts else np.nan
        summary.append(out)
    write_csv(output_dir / "input_statistics_summary.csv", summary)

    try:
        import matplotlib.pyplot as plt
        labels = [f"{r['degradation']}\n{r['level']}" for r in summary]
        values = [r["overall_abs_stat_shift_mean"] for r in summary]
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.bar(np.arange(len(values)), values, color="#4C78A8")
        ax.set_xticks(np.arange(len(values)), labels, rotation=30, ha="right")
        ax.set_ylabel("Mean absolute relative shift")
        ax.set_title("Input-statistics drift under radar-point-cloud degradation")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "fig_input_statistics_drift.png", dpi=300)
        fig.savefig(output_dir / "fig_input_statistics_drift.pdf")
        plt.close(fig)
    except ImportError:
        pass
    print(f"saved: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
