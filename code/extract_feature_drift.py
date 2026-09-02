"""Measure internal representation retention under radar degradations.

The script registers a hook on each model's temporal pooling layer, so the
reported vectors are the features immediately used by the classifier. Clean
and degraded windows are generated from the same raw window and seed.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset_loader import DatasetConfig, Transforms
from experiment1_action_classification import build_deep_model, collect_records
from experiment2_degradation_robustness import DegradeConfig, DegradedWindowDataset
from experiment2_realistic_clutter import apply_realistic_clutter, make_trial_profile

METHODS = ["pointnet_gru", "robhar_like", "pointnet_tcn", "pct_gru", "dgcnn_gru", "radar_stnet"]


def load_model(method, checkpoint, class_names, seq_len, device):
    try:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location=device)
    model = build_deep_model(method, len(class_names), seq_len).to(device)
    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


class PairDataset(Dataset):
    def __init__(self, records, cfg, seq_len, kind, level, seed, stride, max_windows, realistic_mode):
        self.base = DegradedWindowDataset(records, cfg, seq_len, DegradeConfig("clean", 0, seed), stride, max_windows)
        self.degrader = DegradedWindowDataset(records, cfg, seq_len, DegradeConfig(kind, level, seed), stride, max_windows)
        self.cfg, self.seq_len, self.kind, self.level, self.seed = cfg, seq_len, kind, level, seed
        self.realistic_mode = realistic_mode
        self.profiles = [make_trial_profile(i, seed, cfg) for i in range(len(self.base.chunks_by_file))]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        fi, start, label = self.base.items[idx]
        chunks = self.base.chunks_by_file[fi]
        clean_out, deg_out = [], []
        tr = Transforms(self.cfg, augment=False, deterministic=True)
        for local_i, pts in enumerate(chunks[start:start + self.seq_len]):
            clean = np.asarray(pts, dtype=np.float32)[:, :4]
            rng = np.random.default_rng(self.seed + idx * 1009 + local_i * 9173)
            if self.kind == "structured_realistic_clutter" and self.level > 0:
                deg = apply_realistic_clutter(clean, self.level, rng, self.profiles[fi], start + local_i, self.cfg, self.realistic_mode)
            elif self.level > 0:
                deg = self.degrader.degrade_points(clean, rng)
            else:
                deg = clean
            clean_out.append(tr.normalize(tr.resampler.resample(clean)))
            deg_out.append(tr.normalize(tr.resampler.resample(deg)))
        return torch.tensor(np.stack(clean_out), dtype=torch.float32), torch.tensor(np.stack(deg_out), dtype=torch.float32), label


def hook_pool(model):
    holder = {}
    module = getattr(model, "pool", None)
    if module is None:
        raise ValueError("model has no temporal pool module")
    def capture(_m, _inp, output):
        holder["value"] = output.detach()
    handle = module.register_forward_hook(capture)
    return holder, handle


def auc(levels, values):
    order = np.argsort(levels)
    x, y = np.asarray(levels)[order], np.asarray(values)[order]
    integ = np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x)
    return float(integ / (x[-1] - x[0])) if len(x) > 1 and x[-1] > x[0] else float(y[0])


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Internal feature drift and retention AUC")
    ap.add_argument("--dataset_path", default="Processed_Dataset_NPY")
    ap.add_argument("--checkpoint_root", default="experiment1_results")
    ap.add_argument("--output_dir", default="missing_feature_drift")
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--test_subjects", nargs="+", type=int, default=[7])
    ap.add_argument("--levels", nargs="+", default=["0", "0.1", "0.3", "0.5", "0.7"])
    ap.add_argument("--seeds", nargs="+", default=["42", "43", "44"])
    ap.add_argument("--degradations", nargs="+", choices=["missing", "noise", "clutter", "structured_realistic_clutter"], default=["missing", "noise", "clutter"])
    ap.add_argument("--realistic_mode", choices=["mixed", "static", "aid"], default="mixed")
    ap.add_argument("--seq_len", type=int, default=25); ap.add_argument("--n_points", type=int, default=128)
    ap.add_argument("--window_stride", type=int, default=1); ap.add_argument("--max_windows_per_file", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64); ap.add_argument("--num_workers", type=int, default=0); ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    root = Path(args.dataset_path); root = root if root.is_absolute() else PROJECT_ROOT / root
    out = Path(args.output_dir); out = out if out.is_absolute() else PROJECT_ROOT / out; out.mkdir(parents=True, exist_ok=True)
    checkpoint_root = Path(args.checkpoint_root)
    if not checkpoint_root.is_absolute():
        package_candidate = PROJECT_ROOT / checkpoint_root
        checkpoint_root = package_candidate if package_candidate.exists() else Path.cwd() / checkpoint_root
    records, classes = collect_records(str(root)); records = [r for r in records if r.subject in args.test_subjects]
    cfg = DatasetConfig(dataset_path=str(root), n_chunk_per_data=args.seq_len, n_sample_per_chunk=args.n_points)
    levels, seeds = [float(x) for x in args.levels], [int(x) for x in args.seeds]
    raw, summary = [], []
    for method in args.methods:
        candidates = sorted(checkpoint_root.rglob(f"*{method}.pth"))
        if not candidates: print(f"[skip] checkpoint not found: {method}"); continue
        ckpt = next((p for p in candidates if "train_p1-p5_val_p6_test_p7" in p.name), candidates[-1])
        model = load_model(method, ckpt, classes, args.seq_len, device); holder, handle = hook_pool(model)
        for degradation in args.degradations:
            for level in levels:
                run_seeds = [seeds[0]] if level == 0 else seeds
                vals = []
                for seed in run_seeds:
                    ds = PairDataset(records, cfg, args.seq_len, degradation, level, seed, args.window_stride, args.max_windows_per_file, args.realistic_mode)
                    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
                    sims = []
                    for clean, deg, _ in loader:
                        clean, deg = clean.to(device), deg.to(device)
                        model(clean); fc = holder["value"].flatten(1)
                        model(deg); fd = holder["value"].flatten(1)
                        sims.extend(F.cosine_similarity(fc, fd, dim=1).cpu().numpy().tolist())
                    val = float(np.mean(sims)) if sims else math.nan; vals.append(val)
                    raw.append({"method": method, "degradation": degradation, "level": level, "seed": seed, "cosine_similarity_mean": val, "n_windows": len(sims)})
                summary.append({"method": method, "degradation": degradation, "level": level, "retention_mean": float(np.nanmean(vals)), "retention_std": float(np.nanstd(vals)), "runs": len(vals)})
        handle.remove()
    write_csv(out / "feature_drift_raw.csv", raw); write_csv(out / "feature_drift_summary.csv", summary)
    auc_rows = []
    for method in sorted({r["method"] for r in summary}):
        for deg in sorted({r["degradation"] for r in summary if r["method"] == method}):
            g = [r for r in summary if r["method"] == method and r["degradation"] == deg]
            auc_rows.append({"method": method, "degradation": deg, "retention_auc": auc([r["level"] for r in g], [r["retention_mean"] for r in g])})
    write_csv(out / "feature_retention_auc.csv", auc_rows)
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        for method in sorted({r["method"] for r in summary}):
            g = [r for r in summary if r["method"] == method and r["degradation"] == args.degradations[0]]
            g.sort(key=lambda r: r["level"]); ax.plot([r["level"] for r in g], [r["retention_mean"] for r in g], marker="o", label=method)
        ax.set(xlabel="Degradation level", ylabel="Cosine feature retention", ylim=(0, 1.02)); ax.grid(alpha=.3); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "fig_feature_retention.png", dpi=300); fig.savefig(out / "fig_feature_retention.pdf"); plt.close(fig)
    except ImportError: pass
    print(f"saved: {out.resolve()}")


if __name__ == "__main__": main()
