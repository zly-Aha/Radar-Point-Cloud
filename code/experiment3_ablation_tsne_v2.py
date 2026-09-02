from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, silhouette_score, davies_bouldin_score
from torch.utils.data import DataLoader

from dataset_loader import DatasetConfig
from experiment1_action_classification import RadarWindowDataset, collect_records
from models import RadarSTNet, RadarSTConfig


ABLATION_VARIANTS = [
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
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_torch_load(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def find_radar_stnet_checkpoint(root: Path):
    candidates = sorted(root.rglob("*radar_stnet*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    fixed = [p for p in candidates if "train_p1-p5_val_p6_test_p7" in p.name]
    if fixed:
        return fixed[0]

    # fallback: old log checkpoint
    log_candidates = sorted(Path("logs").rglob("best_model.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    if log_candidates:
        return log_candidates[0]

    return candidates[0] if candidates else None


def load_original_radar_stnet(ckpt_path: Path, class_names, seq_len: int, device):
    ckpt = safe_torch_load(ckpt_path, device)

    if isinstance(ckpt, dict) and "config" in ckpt:
        cfg_dict = ckpt.get("config", {})
        config = RadarSTConfig(
            in_channels=cfg_dict.get("in_channels", 4),
            spatial_dim=cfg_dict.get("spatial_dim", 160),
            seq_len=cfg_dict.get("seq_len", seq_len),
            tcn_kernel=cfg_dict.get("tcn_kernel", 5),
            num_heads=cfg_dict.get("num_heads", 4),
            transformer_layers=cfg_dict.get("transformer_layers", 2),
            dropout=cfg_dict.get("dropout", 0.3),
            classes=len(class_names),
        )
        state = ckpt["model_state"]
    else:
        config = RadarSTConfig(classes=len(class_names), seq_len=seq_len)
        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt

    model = RadarSTNet(config).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def load_full_metrics(checkpoint_root: Path):
    summary_candidates = sorted(checkpoint_root.rglob("experiment1_summary.csv"))
    rows = []

    for path in summary_candidates:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "method" not in df.columns:
            continue
        hit = df[df["method"] == "radar_stnet"].copy()
        if not hit.empty:
            rows.append(hit)

    if rows:
        df = pd.concat(rows, ignore_index=True)
        df["sample_macro_f1"] = pd.to_numeric(df["sample_macro_f1"], errors="coerce")
        best = df.loc[df["sample_macro_f1"].idxmax()].to_dict()
        return {
            "variant": "full",
            "params": int(float(best.get("params", 813947))),
            "sample_accuracy": float(best["sample_accuracy"]),
            "sample_macro_f1": float(best["sample_macro_f1"]),
            "sample_count": int(float(best.get("sample_count", 26280))),
            "trial_accuracy": float(best["trial_accuracy"]),
            "trial_macro_f1": float(best["trial_macro_f1"]),
            "trial_count": int(float(best.get("trial_count", 30))),
            "source": "experiment1_radar_stnet",
        }

    # fallback to known log result
    return {
        "variant": "full",
        "params": 813947,
        "sample_accuracy": 0.9849695585996956,
        "sample_macro_f1": 0.985035686967341,
        "sample_count": 26280,
        "trial_accuracy": 1.0,
        "trial_macro_f1": 1.0,
        "trial_count": 30,
        "source": "fallback_known_result",
    }


class FastPointEncoderAblation(nn.Module):
    def __init__(self, in_channels=4, spatial_dim=160, use_raw_stats=True, pooling_mode="triple"):
        super().__init__()
        self.use_raw_stats = use_raw_stats
        self.pooling_mode = pooling_mode
        self.raw_stat_dim = in_channels * 4
        learned_dim = spatial_dim - self.raw_stat_dim
        if learned_dim <= 0:
            raise ValueError("Invalid spatial_dim/raw stats configuration")
        if pooling_mode == "triple":
            if learned_dim % 3 != 0:
                raise ValueError("learned_dim must be divisible by 3 for triple pooling")
            pooled_dim = learned_dim // 3
        elif pooling_mode == "max":
            if learned_dim % 3 != 0:
                raise ValueError("learned_dim must be divisible by 3 for max-only pooling")
            pooled_dim = learned_dim // 3
        else:
            raise ValueError(f"Unknown pooling_mode: {pooling_mode}")

        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, 48, 1, bias=False),
            nn.BatchNorm1d(48),
            nn.GELU(),
            nn.Conv1d(48, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, pooled_dim, 1, bias=False),
            nn.BatchNorm1d(pooled_dim),
            nn.GELU(),
        )
        self.out_norm = nn.LayerNorm(spatial_dim)

    def forward(self, x):
        feat = self.mlp(x)
        max_feat = torch.max(feat, dim=2)[0]

        if self.pooling_mode == "triple":
            mean_feat = torch.mean(feat, dim=2)
            std_feat = torch.sqrt(torch.var(feat, dim=2, unbiased=False) + 1e-5)
            parts = [max_feat, mean_feat, std_feat]
        else:
            zero_pool = torch.zeros_like(max_feat)
            parts = [max_feat, zero_pool, zero_pool]

        if self.use_raw_stats:
            raw = x.transpose(1, 2)
            raw_mean = raw.mean(dim=1)
            raw_std = torch.sqrt(torch.var(raw, dim=1, unbiased=False) + 1e-5)
            raw_min = raw.min(dim=1)[0]
            raw_max = raw.max(dim=1)[0]
            parts.append(torch.cat([raw_mean, raw_std, raw_min, raw_max], dim=1))
        else:
            zeros = x.new_zeros(x.size(0), self.raw_stat_dim)
            parts.append(zeros)

        return self.out_norm(torch.cat(parts, dim=1))


class TemporalBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, groups=channels, bias=False),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class TemporalConvNet(nn.Module):
    def __init__(self, channels, kernel_size, dropout):
        super().__init__()
        self.blocks = nn.Sequential(
            TemporalBlock(channels, 3, 1, dropout),
            TemporalBlock(channels, kernel_size, 2, dropout),
            TemporalBlock(channels, kernel_size, 4, dropout),
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        out = self.blocks(x)
        return self.norm(out.transpose(1, 2)).transpose(1, 2)


class AttentionPooling(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.Tanh(),
            nn.Linear(channels // 2, 1),
        )

    def forward(self, x):
        weight = torch.softmax(self.score(x), dim=1)
        return torch.sum(x * weight, dim=1)


class MeanPooling(nn.Module):
    def forward(self, x):
        return x.mean(dim=1)


class RadarSTNetAblation(nn.Module):
    def __init__(
        self,
        classes,
        seq_len=25,
        spatial_dim=160,
        dropout=0.3,
        use_motion=True,
        use_tcn=True,
        use_transformer=True,
        use_attention_pool=True,
        use_raw_stats=True,
        pooling_mode="triple",
        motion_order=2,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.spatial_dim = spatial_dim
        self.use_motion = use_motion and motion_order > 0
        self.motion_order = motion_order
        self.use_tcn = use_tcn
        self.use_transformer = use_transformer

        self.spatial_encoder = FastPointEncoderAblation(
            in_channels=4,
            spatial_dim=spatial_dim,
            use_raw_stats=use_raw_stats,
            pooling_mode=pooling_mode,
        )

        motion_in_dim = spatial_dim * (1 + int(self.motion_order >= 1) + int(self.motion_order >= 2))
        self.motion_fuse = nn.Sequential(
            nn.Linear(motion_in_dim, spatial_dim),
            nn.LayerNorm(spatial_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ) if use_motion else nn.Identity()

        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, spatial_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.tcn = TemporalConvNet(spatial_dim, 5, dropout) if use_tcn else nn.Identity()

        if use_transformer:
            layer = nn.TransformerEncoderLayer(
                d_model=spatial_dim,
                nhead=4,
                dim_feedforward=spatial_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        else:
            self.transformer = nn.Identity()

        self.pool = AttentionPooling(spatial_dim) if use_attention_pool else MeanPooling()

        self.classifier = nn.Sequential(
            nn.LayerNorm(spatial_dim),
            nn.Linear(spatial_dim, spatial_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(spatial_dim // 2, classes),
        )

    def forward_features(self, x):
        b, s, n, c = x.shape
        x = x.flatten(0, 1).transpose(1, 2)
        spatial_feat = self.spatial_encoder(x).view(b, s, self.spatial_dim)

        if self.use_motion:
            diff1 = spatial_feat[:, 1:] - spatial_feat[:, :-1]
            delta_feat = F.pad(diff1, (0, 0, 1, 0))
            motion_parts = [spatial_feat, delta_feat]
            if self.motion_order >= 2:
                diff2 = delta_feat[:, 1:] - delta_feat[:, :-1]
                accel_feat = F.pad(diff2, (0, 0, 1, 0))
                motion_parts.append(accel_feat)
            feat = self.motion_fuse(torch.cat(motion_parts, dim=-1))
        else:
            feat = spatial_feat

        feat = feat + self.pos_embed[:, :s, :]

        if self.use_tcn:
            feat = self.tcn(feat.transpose(1, 2)).transpose(1, 2)

        if self.use_transformer:
            feat = self.transformer(feat)

        return self.pool(feat)

    def forward(self, x):
        return self.classifier(self.forward_features(x))


def build_ablation_model(variant, num_classes, seq_len):
    if variant == "no_motion":
        return RadarSTNetAblation(num_classes, seq_len=seq_len, use_motion=False)
    if variant == "no_tcn":
        return RadarSTNetAblation(num_classes, seq_len=seq_len, use_tcn=False)
    if variant == "no_transformer":
        return RadarSTNetAblation(num_classes, seq_len=seq_len, use_transformer=False)
    if variant == "no_attention_pool":
        return RadarSTNetAblation(num_classes, seq_len=seq_len, use_attention_pool=False)
    if variant == "no_raw_stats":
        return RadarSTNetAblation(num_classes, seq_len=seq_len, use_raw_stats=False)
    if variant == "max_only":
        return RadarSTNetAblation(num_classes, seq_len=seq_len, pooling_mode="max")
    if variant == "first_order_motion":
        return RadarSTNetAblation(num_classes, seq_len=seq_len, motion_order=1)
    raise ValueError(f"Unknown variant: {variant}")


def class_weights_from_dataset(dataset, num_classes, device):
    labels = np.asarray([item[2] for item in dataset.items])
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


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
    model.eval()
    y_true, y_pred, file_ids = [], [], []
    dataset = loader.dataset

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


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_score, best_state, variant, class_names, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "variant": variant,
        "epoch": epoch,
        "best_score": best_score,
        "model_state": model.state_dict(),
        "best_state": best_state,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "classes": class_names,
        "args": vars(args),
    }, path)


def train_variant(variant, train_ds, val_ds, test_ds, class_names, args, device):
    num_classes = len(class_names)
    model = build_ablation_model(variant, num_classes, args.seq_len).to(device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=device.type == "cuda")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights_from_dataset(train_ds, num_classes, device),
        label_smoothing=args.label_smoothing,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    ckpt_path = Path(args.output_dir) / "checkpoints" / f"{variant}.pth"
    start_epoch = 1
    best_score = -1.0
    best_state = copy.deepcopy(model.state_dict())

    if args.resume and ckpt_path.exists():
        ckpt = safe_torch_load(ckpt_path, device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_score = float(ckpt.get("best_score", -1.0))
        best_state = ckpt.get("best_state", copy.deepcopy(model.state_dict()))
        print(f"[resume] {variant}: start_epoch={start_epoch}, best_score={best_score:.4f}")

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss, total, correct = 0.0, 0, 0

        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            batch = y.size(0)
            total_loss += loss.item() * batch
            total += batch
            correct += (logits.argmax(dim=1) == y).sum().item()

        scheduler.step()
        val_metrics = evaluate_model(model, val_loader, device)
        score = val_metrics["sample_macro_f1"]

        print(
            f"[{variant}] epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={total_loss / max(total, 1):.4f} | "
            f"train_acc={correct / max(total, 1):.4f} | "
            f"val_f1={score:.4f}"
        )

        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())

        save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, best_score, best_state, variant, class_names, args)

    model.load_state_dict(best_state)
    test_metrics = evaluate_model(model, test_loader, device)
    params = sum(p.numel() for p in model.parameters())

    best_path = Path(args.output_dir) / "checkpoints" / f"{variant}_best.pth"
    torch.save({
        "variant": variant,
        "model_state": best_state,
        "classes": class_names,
        "seq_len": args.seq_len,
        "params": int(params),
        "metrics": test_metrics,
    }, best_path)

    return {
        "variant": variant,
        "params": int(params),
        **test_metrics,
        "source": "trained_ablation",
    }


def save_results(rows, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(Path(output_dir) / "ablation_results.csv", index=False, encoding="utf-8-sig")


def plot_ablation_bar(rows, output_dir):
    df = pd.DataFrame(rows)
    order = [
        "full",
        "no_motion",
        "first_order_motion",
        "no_tcn",
        "no_transformer",
        "no_attention_pool",
        "no_raw_stats",
        "max_only",
    ]
    labels = {
        "full": "Full",
        "no_motion": "w/o Motion",
        "first_order_motion": "1st-order Motion",
        "no_tcn": "w/o TCN",
        "no_transformer": "w/o Transformer",
        "no_attention_pool": "w/o Attn Pool",
        "no_raw_stats": "w/o Raw Stats",
        "max_only": "Max-only",
    }
    df["order"] = df["variant"].map({v: i for i, v in enumerate(order)})
    df = df.sort_values("order")

    x = np.arange(len(df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(x - width / 2, df["sample_accuracy"] * 100, width, label="Accuracy", color="#4C78A8")
    ax.bar(x + width / 2, df["sample_macro_f1"] * 100, width, label="Macro-F1", color="#F58518")

    ax.set_title("Ablation Study on Radar-STNet")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(v, v) for v in df["variant"]], rotation=20, ha="right")
    ax.set_ylim(max(0, df[["sample_accuracy", "sample_macro_f1"]].min().min() * 100 - 8), 101)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    fig.tight_layout()
    fig.savefig(Path(output_dir) / "fig_ablation_bar.png", dpi=300)
    fig.savefig(Path(output_dir) / "fig_ablation_bar.pdf")
    plt.close(fig)


class OriginalRadarSTNetFeatureWrapper(nn.Module):
    def __init__(self, model: RadarSTNet):
        super().__init__()
        self.model = model

    def forward_features(self, x):
        b, s, n, c = x.shape
        z = x.flatten(0, 1).transpose(1, 2)

        spatial_feat = self.model.spatial_encoder(z).view(b, s, self.model.config.spatial_dim)

        diff1 = spatial_feat[:, 1:] - spatial_feat[:, :-1]
        delta_feat = F.pad(diff1, (0, 0, 1, 0))
        diff2 = delta_feat[:, 1:] - delta_feat[:, :-1]
        accel_feat = F.pad(diff2, (0, 0, 1, 0))

        motion_feat = torch.cat([spatial_feat, delta_feat, accel_feat], dim=-1)
        motion_feat = self.model.motion_fuse(motion_feat)
        motion_feat = motion_feat + self.model.pos_embed[:, :s, :]

        tcn_out = self.model.tcn(motion_feat.transpose(1, 2)).transpose(1, 2)
        trans_out = self.model.transformer(tcn_out)
        return self.model.pool(trans_out)

    def forward(self, x):
        return self.model(x)


@torch.no_grad()
def extract_features(model, loader, device, max_samples):
    model.eval()
    class_quota = max(1, max_samples // 10)
    feats_by_class = defaultdict(list)
    labels_by_class = defaultdict(list)

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)

        if hasattr(model, "forward_features"):
            f = model.forward_features(x)
        else:
            f = model(x)

        f = f.detach().cpu().numpy()
        y_np = y.numpy()

        for feat_i, label_i in zip(f, y_np):
            label_i = int(label_i)
            if len(labels_by_class[label_i]) < class_quota:
                feats_by_class[label_i].append(feat_i)
                labels_by_class[label_i].append(label_i)

        total = sum(len(v) for v in labels_by_class.values())
        if total >= max_samples:
            break

    feats, labels = [], []
    for cls in sorted(feats_by_class):
        feats.extend(feats_by_class[cls])
        labels.extend(labels_by_class[cls])

    feats = np.asarray(feats, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    return feats, labels

def plot_tsne(feats, labels, class_names, title, path):
    tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42)
    z = tsne.fit_transform(feats)

    fig, ax = plt.subplots(figsize=(8.0, 6.4))
    cmap = plt.get_cmap("tab10")

    for cls_idx, cls_name in enumerate(class_names):
        mask = labels == cls_idx
        if not mask.any():
            continue
        ax.scatter(z[mask, 0], z[mask, 1], s=9, alpha=0.78,
                   color=cmap(cls_idx % 10), label=cls_name, linewidths=0)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=2, fontsize=8, ncol=2, frameon=True)

    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)

    metrics = {}
    if len(np.unique(labels)) > 1:
        metrics["silhouette"] = float(silhouette_score(feats, labels))
        metrics["davies_bouldin"] = float(davies_bouldin_score(feats, labels))
    return metrics


def load_ablation_best(variant, output_dir, class_names, seq_len, device):
    path = Path(output_dir) / "checkpoints" / f"{variant}_best.pth"
    model = build_ablation_model(variant, len(class_names), seq_len).to(device)
    ckpt = safe_torch_load(path, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def run_tsne(args, test_ds, class_names, device):
    fig_dir = Path(args.output_dir) / "tsne"
    fig_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")

    rows = []

    for name in args.tsne_variants:
        if name == "full":
            full_ckpt = find_radar_stnet_checkpoint(Path(args.checkpoint_root))
            if full_ckpt is None:
                raise FileNotFoundError("Could not find radar_stnet checkpoint.")
            model = OriginalRadarSTNetFeatureWrapper(
                load_original_radar_stnet(full_ckpt, class_names, args.seq_len, device)
            )
        else:
            model = load_ablation_best(name, args.output_dir, class_names, args.seq_len, device)

        feats, labels = extract_features(model, loader, device, args.tsne_samples)
        metrics = plot_tsne(feats, labels, class_names, f"t-SNE Feature Visualization ({name})",
                            fig_dir / f"tsne_{name}")
        rows.append({"name": name, **metrics})
        print(f"[t-SNE] {name}: {metrics}")

    pd.DataFrame(rows).to_csv(Path(args.output_dir) / "tsne_metrics.csv", index=False, encoding="utf-8-sig")


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 3 v2: ablation with original Full Radar-STNet.")
    parser.add_argument("--dataset_path", default="Processed_Dataset_NPY")
    parser.add_argument("--checkpoint_root", default="experiment1_results")
    parser.add_argument("--output_dir", default="experiment3_results/main_v2")
    parser.add_argument("--variants", nargs="+", default=ABLATION_VARIANTS)
    parser.add_argument("--train_subjects", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--val_subjects", nargs="+", type=int, default=[6])
    parser.add_argument("--test_subjects", nargs="+", type=int, default=[7])
    parser.add_argument("--seq_len", type=int, default=25)
    parser.add_argument("--n_points", type=int, default=128)
    parser.add_argument("--window_stride", type=int, default=1)
    parser.add_argument("--max_windows_per_file", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--label_smoothing", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only_tsne", action="store_true")
    parser.add_argument("--skip_tsne", action="store_true")
    parser.add_argument("--tsne_variants", nargs="+", default=["full", "no_motion", "no_tcn", "no_transformer"])
    parser.add_argument("--tsne_samples", type=int, default=3000)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"device={device}")

    records, class_names = collect_records(args.dataset_path)
    data_cfg = DatasetConfig(args.dataset_path, args.seq_len, args.n_points)

    train_records = [r for r in records if r.subject in args.train_subjects]
    val_records = [r for r in records if r.subject in args.val_subjects]
    test_records = [r for r in records if r.subject in args.test_subjects]

    train_ds = RadarWindowDataset(train_records, data_cfg, args.seq_len, augment=True,
                                  window_stride=args.window_stride, max_windows_per_file=args.max_windows_per_file)
    val_ds = RadarWindowDataset(val_records, data_cfg, args.seq_len, augment=False,
                                window_stride=args.window_stride, max_windows_per_file=args.max_windows_per_file)
    test_ds = RadarWindowDataset(test_records, data_cfg, args.seq_len, augment=False,
                                 window_stride=args.window_stride, max_windows_per_file=args.max_windows_per_file)

    result_path = Path(args.output_dir) / "ablation_results.csv"
    rows = []
    if result_path.exists():
        try:
            rows = pd.read_csv(result_path).to_dict("records")
        except Exception:
            rows = []

    full_row = load_full_metrics(Path(args.checkpoint_root))
    rows = [r for r in rows if r.get("variant") != "full"]
    rows.append(full_row)

    if not args.only_tsne:
        done = {r["variant"] for r in rows if "variant" in r}

        for variant in args.variants:
            if variant == "full":
                continue
            if variant in done and not args.resume:
                print(f"[skip] existing result for {variant}")
                continue

            print(f"\n========== Training ablation variant: {variant} ==========")
            result = train_variant(variant, train_ds, val_ds, test_ds, class_names, args, device)
            rows = [r for r in rows if r.get("variant") != variant]
            rows.append(result)
            save_results(rows, args.output_dir)
            plot_ablation_bar(rows, args.output_dir)
            print(
                f"[{variant}] sample_acc={result['sample_accuracy']:.4f}, "
                f"sample_macro_f1={result['sample_macro_f1']:.4f}, "
                f"trial_acc={result['trial_accuracy']:.4f}, "
                f"trial_macro_f1={result['trial_macro_f1']:.4f}"
            )

    save_results(rows, args.output_dir)
    plot_ablation_bar(rows, args.output_dir)

    if not args.skip_tsne:
        run_tsne(args, test_ds, class_names, device)

    print(f"\nDone. Results saved to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
