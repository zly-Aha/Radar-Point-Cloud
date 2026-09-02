from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Dataset

from dataset_loader import DatasetConfig, Transforms
from models import RadarSTConfig, RadarSTNet


TRADITIONAL_METHODS = {"stats_svm", "stats_rf", "stats_knn", "stats_lr"}
DEEP_METHODS = {
    "pointnet_gru",
    "pointnet_tcn",
    "robhar_like",
    "pct_gru",
    "dgcnn_gru",
    "radar_stnet",
}
DEFAULT_METHODS = ["stats_svm", "stats_rf", "stats_knn", "pointnet_gru", "robhar_like", "radar_stnet"]
ALL_METHODS = ["stats_svm", "stats_rf", "stats_knn", "stats_lr", "pointnet_gru", "pointnet_tcn",
               "robhar_like", "pct_gru", "dgcnn_gru", "radar_stnet"]


@dataclass
class FileRecord:
    path: Path
    label_name: str
    label: int
    subject: int
    source_split: str


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_subject_id(path: Path) -> int:
    token = path.stem.split("_")[0]
    digits = "".join(ch for ch in token if ch.isdigit())
    return int(digits) if digits else -1


def collect_records(dataset_path: str):
    base = Path(dataset_path)
    label_names = set()

    for split_name in ["Train", "Ver", "Test"]:
        split_dir = base / split_name
        if not split_dir.exists():
            continue
        for action_dir in split_dir.iterdir():
            if action_dir.is_dir():
                label_names.add(action_dir.name)

    class_names = sorted(label_names)
    label2idx = {name: idx for idx, name in enumerate(class_names)}
    records = []

    for split_name in ["Train", "Ver", "Test"]:
        split_dir = base / split_name
        if not split_dir.exists():
            continue
        for action_dir in sorted(split_dir.iterdir()):
            if not action_dir.is_dir():
                continue
            for npy_path in sorted(action_dir.glob("*.npy")):
                records.append(FileRecord(
                    path=npy_path,
                    label_name=action_dir.name,
                    label=label2idx[action_dir.name],
                    subject=parse_subject_id(npy_path),
                    source_split=split_name,
                ))

    if not records:
        raise FileNotFoundError(f"没有在 {dataset_path} 下找到 Train/Ver/Test 数据。")

    return records, class_names


class RadarWindowDataset(Dataset):
    def __init__(
        self,
        records,
        data_cfg: DatasetConfig,
        seq_len: int,
        augment: bool,
        window_stride: int = 1,
        max_windows_per_file: int = 0,
    ):
        self.seq_len = seq_len
        self.transform = Transforms(data_cfg, augment=augment, deterministic=not augment)
        self.chunks_by_file = []
        self.file_keys = []
        self.file_records = []
        self.items = []

        for rec in records:
            chunks = np.load(rec.path, allow_pickle=True)
            if len(chunks) < seq_len:
                continue

            local_file_idx = len(self.chunks_by_file)
            self.chunks_by_file.append(chunks)
            self.file_records.append(rec)
            self.file_keys.append(f"p{rec.subject}:{rec.label_name}:{rec.path.name}")

            starts = list(range(0, len(chunks) - seq_len + 1, window_stride))
            if max_windows_per_file > 0 and len(starts) > max_windows_per_file:
                rng = np.random.default_rng(2026 + local_file_idx)
                starts = sorted(rng.choice(starts, size=max_windows_per_file, replace=False).tolist())

            for start in starts:
                self.items.append((local_file_idx, start, rec.label))

        if not self.items:
            raise RuntimeError("当前划分没有可用滑窗样本，请检查 seq_len 或数据路径。")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        file_idx, start, label = self.items[idx]
        chunks = self.chunks_by_file[file_idx]
        window = list(chunks[start:start + self.seq_len])
        x = self.transform.transform(window)
        return x, torch.tensor(label, dtype=torch.long), torch.tensor(file_idx, dtype=torch.long)


def frame_stats(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 4 or len(points) == 0:
        return np.zeros(21, dtype=np.float32)

    pts = points[:, :4]
    count = np.array([len(pts) / 128.0], dtype=np.float32)
    mean = pts.mean(axis=0)
    std = pts.std(axis=0)
    min_v = pts.min(axis=0)
    max_v = pts.max(axis=0)
    span_xyz = max_v[:3] - min_v[:3]
    intensity_sum = np.array([pts[:, 3].sum() / 60.0], dtype=np.float32)

    return np.concatenate([count, mean, std, min_v, max_v, span_xyz, intensity_sum]).astype(np.float32)


def handcrafted_feature(window) -> np.ndarray:
    per_frame = np.stack([frame_stats(chunk) for chunk in window], axis=0)
    base = np.concatenate([
        per_frame.mean(axis=0),
        per_frame.std(axis=0),
        per_frame.min(axis=0),
        per_frame.max(axis=0),
    ])

    centroids = per_frame[:, 1:4]
    velocity = np.diff(centroids, axis=0)
    if len(velocity) == 0:
        motion = np.zeros(9, dtype=np.float32)
    else:
        motion = np.concatenate([
            velocity.mean(axis=0),
            velocity.std(axis=0),
            np.abs(velocity).max(axis=0),
        ])

    counts = per_frame[:, 0]
    count_summary = np.array([counts.mean(), counts.std(), counts.min(), counts.max()], dtype=np.float32)

    return np.concatenate([base, motion, count_summary]).astype(np.float32)


def build_feature_matrix(dataset: RadarWindowDataset):
    x_list, y_list, file_ids = [], [], []
    for file_idx, start, label in dataset.items:
        chunks = dataset.chunks_by_file[file_idx]
        window = chunks[start:start + dataset.seq_len]
        x_list.append(handcrafted_feature(window))
        y_list.append(label)
        file_ids.append(dataset.file_keys[file_idx])
    return np.vstack(x_list), np.asarray(y_list), np.asarray(file_ids)


class AttentionPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, max(dim // 2, 16)),
            nn.Tanh(),
            nn.Linear(max(dim // 2, 16), 1),
        )

    def forward(self, x):
        weight = torch.softmax(self.score(x), dim=1)
        return torch.sum(x * weight, dim=1)


class FramePointNetEncoder(nn.Module):
    def __init__(self, in_channels=4, out_dim=128, width=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, width, 1, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.Conv1d(width, width * 2, 1, bias=False),
            nn.BatchNorm1d(width * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(width * 2, out_dim, 1, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        b, s, n, c = x.shape
        x = x.reshape(b * s, n, c).transpose(1, 2).contiguous()
        feat = self.net(x)
        pooled = torch.cat([feat.max(dim=2).values, feat.mean(dim=2)], dim=1)
        return self.fuse(pooled).view(b, s, -1)


class PointNetRNNClassifier(nn.Module):
    def __init__(self, num_classes, spatial_dim=128, rnn_hidden=128, rnn_type="gru", dropout=0.3):
        super().__init__()
        self.encoder = FramePointNetEncoder(out_dim=spatial_dim)
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=spatial_dim,
            hidden_size=rnn_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        temporal_dim = rnn_hidden * 2
        self.pool = AttentionPooling(temporal_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(temporal_dim),
            nn.Dropout(dropout),
            nn.Linear(temporal_dim, num_classes),
        )

    def forward(self, x):
        feat = self.encoder(x)
        out, _ = self.rnn(feat)
        return self.classifier(self.pool(out))


class TemporalBlock(nn.Module):
    def __init__(self, dim, kernel=5, dilation=1, dropout=0.3):
        super().__init__()
        pad = (kernel // 2) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel, padding=pad, dilation=dilation, groups=dim, bias=False),
            nn.Conv1d(dim, dim, 1, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class PointNetTCNClassifier(nn.Module):
    def __init__(self, num_classes, spatial_dim=128, dropout=0.3):
        super().__init__()
        self.encoder = FramePointNetEncoder(out_dim=spatial_dim)
        self.tcn = nn.Sequential(
            TemporalBlock(spatial_dim, kernel=3, dilation=1, dropout=dropout),
            TemporalBlock(spatial_dim, kernel=5, dilation=2, dropout=dropout),
            TemporalBlock(spatial_dim, kernel=5, dilation=4, dropout=dropout),
        )
        self.pool = AttentionPooling(spatial_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(spatial_dim),
            nn.Dropout(dropout),
            nn.Linear(spatial_dim, num_classes),
        )

    def forward(self, x):
        feat = self.encoder(x).transpose(1, 2)
        feat = self.tcn(feat).transpose(1, 2)
        return self.classifier(self.pool(feat))


class PCTFrameEncoder(nn.Module):
    def __init__(self, in_channels=4, embed_dim=128, heads=4, layers=2, dropout=0.2):
        super().__init__()
        self.embed = nn.Linear(in_channels, embed_dim)
        self.attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)
            for _ in range(layers)
        ])
        self.norm1 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(layers)])
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, embed_dim),
            )
            for _ in range(layers)
        ])
        self.fuse = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, x):
        b, s, n, c = x.shape
        h = self.embed(x.reshape(b * s, n, c))
        for attn, n1, n2, ffn in zip(self.attn, self.norm1, self.norm2, self.ffn):
            q = n1(h)
            a, _ = attn(q, q, q, need_weights=False)
            h = h + a
            h = h + ffn(n2(h))
        pooled = torch.cat([h.max(dim=1).values, h.mean(dim=1)], dim=1)
        return self.fuse(pooled).view(b, s, -1)


class PCTGRUClassifier(nn.Module):
    def __init__(self, num_classes, dim=128, dropout=0.3):
        super().__init__()
        self.encoder = PCTFrameEncoder(embed_dim=dim, dropout=dropout)
        self.rnn = nn.GRU(dim, dim, batch_first=True, bidirectional=True)
        self.pool = AttentionPooling(dim * 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, num_classes),
        )

    def forward(self, x):
        feat = self.encoder(x)
        out, _ = self.rnn(feat)
        return self.classifier(self.pool(out))


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx.transpose(2, 1) - inner - xx
    return pairwise_distance.topk(k=k, dim=-1)[1]


def graph_feature(x, k=12):
    b, c, n = x.size()
    k = min(k, n)
    idx = knn(x, k)
    idx_base = torch.arange(0, b, device=x.device).view(-1, 1, 1) * n
    idx = (idx + idx_base).view(-1)

    x_t = x.transpose(2, 1).contiguous()
    neighbors = x_t.view(b * n, c)[idx, :].view(b, n, k, c)
    center = x_t.view(b, n, 1, c).repeat(1, 1, k, 1)
    feat = torch.cat((neighbors - center, center), dim=3).permute(0, 3, 1, 2).contiguous()
    return feat


class DGCNNFrameEncoder(nn.Module):
    def __init__(self, in_channels=4, out_dim=128, k=12):
        super().__init__()
        self.k = k
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Linear((64 + 128) * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        b, s, n, c = x.shape
        x = x.reshape(b * s, n, c).transpose(1, 2).contiguous()
        f1 = self.conv1(graph_feature(x, self.k)).max(dim=-1).values
        f2 = self.conv2(graph_feature(f1, self.k)).max(dim=-1).values
        h = torch.cat([f1, f2], dim=1)
        pooled = torch.cat([h.max(dim=2).values, h.mean(dim=2)], dim=1)
        return self.fuse(pooled).view(b, s, -1)


class DGCNNGRUClassifier(nn.Module):
    def __init__(self, num_classes, dim=128, dropout=0.3):
        super().__init__()
        self.encoder = DGCNNFrameEncoder(out_dim=dim)
        self.rnn = nn.GRU(dim, dim, batch_first=True, bidirectional=True)
        self.pool = AttentionPooling(dim * 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, num_classes),
        )

    def forward(self, x):
        feat = self.encoder(x)
        out, _ = self.rnn(feat)
        return self.classifier(self.pool(out))


def build_deep_model(method: str, num_classes: int, seq_len: int):
    if method == "radar_stnet":
        return RadarSTNet(RadarSTConfig(classes=num_classes, seq_len=seq_len))
    if method == "pointnet_gru":
        return PointNetRNNClassifier(num_classes, spatial_dim=128, rnn_hidden=128, rnn_type="gru")
    if method == "robhar_like":
        return PointNetRNNClassifier(num_classes, spatial_dim=96, rnn_hidden=96, rnn_type="lstm")
    if method == "pointnet_tcn":
        return PointNetTCNClassifier(num_classes, spatial_dim=128)
    if method == "pct_gru":
        return PCTGRUClassifier(num_classes, dim=128)
    if method == "dgcnn_gru":
        return DGCNNGRUClassifier(num_classes, dim=128)
    raise ValueError(f"未知深度方法: {method}")


def make_traditional_model(method: str, seed: int):
    if method == "stats_svm":
        return make_pipeline(StandardScaler(), LinearSVC(C=1.0, class_weight="balanced", max_iter=5000))
    if method == "stats_rf":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    if method == "stats_knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights="distance"))
    if method == "stats_lr":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1),
        )
    raise ValueError(f"未知传统方法: {method}")


def majority_vote_metrics(y_true, y_pred, file_ids, class_names):
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
        "trial_weighted_f1": float(f1_score(trial_true, trial_pred, average="weighted", zero_division=0)),
        "trial_count": len(trial_true),
    }


def evaluate_predictions(y_true, y_pred, file_ids, class_names):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
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
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    metrics = {
        "sample_accuracy": float(accuracy_score(y_true, y_pred)),
        "sample_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "sample_weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "sample_count": int(len(y_true)),
        "classification_report": report,
        "confusion_matrix": cm,
    }
    metrics.update(majority_vote_metrics(y_true, y_pred, file_ids, class_names))
    return metrics


def run_traditional(method, train_ds, test_ds, class_names, seed):
    x_train, y_train, _ = build_feature_matrix(train_ds)
    x_test, y_test, file_ids = build_feature_matrix(test_ds)

    model = make_traditional_model(method, seed)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    return evaluate_predictions(y_test, pred, file_ids, class_names), 0


def class_weights_from_dataset(dataset, num_classes, device):
    labels = np.asarray([item[2] for item in dataset.items])
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.no_grad()
def predict_deep(model, loader, device, class_names):
    model.eval()
    all_true, all_pred, all_file_ids = [], [], []
    dataset = loader.dataset

    for x, y, file_idx in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().numpy()

        all_pred.extend(pred.tolist())
        all_true.extend(y.numpy().tolist())
        all_file_ids.extend([dataset.file_keys[int(i)] for i in file_idx.numpy().tolist()])

    return evaluate_predictions(all_true, all_pred, all_file_ids, class_names)


def train_deep(method, train_ds, val_ds, test_ds, class_names, args, fold_name):
    device = get_device(args.device)
    num_classes = len(class_names)
    model = build_deep_model(method, num_classes, args.seq_len).to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights_from_dataset(train_ds, num_classes, device),
        label_smoothing=args.label_smoothing,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_score = -1.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, args.epochs + 1):
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
        val_metrics = predict_deep(model, val_loader, device, class_names)
        score = val_metrics["sample_macro_f1"]

        print(
            f"[{fold_name}] {method} epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={total_loss / max(total, 1):.4f} | "
            f"train_acc={correct / max(total, 1):.4f} | "
            f"val_f1={score:.4f}"
        )

        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    test_metrics = predict_deep(model, test_loader, device, class_names)

    param_count = sum(p.numel() for p in model.parameters())
    if args.save_checkpoints:
        ckpt_dir = Path(args.output_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "method": method,
                "fold": fold_name,
                "classes": class_names,
                "seq_len": args.seq_len,
                "n_points": args.n_points,
                "model_state": best_state,
            },
            ckpt_dir / f"{fold_name}_{method}.pth",
        )

    return test_metrics, int(param_count)


def get_device(device_arg: str):
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    return obj


def save_results(rows, details, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "experiment1_summary.csv"
    if rows:
        fields = sorted(set().union(*(row.keys() for row in rows)))
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    json_path = out / "experiment1_details.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(details), f, indent=2, ensure_ascii=False)


def compact_row(protocol, fold, method, train_ids, val_ids, test_ids, metrics, params, seconds):
    return {
        "protocol": protocol,
        "fold": fold,
        "method": method,
        "train_subjects": ",".join(f"p{i}" for i in train_ids),
        "val_subjects": ",".join(f"p{i}" for i in val_ids),
        "test_subjects": ",".join(f"p{i}" for i in test_ids),
        "sample_accuracy": metrics["sample_accuracy"],
        "sample_macro_f1": metrics["sample_macro_f1"],
        "sample_weighted_f1": metrics["sample_weighted_f1"],
        "trial_accuracy": metrics["trial_accuracy"],
        "trial_macro_f1": metrics["trial_macro_f1"],
        "trial_weighted_f1": metrics["trial_weighted_f1"],
        "sample_count": metrics["sample_count"],
        "trial_count": metrics["trial_count"],
        "params": params,
        "seconds": seconds,
    }


def run_split(protocol, fold, records, class_names, train_ids, val_ids, test_ids, args, rows, details):
    train_records = [r for r in records if r.subject in train_ids]
    val_records = [r for r in records if r.subject in val_ids]
    test_records = [r for r in records if r.subject in test_ids]

    data_cfg = DatasetConfig(
        dataset_path=args.dataset_path,
        n_chunk_per_data=args.seq_len,
        n_sample_per_chunk=args.n_points,
    )

    train_deep_ds = RadarWindowDataset(
        train_records, data_cfg, args.seq_len, augment=True,
        window_stride=args.window_stride,
        max_windows_per_file=args.max_windows_per_file,
    )
    train_plain_ds = RadarWindowDataset(
        train_records, data_cfg, args.seq_len, augment=False,
        window_stride=args.window_stride,
        max_windows_per_file=args.max_windows_per_file,
    )
    val_ds = RadarWindowDataset(
        val_records, data_cfg, args.seq_len, augment=False,
        window_stride=args.window_stride,
        max_windows_per_file=args.max_windows_per_file,
    )
    test_ds = RadarWindowDataset(
        test_records, data_cfg, args.seq_len, augment=False,
        window_stride=args.window_stride,
        max_windows_per_file=args.max_windows_per_file,
    )

    print(f"\n=== {protocol} | {fold} ===")
    print(f"train windows={len(train_deep_ds)}, val windows={len(val_ds)}, test windows={len(test_ds)}")

    for method in args.methods:
        start = time.time()
        print(f"\n--- Running {method} ---")

        if method in TRADITIONAL_METHODS:
            metrics, params = run_traditional(method, train_plain_ds, test_ds, class_names, args.seed)
        elif method in DEEP_METHODS:
            metrics, params = train_deep(method, train_deep_ds, val_ds, test_ds, class_names, args, fold)
        else:
            raise ValueError(f"未知方法: {method}")

        seconds = round(time.time() - start, 2)
        row = compact_row(protocol, fold, method, train_ids, val_ids, test_ids, metrics, params, seconds)
        rows.append(row)

        details.append({
            "protocol": protocol,
            "fold": fold,
            "method": method,
            "row": row,
            "classification_report": metrics["classification_report"],
            "confusion_matrix": metrics["confusion_matrix"],
        })

        print(
            f"[{method}] sample_acc={metrics['sample_accuracy']:.4f}, "
            f"sample_macro_f1={metrics['sample_macro_f1']:.4f}, "
            f"trial_acc={metrics['trial_accuracy']:.4f}, "
            f"trial_macro_f1={metrics['trial_macro_f1']:.4f}"
        )

        save_results(rows, details, args.output_dir)


def append_loso_summary(rows):
    loso_rows = [r for r in rows if r.get("protocol") == "loso"]
    methods = sorted({r["method"] for r in loso_rows})

    for method in methods:
        method_rows = [r for r in loso_rows if r["method"] == method]
        summary = {
            "protocol": "loso_summary",
            "fold": "mean_std",
            "method": method,
        }
        for key in [
            "sample_accuracy", "sample_macro_f1", "sample_weighted_f1",
            "trial_accuracy", "trial_macro_f1", "trial_weighted_f1",
        ]:
            vals = np.asarray([r[key] for r in method_rows], dtype=np.float64)
            summary[f"{key}_mean"] = float(vals.mean())
            summary[f"{key}_std"] = float(vals.std(ddof=0))
        rows.append(summary)


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 1: action classification and cross-subject evaluation.")
    parser.add_argument("--dataset_path", default="Processed_Dataset_NPY")
    parser.add_argument("--output_dir", default="experiment1_results")
    parser.add_argument("--protocol", choices=["fixed", "loso", "both"], default="fixed")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--seq_len", type=int, default=25)
    parser.add_argument("--n_points", type=int, default=128)
    parser.add_argument("--window_stride", type=int, default=1)
    parser.add_argument("--max_windows_per_file", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--label_smoothing", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save_checkpoints", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.methods) == 1 and args.methods[0] == "all":
        args.methods = ALL_METHODS

    set_seed(args.seed)
    records, class_names = collect_records(args.dataset_path)
    subjects = sorted({r.subject for r in records if r.subject > 0})

    print(f"classes={class_names}")
    print(f"subjects={subjects}")
    print(f"methods={args.methods}")
    print(f"device={get_device(args.device)}")

    rows, details = [], []

    if args.protocol in {"fixed", "both"}:
        run_split(
            protocol="fixed",
            fold="train_p1-p5_val_p6_test_p7",
            records=records,
            class_names=class_names,
            train_ids=[1, 2, 3, 4, 5],
            val_ids=[6],
            test_ids=[7],
            args=args,
            rows=rows,
            details=details,
        )

    if args.protocol in {"loso", "both"}:
        for test_id in subjects:
            test_index = subjects.index(test_id)
            val_id = subjects[(test_index + 1) % len(subjects)]
            train_ids = [sid for sid in subjects if sid not in {test_id, val_id}]
            run_split(
                protocol="loso",
                fold=f"test_p{test_id}_val_p{val_id}",
                records=records,
                class_names=class_names,
                train_ids=train_ids,
                val_ids=[val_id],
                test_ids=[test_id],
                args=args,
                rows=rows,
                details=details,
            )
        append_loso_summary(rows)

    save_results(rows, details, args.output_dir)
    print(f"\n完成。结果保存在: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()