# ==========================================
# dataset_loader.py: 数据增强算法与数据集构建 (优化版)
# ==========================================
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from dataclasses import dataclass, field


# ----------------- 1. 配置管理区 (修改功能的入口) -----------------
@dataclass
class DatasetConfig:
    """数据集与加载器参数配置类"""
    dataset_path: str = "Processed_Dataset_NPY"
    n_chunk_per_data: int = 25
    n_sample_per_chunk: int = 128

    # 物理空间归一化边界 (米) - 需与 data_process.py 保持一致
    min_x: float = 0.5
    max_x: float = 3.0
    min_y: float = -2.5
    max_y: float = 1.0
    min_z: float = -2.5
    max_z: float = 2.2

    # Intensity 归一化范围
    intensity_clip_min: float = 0.0
    intensity_clip_max: float = 60.0

    # 数据增强参数 (提取出魔法数字，方便后续调参)
    aug_shift_std: list[float] = field(default_factory=lambda: [0.02, 0.02, 0.015])
    aug_jitter_std: float = 0.002
    aug_scale_std: list[float] = field(default_factory=lambda: [0.02, 0.02, 0.02])
    aug_intensity_std: float = 0.025


# ----------------- 2. 核心处理类 -----------------
class Resampler:
    def __init__(self, n_points: int, deterministic: bool = False):
        self.n_points = n_points
        self.deterministic = deterministic

    def resample(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)

        if points.ndim != 2 or points.shape[1] < 4:
            points = np.empty((0, 4), dtype=np.float32)
        else:
            points = points[:, :4]

        n_current = points.shape[0]

        # 空帧处理
        if n_current == 0:
            return np.zeros((self.n_points, 4), dtype=np.float32)

        # 点数足够：FPS 下采样
        if n_current >= self.n_points:
            centroids = np.zeros(self.n_points, dtype=np.int32)
            distance = np.full(n_current, 1e10, dtype=np.float32)  # 优化: 使用 np.full 更直观

            farthest = 0 if self.deterministic else np.random.randint(0, n_current)

            for i in range(self.n_points):
                centroids[i] = farthest
                centroid = points[farthest, :3]
                dist = np.sum((points[:, :3] - centroid) ** 2, axis=-1)
                distance = np.minimum(distance, dist)
                farthest = int(np.argmax(distance))

            return points[centroids].astype(np.float32)

        # 点数不足：保留真实点并随机重复补齐
        if self.deterministic:
            pad_indices = np.resize(np.arange(n_current), self.n_points - n_current)
        else:
            pad_indices = np.random.choice(n_current, self.n_points - n_current, replace=True)

        sampled = np.concatenate([points, points[pad_indices]], axis=0)
        return sampled.astype(np.float32)


class Transforms:
    def __init__(self, cfg: DatasetConfig, augment: bool = False, deterministic: bool = False):
        self.cfg = cfg
        self.resampler = Resampler(cfg.n_sample_per_chunk, deterministic=deterministic)
        self.augment = augment
        self.rng = np.random.default_rng()

    def augment_points(self, pts: np.ndarray) -> np.ndarray:
        pts = pts.astype(np.float32, copy=True)
        if pts.shape[0] == 0:
            return pts

        # 引入配置中的增强参数，替换原本的硬编码
        pts[:, :3] += self.rng.normal(0.0, self.cfg.aug_shift_std, size=(1, 3))
        pts[:, :3] += self.rng.normal(0.0, self.cfg.aug_jitter_std, size=pts[:, :3].shape)
        pts[:, :3] *= self.rng.normal(1.0, self.cfg.aug_scale_std, size=(1, 3))
        pts[:, 3] *= float(self.rng.normal(1.0, self.cfg.aug_intensity_std))

        return pts

    def normalize(self, pts: np.ndarray) -> np.ndarray:
        pts = pts.astype(np.float32, copy=True)
        c = self.cfg

        # XYZ 映射到大致 [-1, 1]
        pts[:, 0] = 2.0 * (pts[:, 0] - c.min_x) / (c.max_x - c.min_x) - 1.0
        pts[:, 1] = 2.0 * (pts[:, 1] - c.min_y) / (c.max_y - c.min_y) - 1.0
        pts[:, 2] = 2.0 * (pts[:, 2] - c.min_z) / (c.max_z - c.min_z) - 1.0

        # intensity 映射到 [0, 1]
        pts[:, 3] = np.clip(pts[:, 3], c.intensity_clip_min, c.intensity_clip_max)
        pts[:, 3] = (pts[:, 3] - c.intensity_clip_min) / (c.intensity_clip_max - c.intensity_clip_min + 1e-6)

        return pts

    def transform(self, points_list: list[np.ndarray]) -> torch.Tensor:
        processed = []
        for pts in points_list:
            pts = np.asarray(pts, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[1] < 4:
                pts = np.empty((0, 4), dtype=np.float32)
            else:
                pts = pts[:, :4]

            if self.augment:
                pts = self.augment_points(pts)

            pts = self.resampler.resample(pts)
            pts = self.normalize(pts)
            processed.append(pts)

        return torch.tensor(np.stack(processed, axis=0), dtype=torch.float32)


# ----------------- 3. 数据集构建类 -----------------
class TiDataset(Dataset):
    def __init__(self, split: str, cfg: DatasetConfig = None):
        self.cfg = cfg or DatasetConfig()  # 接收传入的配置，默认为空时自动实例化

        split_map = {"train": "Train", "test": "Test", "ver": "Ver"}
        if split not in split_map:
            raise ValueError(f"未知数据集划分: {split}")

        data_dir = Path(self.cfg.dataset_path) / split_map[split]
        if not data_dir.exists():
            raise FileNotFoundError(f"找不到数据目录: {data_dir}")

        self.transform = Transforms(
            self.cfg,
            augment=(split == "train"),
            deterministic=(split != "train")
        )

        self.idx2label = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
        self.label2idx = {name: idx for idx, name in enumerate(self.idx2label)}

        self.data = []
        self.label = []

        print(f"Loading {split} dataset...")

        for cls_name in self.idx2label:
            npy_files = sorted((data_dir / cls_name).glob("*.npy"))
            cls_idx = self.label2idx[cls_name]

            for npy_file in npy_files:
                chunks = np.load(npy_file, allow_pickle=True)

                if len(chunks) < self.cfg.n_chunk_per_data:
                    continue

                # 滑动窗口截取时序数据
                for idx in range(len(chunks) - self.cfg.n_chunk_per_data + 1):
                    self.data.append(list(chunks[idx: idx + self.cfg.n_chunk_per_data]))
                    self.label.append(cls_idx)

        print(
            f"{split} dataset loaded: "
            f"{len(self.data)} samples, {len(self.idx2label)} classes"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.transform.transform(self.data[idx])
        y = self.label[idx]
        return x, y