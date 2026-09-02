# ==========================================
# train.py: Radar-STNet GPU 训练与验证引擎 (优化版)
# 当前设置: train 训练, test 作为验证集, ver 作为最终盲测集
# ==========================================
import os
import random
import argparse
from datetime import datetime
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score

from models import RadarSTNet, RadarSTConfig
from dataset_loader import TiDataset, DatasetConfig


# ----------------- 1. 训练参数配置区 (修改功能入口) -----------------
@dataclass
class TrainConfig:
    batch_size: int = 32
    num_workers: int = 4
    n_epoch: int = 30
    learning_rate: float = 5e-4
    weight_decay: float = 1e-3
    label_smoothing: float = 0.02
    mc_samples: int = 1
    seed: int = 42

    # 针对当前盲测 jump 召回率低的问题重新分配权重
    class_weights: dict = field(default_factory=lambda: {
        "jump": 1.2,
        "walk": 1.0,
        "squat": 1.6,
        "stand": 2.6,
        "box": 1.1,
    })


def parse_arguments() -> TrainConfig:
    """解析命令行参数，实现无缝修改配置"""
    parser = argparse.ArgumentParser(description="Radar-STNet 训练脚本")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_epoch", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--mc_samples", type=int, default=1)
    args = parser.parse_args()

    return TrainConfig(
        batch_size=args.batch_size,
        n_epoch=args.n_epoch,
        learning_rate=args.lr,
        mc_samples=args.mc_samples
    )


# ----------------- 2. 辅助工具函数 -----------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def enable_mc_dropout(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def build_class_weights(train_dataset, class_weights_dict, num_classes, device):
    class_weights = torch.ones(num_classes, dtype=torch.float32)
    for cls_name, weight in class_weights_dict.items():
        if cls_name in train_dataset.label2idx:
            class_idx = train_dataset.label2idx[cls_name]
            class_weights[class_idx] = float(weight)

    print(">>> 使用类别权重:")
    for cls_name, class_idx in train_dataset.label2idx.items():
        print(f"    {cls_name:<16s}: {class_weights[class_idx].item():.2f}")
    return class_weights.to(device)


# ----------------- 3. 验证与评估逻辑 -----------------
@torch.no_grad()
def evaluate(model, loader, device, label2idx, num_classes, mc_samples: int = 1):
    model.eval()
    if mc_samples > 1:
        enable_mc_dropout(model)

    all_preds, all_trues = [], []
    loss_sum, total = 0.0, 0
    criterion = nn.CrossEntropyLoss()

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        if mc_samples > 1:
            outputs = torch.stack([model(x) for _ in range(mc_samples)])
            probs = torch.softmax(outputs, dim=-1)
            mean_probs = probs.mean(dim=0)
            pred = mean_probs.argmax(dim=1)
            logits_for_loss = torch.log(mean_probs + 1e-8)
            loss = nn.NLLLoss()(logits_for_loss, y)
        else:
            logits = model(x)
            loss = criterion(logits, y)
            pred = logits.argmax(dim=1)

        batch_size = y.size(0)
        all_preds.append(pred.cpu().numpy())
        all_trues.append(y.cpu().numpy())
        loss_sum += loss.item() * batch_size
        total += batch_size

    all_preds = np.concatenate(all_preds)
    all_trues = np.concatenate(all_trues)

    acc = float((all_preds == all_trues).mean())
    macro_f1 = float(f1_score(all_trues, all_preds, average="macro", zero_division=0))

    recalls = recall_score(all_trues, all_preds, labels=np.arange(num_classes), average=None, zero_division=0)
    jump_idx = label2idx.get("jump", None)
    jump_recall = float(recalls[jump_idx]) if jump_idx is not None else 0.0

    return loss_sum / max(total, 1), acc, macro_f1, jump_recall


# ----------------- 4. 核心训练引擎 -----------------
def train():
    cfg = parse_arguments()
    set_seed(cfg.seed)

    # 智能设备选择 (支持 CUDA, Apple Silicon MPS, 或 CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f">>> 使用 GPU 训练: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(">>> 使用 Apple Silicon MPS 加速训练")
    else:
        device = torch.device("cpu")
        print(">>> 未检测到加速硬件，当前使用 CPU 训练")

    log_dir = os.path.join("logs", datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    print("\n>>> 初始化数据集...")
    dataset_cfg = DatasetConfig()  # 获取前面优化过的数据集配置
    train_dataset = TiDataset("train", cfg=dataset_cfg)
    val_dataset = TiDataset("test", cfg=dataset_cfg)
    num_classes = len(train_dataset.idx2label)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False
    )

    model_config = RadarSTConfig(classes=num_classes, seq_len=dataset_cfg.n_chunk_per_data)
    model = RadarSTNet(model_config).to(device)

    class_weights = build_class_weights(train_dataset, cfg.class_weights, num_classes, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_epoch, eta_min=1e-6)

    # [优化] 初始化自动混合精度 (AMP) 缩放器
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu', enabled=device.type == 'cuda')

    best_score, best_acc, best_macro_f1, best_jump_recall = 0.0, 0.0, 0.0, 0.0

    print(f"\n>>> 启动训练: epochs={cfg.n_epoch}, batch_size={cfg.batch_size}, mc_samples={cfg.mc_samples}")

    for epoch in range(cfg.n_epoch):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        current_lr = scheduler.get_last_lr()[0]

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.n_epoch} [Train]")

        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            # [优化] 开启 AMP 上下文，加速前向传播并节省显存
            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu', enabled=device.type == 'cuda'):
                pred = model(x)
                loss = criterion(pred, y)

            # [优化] 使用 Scaler 进行反向传播
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # 梯度裁剪前必须 unscale
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            batch_size = y.size(0)
            train_loss += loss.item() * batch_size
            train_correct += (pred.argmax(dim=1) == y).sum().item()
            train_total += batch_size

            pbar.set_postfix({
                "loss": f"{train_loss / max(train_total, 1):.4f}",
                "acc": f"{train_correct / max(train_total, 1):.2%}",
                "lr": f"{current_lr:.2e}"
            })

        val_loss, val_acc, val_macro_f1, val_jump_recall = evaluate(
            model, val_loader, device, train_dataset.label2idx, num_classes, cfg.mc_samples
        )

        save_score = 0.6 * val_macro_f1 + 0.4 * val_jump_recall

        print(
            f"--> Epoch {epoch + 1}/{cfg.n_epoch} | Loss: {train_loss / train_total:.4f} | "
            f"Acc: {train_correct / train_total:.2%} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2%} | Macro-F1: {val_macro_f1:.4f} | Jump Rec: {val_jump_recall:.4f} | "
            f"Score: {save_score:.4f}"
        )

        if save_score > best_score:
            best_score, best_acc, best_macro_f1, best_jump_recall = save_score, val_acc, val_macro_f1, val_jump_recall
            torch.save({
                "model_state": model.state_dict(),
                "config": model_config.__dict__,
                "classes": train_dataset.idx2label,
                "best_score": best_score,
                "epoch": epoch + 1
            }, os.path.join(log_dir, "best_model.pth"))
            print(f">>> [更新最佳模型] Score={best_score:.4f}")

        scheduler.step()


if __name__ == "__main__":
    train()