from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
TRAIN = CODE_DIR / "train.py"
EXP1 = CODE_DIR / "experiment1_action_classification.py"
LOSO = CODE_DIR / "run_loso_7fold.py"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def has(pattern: str, text: str) -> bool:
    return re.search(pattern, text, flags=re.S) is not None


def show(name: str, ok: bool, evidence: str = ""):
    mark = "OK" if ok else "MISSING"
    suffix = f" | {evidence}" if evidence else ""
    print(f"{mark:<7} {name}{suffix}")


def main():
    train_text = read_text(TRAIN)
    exp1_text = read_text(EXP1)
    loso_text = read_text(LOSO)

    print("=== 训练实现细节代码核对 ===")

    checks = [
        (
            "PyTorch 实现",
            all(token in train_text + exp1_text for token in ["import torch", "torch.nn"]),
            "train.py / experiment1_action_classification.py 均使用 torch",
        ),
        (
            "AdamW 优化器",
            has(r"torch\.optim\.AdamW.*lr\s*=.*weight_decay\s*=", exp1_text)
            and has(r"torch\.optim\.AdamW.*lr\s*=.*weight_decay\s*=", train_text),
            "optimizer = torch.optim.AdamW(...)",
        ),
        (
            "初始学习率 5e-4",
            has(r"learning_rate\s*:\s*float\s*=\s*5e-4", train_text)
            and has(r"--lr[^\n]*default\s*=\s*5e-4", exp1_text)
            and has(r"--lr[^\n]*default\s*=\s*5e-4", loso_text),
            "train.py 默认 learning_rate=5e-4，主实验/LOSO 默认 --lr=5e-4",
        ),
        (
            "权重衰减 1e-3",
            has(r"weight_decay\s*:\s*float\s*=\s*1e-3", train_text)
            and has(r"--weight_decay[^\n]*default\s*=\s*1e-3", exp1_text)
            and has(r"--weight_decay[^\n]*default\s*=\s*1e-3", loso_text),
            "weight_decay=1e-3",
        ),
        (
            "带类别权重的交叉熵损失",
            has(r"CrossEntropyLoss\(\s*weight\s*=", exp1_text)
            and has(r"class_weights_from_dataset", exp1_text),
            "类别权重由训练集类别计数反比计算",
        ),
        (
            "标签平滑系数 0.02",
            has(r"label_smoothing\s*:\s*float\s*=\s*0\.02", train_text)
            and has(r"--label_smoothing[^\n]*default\s*=\s*0\.02", exp1_text)
            and has(r"--label_smoothing[^\n]*default\s*=\s*0\.02", loso_text),
            "CrossEntropyLoss(..., label_smoothing=args.label_smoothing)",
        ),
        (
            "batch size 32",
            has(r"batch_size\s*:\s*int\s*=\s*32", train_text)
            and has(r"--batch_size[^\n]*default\s*=\s*32", exp1_text)
            and has(r"--batch_size[^\n]*default\s*=\s*32", loso_text),
            "DataLoader batch_size=args.batch_size",
        ),
        (
            "最大训练轮数 30",
            has(r"n_epoch\s*:\s*int\s*=\s*30", train_text)
            and has(r"--epochs[^\n]*default\s*=\s*30", exp1_text)
            and has(r"--epochs[^\n]*default\s*=\s*30", loso_text),
            "主实验/LOSO 默认 --epochs=30",
        ),
        (
            "随机种子 42",
            has(r"seed\s*:\s*int\s*=\s*42", train_text)
            and has(r"--seed[^\n]*default\s*=\s*42", exp1_text)
            and has(r"--seed[^\n]*default\s*=\s*42", loso_text)
            and has(r"random\.seed\(seed\).*np\.random\.seed\(seed\).*torch\.manual_seed\(seed\)", exp1_text),
            "random / numpy / torch 均设置 seed",
        ),
        (
            "CosineAnnealingLR 调度器",
            has(r"CosineAnnealingLR\([^)]*T_max\s*=\s*args\.epochs[^)]*eta_min\s*=\s*1e-6", exp1_text)
            and has(r"CosineAnnealingLR\([^)]*T_max\s*=\s*cfg\.n_epoch[^)]*eta_min\s*=\s*1e-6", train_text),
            "T_max 为训练轮数，eta_min=1e-6",
        ),
        (
            "依据验证集样本级 Macro-F1 保存最优模型",
            has(r"val_metrics\s*=\s*predict_deep\(model,\s*val_loader", exp1_text)
            and has(r"score\s*=\s*val_metrics\[[\"']sample_macro_f1[\"']\]", exp1_text)
            and has(r"if\s+score\s*>\s*best_score:", exp1_text),
            "主实验脚本使用 val sample_macro_f1 选模",
        ),
        (
            "独立测试集报告样本级与试次级指标",
            has(r"test_metrics\s*=\s*predict_deep\(model,\s*test_loader", exp1_text)
            and all(key in exp1_text for key in [
                "sample_accuracy",
                "sample_macro_f1",
                "trial_accuracy",
                "trial_macro_f1",
                "majority_vote_metrics",
            ]),
            "predict_deep(test_loader) + majority_vote_metrics",
        ),
    ]

    for name, ok, evidence in checks:
        show(name, ok, evidence)

    all_ok = all(ok for _, ok, _ in checks)
    print("\n=== 可复制到论文的写法 ===")
    if not all_ok:
        print("注意：存在未通过项，请先核对代码或脚本路径。")

    print(
        "模型基于 PyTorch 实现。深度学习模型训练采用 AdamW 优化器，初始学习率为 5e-4，"
        "权重衰减系数为 1e-3；损失函数为带类别权重和标签平滑的交叉熵损失，"
        "其中类别权重由训练集类别频数的反比计算，标签平滑系数为 0.02。"
        "训练批次大小为 32，最大训练轮数为 30，随机种子为 42。学习率调度器采用 "
        "CosineAnnealingLR，T_max 为训练轮数，最低学习率 eta_min 为 1e-6。"
        "训练过程中以验证集样本级 Macro-F1 作为模型选择依据保存最优模型，"
        "并在独立测试集上报告样本级 Acc、样本级 Macro-F1 以及基于试次多数投票得到的"
        "试次级 Acc 和试次级 Macro-F1。"
    )


if __name__ == "__main__":
    main()
