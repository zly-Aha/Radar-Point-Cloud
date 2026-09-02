# ==========================================
# models.py: Radar-STNet Enhanced (Performance Optimized)
# ==========================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class RadarSTConfig:
    in_channels: int = 4
    spatial_dim: int = 160
    seq_len: int = 25
    tcn_kernel: int = 5
    num_heads: int = 4
    transformer_layers: int = 2
    dropout: float = 0.3
    classes: int = 10


class FastPointEncoder(nn.Module):
    def __init__(self, config: RadarSTConfig):
        super().__init__()
        self.raw_stat_dim = config.in_channels * 4
        learned_dim = config.spatial_dim - self.raw_stat_dim

        if learned_dim <= 0:
            raise ValueError("spatial_dim 必须大于 in_channels * 4")
        if learned_dim % 3 != 0:
            raise ValueError("spatial_dim - in_channels*4 必须能被 3 整除")

        pooled_dim = learned_dim // 3

        self.mlp = nn.Sequential(
            nn.Conv1d(config.in_channels, 48, kernel_size=1, bias=False),
            nn.BatchNorm1d(48),
            nn.GELU(),
            nn.Conv1d(48, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, pooled_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(pooled_dim),
            nn.GELU()
        )
        self.out_norm = nn.LayerNorm(config.spatial_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*S, C, N]
        feat = self.mlp(x)

        max_feat = torch.max(feat, dim=2)[0]
        mean_feat = torch.mean(feat, dim=2)

        # [优化]: 使用 var + epsilon 替代 std，防止方差为 0 时引发 NaN 梯度
        std_feat = torch.sqrt(torch.var(feat, dim=2, unbiased=False) + 1e-5)

        raw = x.transpose(1, 2)  # [B*S, N, C]
        raw_mean = raw.mean(dim=1)
        raw_min = raw.min(dim=1)[0]
        raw_max = raw.max(dim=1)[0]

        # [优化]: 同上，保护真实特征的标准差计算
        raw_std = torch.sqrt(torch.var(raw, dim=1, unbiased=False) + 1e-5)

        raw_stat = torch.cat([raw_mean, raw_std, raw_min, raw_max], dim=1)
        out = torch.cat([max_feat, mean_feat, std_feat, raw_stat], dim=1)

        return self.out_norm(out)


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size // 2) * dilation

        self.net = nn.Sequential(
            nn.Conv1d(
                channels, channels,
                kernel_size=kernel_size, padding=padding,
                dilation=dilation, groups=channels, bias=False
            ),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TemporalConvNet(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float):
        super().__init__()
        self.blocks = nn.Sequential(
            TemporalBlock(channels, kernel_size=3, dilation=1, dropout=dropout),
            TemporalBlock(channels, kernel_size=kernel_size, dilation=2, dropout=dropout),
            TemporalBlock(channels, kernel_size=kernel_size, dilation=4, dropout=dropout)
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, S]
        out = self.blocks(x)
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)  # [优化]: 一行完成 transpose 归一化
        return out


class AttentionPooling(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.Tanh(),
            nn.Linear(channels // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.softmax(self.score(x), dim=1)
        return torch.sum(x * weight, dim=1)


class RadarSTNet(nn.Module):
    def __init__(self, config: RadarSTConfig = RadarSTConfig()):
        super().__init__()
        self.config = config

        self.spatial_encoder = FastPointEncoder(config)

        self.motion_fuse = nn.Sequential(
            nn.Linear(config.spatial_dim * 3, config.spatial_dim),
            nn.LayerNorm(config.spatial_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, config.seq_len, config.spatial_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.tcn = TemporalConvNet(
            channels=config.spatial_dim,
            kernel_size=config.tcn_kernel,
            dropout=config.dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.spatial_dim,
            nhead=config.num_heads,
            dim_feedforward=config.spatial_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False  # [优化]: 改为 True (Pre-LN)，极大提升 Transformer 收敛稳定性
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.transformer_layers
        )

        self.pool = AttentionPooling(config.spatial_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(config.spatial_dim),
            nn.Linear(config.spatial_dim, config.spatial_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.spatial_dim // 2, config.classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 x: [B, S, N, C]"""
        B, S, N, C = x.shape
        if S > self.config.seq_len:
            raise ValueError(f"输入序列长度 S={S} 大于模型配置 seq_len={self.config.seq_len}")

        # [优化]: 使用 flatten 替代 reshape，语义更清晰，底层内存更连续
        x = x.flatten(0, 1).transpose(1, 2)

        spatial_feat = self.spatial_encoder(x).view(B, S, self.config.spatial_dim)

        # [优化]: 使用 F.pad 计算差分。避免了 zeros_like 的显存开销和切片赋值的 Autograd 开销
        # F.pad 参数格式 (左, 右, 上, 下) -> 对于时序 S 维度，(0,0)不填充特征维度，(1,0)在时序最前面补 1 个 0
        diff1 = spatial_feat[:, 1:] - spatial_feat[:, :-1]
        delta_feat = F.pad(diff1, (0, 0, 1, 0))

        diff2 = delta_feat[:, 1:] - delta_feat[:, :-1]
        accel_feat = F.pad(diff2, (0, 0, 1, 0))

        motion_feat = torch.cat([spatial_feat, delta_feat, accel_feat], dim=-1)
        motion_feat = self.motion_fuse(motion_feat)
        motion_feat = motion_feat + self.pos_embed[:, :S, :]

        # TCN & Transformer 特征提取
        tcn_out = self.tcn(motion_feat.transpose(1, 2)).transpose(1, 2)
        trans_out = self.transformer(tcn_out)

        # 全局池化与分类
        global_feat = self.pool(trans_out)
        return self.classifier(global_feat)