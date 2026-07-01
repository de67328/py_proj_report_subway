# -*- coding: utf-8 -*-
"""
ASTGCN 模型定义 — 全三线 GPU 版
- Spatial Attention + Temporal Attention + GCN + Time Conv
- 支持多 ST Block 堆叠
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 注意力模块
# ============================================================

class SpatialAttention(nn.Module):
    """空间注意力：(B, T, N, C) → (B, N, N) 注意力矩阵"""
    def __init__(self, in_channels, num_nodes):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(in_channels, 1))
        self.W2 = nn.Parameter(torch.empty(in_channels, 1))
        self.b  = nn.Parameter(torch.empty(num_nodes, num_nodes))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        nn.init.xavier_uniform_(self.b)

    def forward(self, x):
        B, T, N, C = x.shape
        x_mean = x.mean(dim=1)                        # (B, N, C)
        lhs = x_mean @ self.W1                        # (B, N, 1)
        rhs = x_mean @ self.W2                        # (B, N, 1)
        att = lhs + rhs.transpose(-1, -2) + self.b    # (B, N, N)
        return F.softmax(att, dim=-1)


class TemporalAttention(nn.Module):
    """时间注意力：(B, T, N, C) → (B, T, T) 注意力矩阵"""
    def __init__(self, in_channels, num_timesteps):
        super().__init__()
        self.U1 = nn.Parameter(torch.empty(in_channels, 1))
        self.U2 = nn.Parameter(torch.empty(in_channels, 1))
        self.b_t = nn.Parameter(torch.empty(num_timesteps, num_timesteps))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.U1)
        nn.init.xavier_uniform_(self.U2)
        nn.init.xavier_uniform_(self.b_t)

    def forward(self, x):
        B, T, N, C = x.shape
        x_mean = x.mean(dim=2)                        # (B, T, C)
        lhs = x_mean @ self.U1                        # (B, T, 1)
        rhs = x_mean @ self.U2                        # (B, T, 1)
        att = lhs + rhs.transpose(-1, -2) + self.b_t  # (B, T, T)
        return F.softmax(att, dim=-1)


# ============================================================
# GCN 层
# ============================================================

class GCNLayer(nn.Module):
    """图卷积：H' = ReLU(A_norm @ H @ W + b)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.W = nn.Parameter(torch.empty(in_channels, out_channels))
        self.b = nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.b)

    def forward(self, x, adj):
        support = x @ self.W
        output  = adj @ support + self.b
        return F.relu(output)


# ============================================================
# ST Block
# ============================================================

class STBlock(nn.Module):
    """时空块：时间注意力 → 空间注意力 → GCN → 时间卷积 + 残差"""
    def __init__(self, channels, num_nodes, num_timesteps, dropout=0.1):
        super().__init__()
        self.temporal_att   = TemporalAttention(channels, num_timesteps)
        self.spatial_att    = SpatialAttention(channels, num_nodes)
        self.gcn            = GCNLayer(channels, channels)
        self.time_conv      = nn.Conv2d(channels, channels, kernel_size=(3, 1),
                                         padding=(1, 0))
        self.bn             = nn.BatchNorm2d(channels)
        self.dropout        = nn.Dropout(dropout)
        self.channels       = channels

    def forward(self, x, adj):
        B, T, N, C = x.shape
        residual = x

        # 时间注意力
        t_att = self.temporal_att(x)
        x = torch.einsum('btnc,btk->bknc', x, t_att)

        # 空间注意力
        s_att = self.spatial_att(x)
        x = torch.einsum('btnc,bnm->btmc', x, s_att)

        # GCN
        x_flat = x.reshape(B * T, N, C)
        x_gcn  = self.gcn(x_flat, adj)
        x = x_gcn.reshape(B, T, N, C)
        x = self.dropout(x)

        # 时间卷积
        x = x.permute(0, 3, 1, 2)       # (B, C, T, N)
        x = self.time_conv(x)
        x = self.bn(x)
        x = x.permute(0, 2, 3, 1)       # (B, T, N, C)

        return F.relu(x + residual)


# ============================================================
# ASTGCN 完整模型
# ============================================================

class ASTGCN(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels,
                 num_timesteps, num_blocks=3, pred_timesteps=6,
                 out_channels=2, dropout=0.1):
        super().__init__()
        self.num_nodes      = num_nodes
        self.pred_timesteps = pred_timesteps

        # 输入投影
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # ST Blocks
        self.st_blocks = nn.ModuleList([
            STBlock(hidden_channels, num_nodes, num_timesteps, dropout)
            for _ in range(num_blocks)
        ])

        # 输出层
        self.station_embed = nn.Embedding(num_nodes, hidden_channels // 2)
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_channels + hidden_channels // 2, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def forward(self, x, adj):
        B, T, N, _ = x.shape

        x = self.input_proj(x)

        for st_block in self.st_blocks:
            x = st_block(x, adj)

        # 输出层
        x_last = x[:, -1, :, :]                              # (B, N, C)
        sids   = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        s_emb  = self.station_embed(sids)                    # (B, N, C/2)
        z      = torch.cat([x_last, s_emb], dim=-1)          # (B, N, 3C/2)

        preds = []
        for _ in range(self.pred_timesteps):
            preds.append(self.output_fc(z))
        return torch.stack(preds, dim=1)                     # (B, T_pred, N, 2)

    def extract_embeddings(self, x, adj):
        """提取 GCN 空间嵌入（不经过输出层）"""
        self.eval()
        with torch.no_grad():
            x = self.input_proj(x)
            for st_block in self.st_blocks:
                x = st_block(x, adj)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
