# -*- coding: utf-8 -*-
"""
ASTGCN 模型定义
- Spatial Attention: 学习站点间的重要性权重
- Temporal Attention: 学习时段间的重要性权重
- GCN Layer: 图卷积空间特征提取
- ST-Block: 时空组合块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """空间注意力：给定 (B, T, N, C)，输出站点间注意力矩阵"""
    def __init__(self, in_channels, num_nodes):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(in_channels, 1))
        self.W2 = nn.Parameter(torch.randn(in_channels, 1))
        self.b = nn.Parameter(torch.randn(num_nodes, num_nodes))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        nn.init.xavier_uniform_(self.b)

    def forward(self, x):
        # x: (B, T, N, C)
        B, T, N, C = x.shape
        # 沿时间维取平均作为站点表征
        x_mean = x.mean(dim=1)  # (B, N, C)
        lhs = torch.matmul(x_mean, self.W1).squeeze(-1)  # (B, N)
        rhs = torch.matmul(x_mean, self.W2).squeeze(-1)  # (B, N)
        # 计算注意力
        att = lhs.unsqueeze(-1) + rhs.unsqueeze(-2) + self.b  # (B, N, N)
        att = F.softmax(att, dim=-1)
        return att


class TemporalAttention(nn.Module):
    """时间注意力：给定 (B, T, N, C)，输出时段间注意力矩阵"""
    def __init__(self, in_channels, num_timesteps):
        super().__init__()
        self.U1 = nn.Parameter(torch.randn(in_channels, 1))
        self.U2 = nn.Parameter(torch.randn(in_channels, 1))
        self.b_t = nn.Parameter(torch.randn(num_timesteps, num_timesteps))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.U1)
        nn.init.xavier_uniform_(self.U2)
        nn.init.xavier_uniform_(self.b_t)

    def forward(self, x):
        # x: (B, T, N, C)
        B, T, N, C = x.shape
        # 沿站点维取平均作为时段表征
        x_mean = x.mean(dim=2)  # (B, T, C)
        lhs = torch.matmul(x_mean, self.U1).squeeze(-1)  # (B, T)
        rhs = torch.matmul(x_mean, self.U2).squeeze(-1)  # (B, T)
        att = lhs.unsqueeze(-1) + rhs.unsqueeze(-2) + self.b_t  # (B, T, T)
        att = F.softmax(att, dim=-1)
        return att


class GCNLayer(nn.Module):
    """图卷积层: H' = relu(A_norm @ H @ W)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_channels, out_channels))
        self.b = nn.Parameter(torch.randn(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.b)

    def forward(self, x, adj):
        # x: (B*T, N, C_in)
        # adj: (N, N)
        support = torch.matmul(x, self.W)  # (B*T, N, C_out)
        output = torch.matmul(adj, support) + self.b  # 图卷积
        return F.relu(output)


class STBlock(nn.Module):
    """时空块：时间注意力 → 空间注意力 → GCN → 时间卷积"""
    def __init__(self, in_channels, out_channels, num_nodes, num_timesteps):
        super().__init__()
        self.temporal_att = TemporalAttention(in_channels, num_timesteps)
        self.spatial_att = SpatialAttention(in_channels, num_nodes)
        self.gcn = GCNLayer(in_channels, out_channels)
        # 时间卷积 (1D conv along time)
        self.time_conv = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 1),
            padding=(1, 0)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.out_channels = out_channels

    def forward(self, x, adj):
        # x: (B, T, N, C)
        B, T, N, C = x.shape

        # ---- 1. 时间注意力 ----
        t_att = self.temporal_att(x)  # (B, T, T)
        x_t = torch.einsum('btnc,btk->bknc', x, t_att)  # (B, T, N, C)

        # ---- 2. 空间注意力 ----
        s_att = self.spatial_att(x_t)  # (B, N, N)
        x_s = torch.einsum('btnc,bnm->btmc', x_t, s_att)  # (B, T, N, C)

        # ---- 3. GCN ----
        x_flat = x_s.reshape(B * T, N, C)  # (B*T, N, C)
        x_gcn = self.gcn(x_flat, adj)      # (B*T, N, C_out)
        x_gcn = x_gcn.reshape(B, T, N, self.out_channels)  # (B, T, N, C_out)

        # ---- 4. 时间卷积 ----
        x_gcn = x_gcn.permute(0, 3, 1, 2)  # (B, C_out, T, N)
        x_conv = self.time_conv(x_gcn)      # (B, C_out, T, N)
        x_out = self.bn(x_conv)
        x_out = x_out.permute(0, 2, 3, 1)   # (B, T, N, C_out)

        # 残差连接（如果通道数匹配）
        if C == self.out_channels:
            x_out = x_out + x_t
        else:
            # 通道不匹配时用1x1卷积对齐
            shortcut = x_t.permute(0, 3, 1, 2)  # (B, C, T, N)
            shortcut = F.conv2d(shortcut,
                                torch.ones(self.out_channels, C, 1, 1, device=x.device) * 0.1,
                                padding=0)
            shortcut = shortcut.permute(0, 2, 3, 1)  # (B, T, N, C_out)
            x_out = x_out + shortcut

        return x_out


class ASTGCN(nn.Module):
    """ASTGCN 完整模型"""
    def __init__(self, num_nodes, in_channels, hidden_channels, num_timesteps,
                 num_blocks=2, pred_timesteps=6, out_channels=2):
        super().__init__()
        self.num_nodes = num_nodes
        self.pred_timesteps = pred_timesteps

        # 输入投影
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # ST 块
        self.st_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.st_blocks.append(
                STBlock(hidden_channels, hidden_channels, num_nodes, num_timesteps)
            )

        # 输出层: 对每个站点每个预测时段做回归
        # 使用轻量方案：所有站点共享输出权重 + 站点嵌入
        self.station_embed = nn.Embedding(num_nodes, hidden_channels // 2)
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_channels + hidden_channels // 2, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def forward(self, x, adj):
        # x: (B, T_hist, N, F), adj: (N, N)
        B, T, N, _ = x.shape
        x = self.input_proj(x)

        for st_block in self.st_blocks:
            x = st_block(x, adj)

        # 保存嵌入供外部提取
        self._last_embeddings = x  # (B, T, N, hidden)

        # 取最后一个时间步
        x_last = x[:, -1, :, :]  # (B, N, hidden)
        station_ids = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        station_emb = self.station_embed(station_ids)
        x_combined = torch.cat([x_last, station_emb], dim=-1)

        preds = []
        for t in range(self.pred_timesteps):
            p = self.output_fc(x_combined)
            preds.append(p)

        y_pred = torch.stack(preds, dim=1)
        return y_pred

    def extract_embeddings(self, x, adj):
        """提取 GCN 空间嵌入（不经过输出层），返回 (B, T, N, hidden)"""
        self.eval()
        with torch.no_grad():
            x = self.input_proj(x)
            for st_block in self.st_blocks:
                x = st_block(x, adj)
        return x  # (B, T, N, hidden_channels)



def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# 测试
# ============================================================
if __name__ == '__main__':
    # 模拟输入
    B, T_hist, N, F = 4, 12, 34, 7
    x = torch.randn(B, T_hist, N, F)
    adj = torch.eye(N) + torch.rand(N, N) * 0.1
    adj = adj / adj.sum(dim=1, keepdim=True)

    model = ASTGCN(
        num_nodes=N,
        in_channels=F,
        hidden_channels=32,
        num_timesteps=T_hist,
        num_blocks=2,
        pred_timesteps=6,
        out_channels=2,
    )

    y = model(x, adj)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Params: {count_parameters(model):,}")
