# -*- coding: utf-8 -*-
"""
图数据构建模块
- 从 B 线聚合数据构造 ASTGCN 输入
- 提取 B 线邻接矩阵
- 滑动窗口生成 (T_hist, N, F) → (T_pred, N, 2) 样本
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# 配置
# ============================================================
FULL_DATA   = r"d:\作业\py展示\data\b_line_full.csv"
ROAD_MAP    = r"d:\作业\py展示\Hangzhou-mobility-data-set\Metro_roadMap.csv"
B_STATIONS  = list(range(0, 34))
SLOTS_PER_DAY = 144
T_HIST = 12    # 用过去 12 个时段（2小时）预测
T_PRED = 3     # 预测未来 3 个时段（30分钟）
BATCH_SIZE = 64
TRAIN_END = "2019-01-22"


def build_graph_data():
    """主入口：返回 (train_loader, valid_loader, adj_matrix, scaler_dict)"""
    print("=" * 50)
    print(" 构建 ASTGCN 图数据")
    print("=" * 50)

    # ---- 1. 加载邻接矩阵 ----
    adj = _load_adj_matrix()
    print(f"  邻接矩阵: {adj.shape}")

    # ---- 2. 加载聚合数据并转为 3D 张量 ----
    X_3d, y_3d, dates = _load_and_reshape()
    print(f"  3D张量: X={X_3d.shape}, y={y_3d.shape}")

    # ---- 3. 标准化 ----
    X_scaled, scaler_dict = _normalize(X_3d)
    print(f"  标准化完成")

    # ---- 4. 滑动窗口生成样本 ----
    X_samples, y_samples = _sliding_window(X_scaled, y_3d, T_HIST, T_PRED)
    print(f"  滑动窗口样本: X={X_samples.shape}, y={y_samples.shape}")

    # ---- 5. 按日期严格拆分训练/验证 ----
    # 取目标第一个时段所属日期判断归属
    train_indices, valid_indices = [], []
    for i in range(len(X_samples)):
        target_date = dates[i + T_HIST]  # 预测的第一个时段
        if target_date <= TRAIN_END:
            train_indices.append(i)
        else:
            valid_indices.append(i)

    X_train = X_samples[train_indices]
    y_train = y_samples[train_indices]
    X_valid = X_samples[valid_indices]
    y_valid = y_samples[valid_indices]

    print(f"  训练窗口: {X_train.shape[0]}  验证窗口: {X_valid.shape[0]}")

    # ---- 6. 构建 DataLoader ----
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    valid_dataset = TensorDataset(
        torch.FloatTensor(X_valid),
        torch.FloatTensor(y_valid)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 邻接矩阵转为 tensor
    adj_tensor = torch.FloatTensor(adj)

    print(f"  DataLoader 就绪: train={len(train_loader)} batches, valid={len(valid_loader)} batches")
    return train_loader, valid_loader, adj_tensor, scaler_dict


# ============================================================
# 内部函数
# ============================================================

def _load_adj_matrix():
    """提取 B 线 34×34 邻接子矩阵"""
    adj_full = pd.read_csv(ROAD_MAP, index_col=0)
    # 列名/索引是字符串，转为 int
    adj_full.index = adj_full.index.astype(int)
    adj_full.columns = adj_full.columns.astype(int)
    # 取 B 线站点
    adj_b = adj_full.loc[B_STATIONS, B_STATIONS].values.astype(np.float32)
    # 添加自连接
    np.fill_diagonal(adj_b, 1.0)
    # 对称归一化: D^(-1/2) * A * D^(-1/2)
    degree = adj_b.sum(axis=1)
    degree_sqrt_inv = np.where(degree > 0, 1.0 / np.sqrt(degree), 0)
    d_inv_sqrt = np.diag(degree_sqrt_inv)
    adj_norm = d_inv_sqrt @ adj_b @ d_inv_sqrt
    return adj_norm


def _load_and_reshape():
    """将聚合表格转为 (T, N, F) 3D 张量"""
    df = pd.read_csv(FULL_DATA, parse_dates=['time_slot'])
    df = df.sort_values(['time_slot', 'stationID']).reset_index(drop=True)

    # 特征列
    feat_cols = ['hour', 'minute', 'weekday', 'is_weekend', 'is_peak',
                 'inNums', 'outNums']
    # 目标列
    target_cols = ['inNums', 'outNums']

    # 总时隙数
    all_slots = df['time_slot'].unique()
    n_slots = len(all_slots)  # 3600 = 25天 × 144

    # 构建 (T, N, F)
    X_list, y_list, dates_list = [], [], []
    for t_idx, ts in enumerate(all_slots):
        slot_data = df[df['time_slot'] == ts].set_index('stationID')
        # 确保 34 站都齐全
        slot_data = slot_data.reindex(B_STATIONS, fill_value=0)
        X_list.append(slot_data[feat_cols].values.astype(np.float32))
        y_list.append(slot_data[target_cols].values.astype(np.float32))
        dates_list.append(slot_data['date_str'].iloc[0] if 'date_str' in slot_data.columns else str(ts))

    X_3d = np.stack(X_list, axis=0)  # (T, N, F)
    y_3d = np.stack(y_list, axis=0)  # (T, N, 2)

    return X_3d, y_3d, dates_list


def _normalize(X_3d):
    """按每个特征做 Z-score 标准化，返回 scaler 参数"""
    T, N, F = X_3d.shape
    # 对每个特征列计算训练集的均值和标准差
    n_train_slots = 22 * SLOTS_PER_DAY  # 前22天
    X_train_part = X_3d[:n_train_slots].reshape(-1, F)

    mean = X_train_part.mean(axis=0)
    std = X_train_part.std(axis=0) + 1e-8

    X_scaled = (X_3d - mean) / std

    scaler_dict = {'mean': mean, 'std': std}
    return X_scaled, scaler_dict


def _sliding_window(X, y, T_hist, T_pred):
    """滑动窗口: (T, N, F) → (samples, T_hist, N, F)"""
    T = X.shape[0]
    X_windows, y_windows = [], []

    for i in range(0, T - T_hist - T_pred + 1, 1):
        x_win = X[i:i + T_hist]           # (T_hist, N, F)
        y_win = y[i + T_hist:i + T_hist + T_pred]  # (T_pred, N, 2)
        X_windows.append(x_win)
        y_windows.append(y_win)

    return np.stack(X_windows, axis=0), np.stack(y_windows, axis=0)


# ============================================================
# 测试
# ============================================================
if __name__ == '__main__':
    train_loader, valid_loader, adj, scaler = build_graph_data()
    print(f"\n邻接矩阵范围: [{adj.min():.2f}, {adj.max():.2f}]")
    xb, yb = next(iter(train_loader))
    print(f"一个 batch: X={xb.shape}, y={yb.shape}")
    print(f"预测目标范围: in=[{yb[:,:,0].min():.2f}, {yb[:,:,0].max():.2f}]")
