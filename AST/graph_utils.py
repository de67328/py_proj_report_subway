# -*- coding: utf-8 -*-
"""
图数据构建 — 全三线
- 将聚合表格转为 (T, N, F) 3D 张量
- 滑动窗口生成训练样本
- 按日期严格拆分训练/验证
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# 配置
# ============================================================
DATA_PATH = r"d:\作业\py展示\AST\data\all_lines_full.csv"
ADJ_PATH   = r"d:\作业\py展示\AST\data\adj_matrix.csv"
MAP_PATH   = r"d:\作业\py展示\AST\data\station_line_map.csv"

SLOTS_PER_DAY = 144
N_STATIONS    = 80       # 54号缺失
FEATURE_COLS  = ['hour', 'minute', 'weekday', 'is_weekend', 'is_peak', 'inNums', 'outNums']
TARGET_COLS   = ['inNums', 'outNums']
T_HIST        = 144      # 输入历史时段（1天）— 基于 ACF 分析：日周期性最强
T_PRED        = 6        # 预测未来时段
BATCH_SIZE    = 32
TRAIN_END     = "2019-01-22"


def build_graph_data():
    """主入口：返回 (train_loader, valid_loader, adj_tensor, scaler_dict)"""
    print("=" * 50)
    print(" 构建图数据 (全三线)")
    print("=" * 50)

    adj      = _load_adj()
    X_3d, y_3d, dates = _load_and_reshape()
    X_scaled, scaler  = _normalize(X_3d)
    X_samples, y_samples = _sliding_window(X_scaled, y_3d)

    # 按日期拆分
    train_idx, valid_idx = [], []
    for i in range(len(X_samples)):
        tgt_date = dates[i + T_HIST]
        if tgt_date <= TRAIN_END:
            train_idx.append(i)
        else:
            valid_idx.append(i)

    X_train = X_samples[train_idx]
    y_train = y_samples[train_idx]
    X_valid = X_samples[valid_idx]
    y_valid = y_samples[valid_idx]

    print(f"  训练窗口: {len(train_idx)}  验证窗口: {len(valid_idx)}")

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    valid_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_valid), torch.FloatTensor(y_valid)),
        batch_size=BATCH_SIZE, shuffle=False
    )

    adj_tensor = torch.FloatTensor(adj)
    print(f"  DataLoader: train={len(train_loader)} batches, valid={len(valid_loader)} batches")
    return train_loader, valid_loader, adj_tensor, scaler


def _load_adj():
    """对称归一化邻接矩阵"""
    adj_raw = pd.read_csv(ADJ_PATH, index_col=0).values.astype(np.float32)
    np.fill_diagonal(adj_raw, 1.0)
    degree = adj_raw.sum(axis=1)
    d_sqrt_inv = np.where(degree > 0, 1.0 / np.sqrt(degree), 0)
    d_inv = np.diag(d_sqrt_inv)
    adj_norm = d_inv @ adj_raw @ d_inv
    print(f"  邻接矩阵: {adj_norm.shape}")
    return adj_norm


def _load_and_reshape():
    """将表格转为 (T, N, F) 张量"""
    df = pd.read_csv(DATA_PATH, parse_dates=['time_slot'])
    df = df.sort_values(['time_slot', 'stationID']).reset_index(drop=True)

    all_slots = df['time_slot'].unique()
    n_slots = len(all_slots)
    station_ids = sorted(df['stationID'].unique())

    X_list, y_list, dates_list = [], [], []
    for ts in all_slots:
        slot_data = df[df['time_slot'] == ts].set_index('stationID')
        slot_data = slot_data.reindex(station_ids, fill_value=0)
        X_list.append(slot_data[FEATURE_COLS].values.astype(np.float32))
        y_list.append(slot_data[TARGET_COLS].values.astype(np.float32))
        dates_list.append(str(ts.date()))

    X_3d = np.stack(X_list, axis=0)
    y_3d = np.stack(y_list, axis=0)
    print(f"  3D 张量: X={X_3d.shape}, y={y_3d.shape}, slots={n_slots}")
    return X_3d, y_3d, dates_list


def _normalize(X_3d):
    """Z-score 标准化（基于训练集统计量）"""
    T, N, F = X_3d.shape
    n_train_slots = 22 * SLOTS_PER_DAY
    X_train_part = X_3d[:n_train_slots].reshape(-1, F)
    mean = X_train_part.mean(axis=0)
    std  = X_train_part.std(axis=0) + 1e-8
    X_scaled = (X_3d - mean) / std
    return X_scaled, {'mean': mean, 'std': std}


def _sliding_window(X, y):
    """(T, N, F) → (windows, T_hist, N, F)"""
    T = X.shape[0]
    X_w, y_w = [], []
    for i in range(0, T - T_HIST - T_PRED + 1, 1):
        X_w.append(X[i:i + T_HIST])
        y_w.append(y[i + T_HIST:i + T_HIST + T_PRED])
    print(f"  滑动窗口: {len(X_w)} 个样本")
    return np.stack(X_w, axis=0), np.stack(y_w, axis=0)


def build_prediction_input(last_day, scaler):
    """构建 closed-loop 预测输入：只用最后一天的最后 T_HIST 个真实时段"""
    df = pd.read_csv(DATA_PATH, parse_dates=['time_slot'])
    df = df.sort_values(['time_slot', 'stationID']).reset_index(drop=True)
    station_ids = sorted(df['stationID'].unique())
    last_day_data = df[df['date_str'] == last_day]
    last_slots = sorted(last_day_data['time_slot'].unique())[-T_HIST:]

    X_input = np.zeros((T_HIST, N_STATIONS, len(FEATURE_COLS)), dtype=np.float32)
    for t_idx, ts in enumerate(last_slots):
        slot_data = last_day_data[last_day_data['time_slot'] == ts]
        slot_data = slot_data.set_index('stationID').reindex(station_ids, fill_value=0)
        X_input[t_idx] = slot_data[FEATURE_COLS].values

    mean, std = scaler['mean'], scaler['std']
    X_input = (X_input - mean) / std

    last_ts = pd.Timestamp(last_day) + pd.Timedelta(hours=23, minutes=50)
    all_slots = pd.date_range(
        last_ts + pd.Timedelta(minutes=10),
        last_ts + pd.Timedelta(days=1),
        freq='10min'
    )

    adj = _load_adj()
    return torch.FloatTensor(X_input).unsqueeze(0), torch.FloatTensor(adj), all_slots


def autoregressive_predict(model, init_input, adj, n_steps, scaler):
    """自回归预测：用预测值滚动，不偷看真实数据"""
    device = next(model.parameters()).device
    model.eval()
    init_input = init_input.to(device)
    adj = adj.to(device)

    in_idx  = FEATURE_COLS.index('inNums')
    out_idx = FEATURE_COLS.index('outNums')
    current_input = init_input.clone()
    all_preds = []

    with torch.no_grad():
        for step in range(0, n_steps, T_PRED):
            pred = model(current_input, adj)
            pred_np = np.maximum(pred.cpu().numpy()[0], 0)
            actual = min(T_PRED, n_steps - step)
            all_preds.append(pred_np[:actual])

            if step + T_PRED < n_steps:
                new_input = current_input[:, T_PRED:, :, :].clone()
                last_feat = current_input[0, -1, :, :].cpu().numpy()
                for p in range(min(T_PRED, n_steps - step)):
                    new_feat = last_feat.copy()
                    new_feat[:, in_idx] = pred_np[p, :, 0]
                    new_feat[:, out_idx] = pred_np[p, :, 1]
                    new_input = torch.cat([
                        new_input,
                        torch.FloatTensor(new_feat).unsqueeze(0).unsqueeze(0).to(device)
                    ], dim=1)
                if new_input.shape[1] > T_HIST:
                    new_input = new_input[:, -T_HIST:, :, :]
                current_input = new_input

    return np.concatenate(all_preds, axis=0)
