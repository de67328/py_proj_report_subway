# -*- coding: utf-8 -*-
"""
GCN 嵌入提取模块
- 加载训练好的 ASTGCN 模型
- 对全部数据提取每个 (station, time_slot) 的空间嵌入
- 保存为 CSV，供 XGBoost 等树模型作为额外特征使用
"""

import numpy as np
import pandas as pd
import torch
import os

from astgcn_model import ASTGCN, count_parameters
from graph_utils import _load_adj_matrix, _load_and_reshape, _normalize, T_HIST

ASTGCN_PATH = r"d:\作业\py展示\programme\models\astgcn\astgcn_best.pth"
EMBED_OUTPUT = r"d:\作业\py展示\data\gcn_embeddings.csv"


def extract_and_save_embeddings():
    """提取 GCN 嵌入并保存到 CSV"""
    print("=" * 50)
    print(" 提取 GCN 空间嵌入")
    print("=" * 50)

    # ---- 1. 加载数据 ----
    adj = _load_adj_matrix()
    X_3d, y_3d, dates = _load_and_reshape()
    X_scaled, scaler = _normalize(X_3d)

    T_total, N, F = X_scaled.shape
    print(f"  数据: T={T_total}, N={N}, F={F}")

    # ---- 2. 加载训练好的 ASTGCN ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ASTGCN(
        num_nodes=N, in_channels=F, hidden_channels=32,
        num_timesteps=T_HIST, num_blocks=2,
        pred_timesteps=3, out_channels=2,
    ).to(device)

    model.load_state_dict(torch.load(ASTGCN_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"  模型加载成功 ({count_parameters(model):,} params)")

    # ---- 3. 滑动窗口提取嵌入 ----
    adj_t = torch.FloatTensor(adj).to(device)
    all_embeddings = []  # 每个窗口: (T_hist, N, hidden) → 取最后一个时刻 (N, hidden)
    all_time_slots = []  # 对应的时间标签

    step = 6  # 每隔6个时段取一个窗口，避免过多重复
    for start in range(0, T_total - T_HIST + 1, step):
        x_win = X_scaled[start:start + T_HIST]  # (T_hist, N, F)
        x_tensor = torch.FloatTensor(x_win).unsqueeze(0).to(device)  # (1, T_hist, N, F)

        emb = model.extract_embeddings(x_tensor, adj_t)  # (1, T_hist, N, hidden)
        # 取最后一个时间步的嵌入
        emb_last = emb[0, -1, :, :].cpu().numpy()  # (N, hidden)

        all_embeddings.append(emb_last)
        all_time_slots.append(start + T_HIST - 1)  # 对应的时间索引

    print(f"  提取了 {len(all_embeddings)} 个窗口的嵌入")

    # ---- 4. 重建每个 (time_slot, stationID) 的嵌入 ----
    # 嵌入覆盖的时段可能不连续，需要插值或取最近
    emb_array = np.stack(all_embeddings, axis=0)  # (n_windows, N, hidden)
    hidden_dim = emb_array.shape[2]

    # 将时间索引映射回实际的 time_slot
    df_full = pd.read_csv(
        r"d:\作业\py展示\data\b_line_full.csv",
        parse_dates=['time_slot']
    )
    df_full = df_full.sort_values(['time_slot', 'stationID']).reset_index(drop=True)
    all_slots = df_full['time_slot'].unique()
    all_slot_indices = list(range(len(all_slots)))

    # 对每个时段，找最近的嵌入窗口
    embedding_map = {}  # slot_idx → (N, hidden)
    for idx in all_slot_indices:
        # 找最近的窗口
        nearest = min(all_time_slots, key=lambda x: abs(x - idx))
        emb_idx = all_time_slots.index(nearest)
        embedding_map[idx] = emb_array[emb_idx]

    # ---- 5. 构建 DataFrame 并保存 ----
    rows = []
    for idx in all_slot_indices:
        emb = embedding_map[idx]
        ts = all_slots[idx]
        for sid in range(N):
            row = {'time_slot': ts, 'stationID': sid}
            for d in range(hidden_dim):
                row[f'gcn_emb_{d}'] = float(emb[sid, d])
            rows.append(row)

    df_emb = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(EMBED_OUTPUT), exist_ok=True)
    df_emb.to_csv(EMBED_OUTPUT, index=False, encoding='utf-8')
    print(f"  ✅ 嵌入已保存: {EMBED_OUTPUT} ({df_emb.shape[0]} rows × {df_emb.shape[1]} cols)")
    print(f"  GCN 嵌入维度: {hidden_dim}")

    return df_emb


if __name__ == '__main__':
    extract_and_save_embeddings()
