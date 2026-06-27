# -*- coding: utf-8 -*-
"""
ASTGCN 训练脚本
- 训练循环 + 验证
- Early Stopping
- MAE 评估
- 模型保存
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error
import time
import os

from graph_utils import build_graph_data, T_HIST, T_PRED
from astgcn_model import ASTGCN, count_parameters

MODEL_SAVE_PATH = r"d:\作业\py展示\programme\models\astgcn\astgcn_best.pth"
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# ============================================================
# 超参数
# ============================================================
EPOCHS = 80
LR = 0.001
WEIGHT_DECAY = 1e-4
HIDDEN_CHANNELS = 32
NUM_BLOCKS = 2
PATIENCE = 15  # early stopping


def train_astgcn():
    """训练 ASTGCN，返回 (results_dict, predictions)"""
    print("\n" + "=" * 60)
    print("  ASTGCN 训练")
    print("=" * 60)

    # ---- 1. 数据 ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  设备: {device}")

    train_loader, valid_loader, adj, scaler_dict = build_graph_data()
    adj = adj.to(device)

    # ---- 2. 模型 ----
    model = ASTGCN(
        num_nodes=34,
        in_channels=7,
        hidden_channels=HIDDEN_CHANNELS,
        num_timesteps=T_HIST,
        num_blocks=NUM_BLOCKS,
        pred_timesteps=T_PRED,
        out_channels=2,
    ).to(device)

    n_params = count_parameters(model)
    print(f"  模型参数量: {n_params:,}")

    # ---- 3. 优化器 & 损失 ----
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.L1Loss()  # MAE

    # ---- 4. 训练循环 ----
    best_val_mae = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_mae_in': [], 'val_mae_out': [], 'val_mae_avg': []}

    print(f"\n  Epochs: {EPOCHS} | LR: {LR} | Hidden: {HIDDEN_CHANNELS}")
    print(f"  Patience: {PATIENCE} | Blocks: {NUM_BLOCKS}")
    print("-" * 55)

    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        # ---- 训练 ----
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb, adj)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # ---- 验证 ----
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb, adj)
                all_preds.append(pred.cpu().numpy())
                all_targets.append(yb.cpu().numpy())

        preds = np.concatenate(all_preds, axis=0)     # (samples, T_pred, N, 2)
        targets = np.concatenate(all_targets, axis=0)

        # 展平计算 MAE
        pred_in = preds[:, :, :, 0].reshape(-1)
        pred_out = preds[:, :, :, 1].reshape(-1)
        true_in = targets[:, :, :, 0].reshape(-1)
        true_out = targets[:, :, :, 1].reshape(-1)

        # 确保非负
        pred_in = np.maximum(pred_in, 0)
        pred_out = np.maximum(pred_out, 0)

        mae_in = mean_absolute_error(true_in, pred_in)
        mae_out = mean_absolute_error(true_out, pred_out)
        mae_avg = (mae_in + mae_out) / 2

        history['val_mae_in'].append(mae_in)
        history['val_mae_out'].append(mae_out)
        history['val_mae_avg'].append(mae_avg)

        scheduler.step(mae_avg)

        # 打印
        marker = ''
        if mae_avg < best_val_mae:
            best_val_mae = mae_avg
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            marker = ' *'
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | "
                  f"Val MAE: in={mae_in:.2f} out={mae_out:.2f} avg={mae_avg:.2f}{marker}")

        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (best: epoch {best_epoch}, MAE={best_val_mae:.2f})")
            break

    train_time = time.time() - t_start

    # ---- 5. 加载最佳模型并最终评估 ----
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))

    # 对每个预测时段单独评估
    print(f"\n  各预测时段 MAE (T+1 ~ T+{T_PRED}):")
    print(f"  {'时段':>6} {'MAE_in':>8} {'MAE_out':>8} {'MAE_avg':>8}")
    print(f"  {'-'*32}")
    per_step_mae = []
    for t in range(T_PRED):
        pi = np.maximum(preds[:, t, :, 0].reshape(-1), 0)
        po = np.maximum(preds[:, t, :, 1].reshape(-1), 0)
        ti = targets[:, t, :, 0].reshape(-1)
        to = targets[:, t, :, 1].reshape(-1)
        mi = mean_absolute_error(ti, pi)
        mo = mean_absolute_error(to, po)
        per_step_mae.append((mi, mo))
        print(f"  T+{t+1:>3} {mi:>8.2f} {mo:>8.2f} {(mi+mo)/2:>8.2f}")

    result = {
        'name': 'ASTGCN',
        'mae_in': mae_in,
        'mae_out': mae_out,
        'mae_avg': best_val_mae,
        'time': train_time,
        'params': n_params,
        'best_epoch': best_epoch,
    }

    # 返回第一个预测时段的预测值（与 XGBoost 可比）
    predictions = {
        'pred_in': np.maximum(preds[:, 0, :, 0].reshape(-1), 0),
        'pred_out': np.maximum(preds[:, 0, :, 1].reshape(-1), 0),
    }

    print(f"\n  ✅ ASTGCN 训练完成: MAE_avg={best_val_mae:.2f} | 耗时 {train_time:.1f}s")
    print(f"  最佳模型: {MODEL_SAVE_PATH}")

    return result, predictions


# ============================================================
# 独立运行
# ============================================================
if __name__ == '__main__':
    result, preds = train_astgcn()
    print(f"\n最终结果: {result}")
