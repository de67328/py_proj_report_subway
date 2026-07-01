# -*- coding: utf-8 -*-
"""
ASTGCN 训练脚本 — GPU 优先
- 训练循环 + 验证
- Early Stopping + 学习率调度
- 模型保存 + 训练历史记录
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error
import time
import os
import signal
from tqdm import tqdm

from graph_utils import build_graph_data, T_HIST, T_PRED
from astgcn_model import ASTGCN, count_parameters

MODEL_DIR = r"d:\作业\py展示\AST\models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# 超参数
# ============================================================
CONFIG = {
    'hidden_channels': 64,
    'num_blocks': 3,
    'dropout': 0.1,
    'lr': 0.001,
    'weight_decay': 1e-4,
    'epochs': 150,
    'patience': 20,
}


def train(config=None):
    """训练 ASTGCN，返回 (result_dict, history)"""
    if config is None:
        config = CONFIG

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  ASTGCN 训练 — 设备: {device}")
    print(f"{'='*60}")
    print(f"  超参数: {config}")

    # ---- 数据 ----
    train_loader, valid_loader, adj, scaler = build_graph_data()
    adj = adj.to(device)

    # ---- 模型 ----
    model = ASTGCN(
        num_nodes=80,
        in_channels=7,
        hidden_channels=config['hidden_channels'],
        num_timesteps=T_HIST,
        num_blocks=config['num_blocks'],
        pred_timesteps=T_PRED,
        out_channels=2,
        dropout=config['dropout'],
    ).to(device)

    n_params = count_parameters(model)
    print(f"  参数量: {n_params:,}")

    # ---- 优化器 & 损失 ----
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8
    )
    criterion = nn.L1Loss()

    # ---- 训练循环 ----
    best_val_mae = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_mae_in': [], 'val_mae_out': [],
               'val_mae_avg': [], 'lr': []}

    # 中断处理：Ctrl+C 时自动保存
    interrupted = False

    def save_checkpoint(filename='astgcn_interrupt.pt'):
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': config,
            'history': history,
            'epoch': epoch,
            'best_val_mae': best_val_mae,
        }, os.path.join(MODEL_DIR, filename))
        print(f"\n  💾 已保存中断检查点: {filename}")

    def signal_handler(sig, frame):
        nonlocal interrupted
        interrupted = True
        print("\n\n  ⚠ 收到中断信号，正在保存...")
        save_checkpoint()
        print("  可恢复训练: train(resume='astgcn_interrupt.pt')\n")

    signal.signal(signal.SIGINT, signal_handler)

    t_start = time.time()

    for epoch in range(1, config['epochs'] + 1):
        if interrupted:
            break
        # --- 训练 ---
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}", leave=False)
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb, adj)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.1f}'})

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # --- 验证 ---
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb, adj)
                all_preds.append(pred.cpu().numpy())
                all_targets.append(yb.cpu().numpy())

        preds   = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        pred_in   = np.maximum(preds[:, :, :, 0].reshape(-1), 0)
        pred_out  = np.maximum(preds[:, :, :, 1].reshape(-1), 0)
        true_in   = targets[:, :, :, 0].reshape(-1)
        true_out  = targets[:, :, :, 1].reshape(-1)

        mae_in  = mean_absolute_error(true_in, pred_in)
        mae_out = mean_absolute_error(true_out, pred_out)
        mae_avg = (mae_in + mae_out) / 2

        history['val_mae_in'].append(mae_in)
        history['val_mae_out'].append(mae_out)
        history['val_mae_avg'].append(mae_avg)

        scheduler.step(mae_avg)

        marker = ''
        if mae_avg < best_val_mae:
            best_val_mae = mae_avg
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'history': history,
                'scaler': scaler,
            }, os.path.join(MODEL_DIR, 'astgcn_best.pt'))
            marker = ' ★'
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Train Loss: {avg_train_loss:7.1f} | "
                  f"Val MAE: in={mae_in:6.2f} out={mae_out:6.2f} "
                  f"avg={mae_avg:6.2f}{marker}")

        if patience_counter >= config['patience']:
            print(f"\n  Early stopping @ epoch {epoch} (best: {best_epoch}, "
                  f"MAE_avg={best_val_mae:.2f})")
            break

    train_time = time.time() - t_start

    # ---- 加载最佳模型 ----
    ckpt = torch.load(os.path.join(MODEL_DIR, 'astgcn_best.pt'),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    # 分步 MAE
    print(f"\n  各预测时段 MAE (T+1 ~ T+{T_PRED}):")
    print(f"  {'时段':>6} {'MAE_in':>8} {'MAE_out':>8} {'MAE_avg':>8}")
    for t in range(T_PRED):
        pi = np.maximum(preds[:, t, :, 0].reshape(-1), 0)
        po = np.maximum(preds[:, t, :, 1].reshape(-1), 0)
        ti = targets[:, t, :, 0].reshape(-1)
        to = targets[:, t, :, 1].reshape(-1)
        print(f"  T+{t+1:>3} {mean_absolute_error(ti, pi):>8.2f} "
              f"{mean_absolute_error(to, po):>8.2f} "
              f"{(mean_absolute_error(ti, pi)+mean_absolute_error(to, po))/2:>8.2f}")

    result = {
        'name': 'ASTGCN',
        'mae_in': mae_in,
        'mae_out': mae_out,
        'mae_avg': best_val_mae,
        'time': train_time,
        'params': n_params,
        'best_epoch': best_epoch,
        'device': str(device),
    }

    print(f"\n  ✅ 训练完成: MAE_avg={best_val_mae:.2f} | "
          f"耗时 {train_time:.0f}s | 参数 {n_params:,}")
    return result, history, model, (preds, targets)
