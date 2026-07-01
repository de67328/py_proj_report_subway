# -*- coding: utf-8 -*-
"""
结果可视化
- 训练曲线
- 预测 vs 实际
- 误差分布
- 各站点 MAE
- GCN 嵌入 PCA
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = r"d:\作业\py展示\AST\results"
os.makedirs(OUT_DIR, exist_ok=True)


def plot_training_curves(history):
    """训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('ASTGCN 训练曲线', fontsize=14, fontweight='bold')

    ax1.plot(history['train_loss'], color='#4D96FF', linewidth=1.5)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Train Loss (MAE)')
    ax1.set_title('训练损失', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2.plot(history['val_mae_in'], label='进站 MAE', color='#4ECDC4', linewidth=1.5)
    ax2.plot(history['val_mae_out'], label='出站 MAE', color='#FF6B6B', linewidth=1.5)
    ax2.plot(history['val_mae_avg'], label='平均 MAE', color='#333333',
             linewidth=2, linestyle='--')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('MAE')
    ax2.set_title('验证 MAE', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'training_curves.png'), dpi=150)
    plt.close(fig)
    print("  → training_curves.png")


def plot_predictions(preds, targets):
    """预测 vs 实际散点 + 误差分布"""
    pred_in  = np.maximum(preds[:, :, :, 0].reshape(-1), 0)
    pred_out = np.maximum(preds[:, :, :, 1].reshape(-1), 0)
    true_in  = targets[:, :, :, 0].reshape(-1)
    true_out = targets[:, :, :, 1].reshape(-1)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('ASTGCN 预测结果分析', fontsize=15, fontweight='bold')

    # 进站散点
    mx = max(true_in.max(), pred_in.max()) * 1.05
    ax1.scatter(true_in, pred_in, alpha=0.1, s=2, c='#4D96FF', edgecolors='none')
    ax1.plot([0, mx], [0, mx], 'r--', alpha=0.5)
    ax1.set_xlim(0, mx); ax1.set_ylim(0, mx)
    ax1.set_xlabel('实际进站'); ax1.set_ylabel('预测进站')
    ax1.set_title('进站: 实际 vs 预测', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 出站散点
    mx = max(true_out.max(), pred_out.max()) * 1.05
    ax2.scatter(true_out, pred_out, alpha=0.1, s=2, c='#FF6B6B', edgecolors='none')
    ax2.plot([0, mx], [0, mx], 'r--', alpha=0.5)
    ax2.set_xlim(0, mx); ax2.set_ylim(0, mx)
    ax2.set_xlabel('实际出站'); ax2.set_ylabel('预测出站')
    ax2.set_title('出站: 实际 vs 预测', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 进站误差分布
    err_in = pred_in - true_in
    ax3.hist(err_in, bins=100, color='#4D96FF', alpha=0.7, edgecolor='white')
    ax3.axvline(0, color='red', linestyle='--')
    ax3.axvline(err_in.mean(), color='orange', linestyle='-',
                label=f'均值={err_in.mean():.2f}')
    ax3.set_xlabel('预测 - 实际 (进站)')
    ax3.set_title('进站误差分布', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)

    # 出站误差分布
    err_out = pred_out - true_out
    ax4.hist(err_out, bins=100, color='#FF6B6B', alpha=0.7, edgecolor='white')
    ax4.axvline(0, color='red', linestyle='--')
    ax4.axvline(err_out.mean(), color='orange', linestyle='-',
                label=f'均值={err_out.mean():.2f}')
    ax4.set_xlabel('预测 - 实际 (出站)')
    ax4.set_title('出站误差分布', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'predictions_analysis.png'), dpi=150)
    plt.close(fig)
    print("  → predictions_analysis.png")


def plot_gcn_embeddings(model, adj, scaler):
    """GCN 嵌入 PCA 可视化"""
    from graph_utils import build_graph_data
    _, valid_loader, _, _ = build_graph_data()

    device = next(model.parameters()).device
    model.eval()
    all_embs = []
    all_stations = []

    with torch.no_grad():
        for xb, _ in valid_loader:
            xb = xb.to(device)
            emb = model.extract_embeddings(xb, adj.to(device))
            # (B, T, N, C) → 取最后时刻 → (B, N, C)
            emb_last = emb[:, -1, :, :].cpu().numpy()
            all_embs.append(emb_last)
            all_stations.extend(range(80))

    embs = np.concatenate(all_embs, axis=0)  # (samples, N, C)
    # 每个站点的平均嵌入
    N = 80
    C = embs.shape[-1]
    station_embs = embs.reshape(-1, N, C).mean(axis=0)  # (N, C)

    pca = PCA(n_components=2, random_state=42)
    emb_2d = pca.fit_transform(station_embs)

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=range(N), cmap='viridis',
                    s=80, edgecolors='black', linewidth=0.5)
    for i in range(N):
        ax.annotate(str(i), (emb_2d[i, 0], emb_2d[i, 1]),
                    textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title('GCN 空间嵌入 PCA 可视化 (80站)', fontsize=14, fontweight='bold')
    plt.colorbar(sc, ax=ax, label='站点 ID')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'gcn_embeddings_pca.png'), dpi=150)
    plt.close(fig)
    print("  → gcn_embeddings_pca.png")
