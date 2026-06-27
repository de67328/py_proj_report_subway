# -*- coding: utf-8 -*-
"""
GCN 嵌入可视化
- 用 PCA 将 32 维嵌入降到 2 维
- 按站点着色，看邻近站点是否聚类在一起
- 计算站点间嵌入余弦相似度热力图
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

EMBED_PATH = r"d:\作业\py展示\data\gcn_embeddings.csv"
OUTPUT_DIR = r"d:\作业\py展示\results_viz"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. 加载嵌入
# ============================================================
print("加载 GCN 嵌入...")
df_emb = pd.read_csv(EMBED_PATH)
emb_cols = [c for c in df_emb.columns if c.startswith('gcn_emb_')]
print(f"  嵌入维度: {len(emb_cols)}")

# 每个站点的平均嵌入（跨所有时段）
station_emb = df_emb.groupby('stationID')[emb_cols].mean()
emb_matrix = station_emb.values  # (34, 32)
print(f"  站点嵌入矩阵: {emb_matrix.shape}")

# ============================================================
# 2. PCA 降维可视化
# ============================================================
pca = PCA(n_components=2, random_state=42)
emb_2d = pca.fit_transform(emb_matrix)
print(f"  PCA 解释方差: {pca.explained_variance_ratio_}")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('GCN 空间嵌入 PCA 可视化', fontsize=16, fontweight='bold')

# 2a: 按站点 ID 着色
ax1 = axes[0]
sc1 = ax1.scatter(emb_2d[:, 0], emb_2d[:, 1], c=station_emb.index, cmap='viridis',
                  s=100, edgecolors='black', linewidth=0.5)
for i, sid in enumerate(station_emb.index):
    ax1.annotate(str(sid), (emb_2d[i, 0], emb_2d[i, 1]),
                 textcoords="offset points", xytext=(5, 5), fontsize=8)
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax1.set_title('站点 ID 着色', fontsize=13, fontweight='bold')
plt.colorbar(sc1, ax=ax1, label='站点 ID')

# 2b: 按站点 ID 范围分组着色（0-11/12-23/24-33 对应上游/中游/下游）
ax2 = axes[1]
groups = []
for sid in station_emb.index:
    if sid <= 11:
        groups.append(0)
    elif sid <= 23:
        groups.append(1)
    else:
        groups.append(2)
sc2 = ax2.scatter(emb_2d[:, 0], emb_2d[:, 1], c=groups, cmap='Set1',
                  s=100, edgecolors='black', linewidth=0.5)
for i, sid in enumerate(station_emb.index):
    ax2.annotate(str(sid), (emb_2d[i, 0], emb_2d[i, 1]),
                 textcoords="offset points", xytext=(5, 5), fontsize=8)
ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax2.set_title('线段分组着色 (0-11/12-23/24-33)', fontsize=13, fontweight='bold')
cbar = plt.colorbar(sc2, ax=ax2, ticks=[0, 1, 2])
cbar.ax.set_yticklabels(['上游(0-11)', '中游(12-23)', '下游(24-33)'])

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '10_gcn_embeddings_pca.png'), dpi=150)
plt.close(fig)
print("  → 10_gcn_embeddings_pca.png")

# ============================================================
# 3. 站点嵌入余弦相似度热力图
# ============================================================
sim_matrix = cosine_similarity(emb_matrix)

fig2, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(sim_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0.5, vmax=1.0)
ax.set_xticks(range(34))
ax.set_yticks(range(34))
ax.set_xticklabels(station_emb.index, fontsize=7)
ax.set_yticklabels(station_emb.index, fontsize=7)
ax.set_title('站点间 GCN 嵌入余弦相似度', fontsize=14, fontweight='bold')
ax.set_xlabel('站点 ID')
ax.set_ylabel('站点 ID')
plt.colorbar(im, ax=ax, shrink=0.85, label='余弦相似度')

# 标记相邻站点对
# 加载邻接矩阵
adj_full = pd.read_csv(
    r"d:\作业\py展示\Hangzhou-mobility-data-set\Metro_roadMap.csv",
    index_col=0
)
adj_full.index = adj_full.index.astype(int)
adj_full.columns = adj_full.columns.astype(int)
adj_b = adj_full.loc[list(range(34)), list(range(34))].values

# 在热力图上标注相邻站点的平均相似度 vs 非相邻
adj_pairs = []
non_adj_pairs = []
for i in range(34):
    for j in range(i+1, 34):
        if adj_b[i, j] == 1:
            adj_pairs.append(sim_matrix[i, j])
        else:
            non_adj_pairs.append(sim_matrix[i, j])

avg_adj = np.mean(adj_pairs) if adj_pairs else 0
avg_non = np.mean(non_adj_pairs) if non_adj_pairs else 0
ax.set_title(f'站点间 GCN 嵌入余弦相似度\n相邻站点平均={avg_adj:.3f}  非相邻平均={avg_non:.3f}',
             fontsize=13, fontweight='bold')

fig2.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, '11_gcn_similarity_heatmap.png'), dpi=150)
plt.close(fig2)
print("  → 11_gcn_similarity_heatmap.png")

# ============================================================
# 4. 相邻 vs 非相邻 相似度分布对比
# ============================================================
fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.hist(adj_pairs, bins=30, alpha=0.6, color='#FF6B6B', label=f'相邻站点 (n={len(adj_pairs)}, avg={avg_adj:.3f})')
ax3.hist(non_adj_pairs, bins=30, alpha=0.6, color='#4D96FF', label=f'非相邻站点 (n={len(non_adj_pairs)}, avg={avg_non:.3f})')
ax3.axvline(avg_adj, color='#FF6B6B', linestyle='--', linewidth=2)
ax3.axvline(avg_non, color='#4D96FF', linestyle='--', linewidth=2)
ax3.set_xlabel('余弦相似度', fontsize=11)
ax3.set_ylabel('站点对数量', fontsize=11)
ax3.set_title('GCN 嵌入：相邻 vs 非相邻站点相似度分布', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)

fig3.tight_layout()
fig3.savefig(os.path.join(OUTPUT_DIR, '12_gcn_adj_vs_nonadj.png'), dpi=150)
plt.close(fig3)
print("  → 12_gcn_adj_vs_nonadj.png")

print(f"\n✅ 3张 GCN 嵌入可视化已保存至 {OUTPUT_DIR}/")
