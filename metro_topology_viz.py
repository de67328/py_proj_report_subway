# -*- coding: utf-8 -*-
"""
地铁线路拓扑可视化
- 读取 Metro_roadMap.csv 邻接矩阵
- 从刷卡数据中获取 stationID → lineID 映射
- 用 networkx 绘制站点拓扑图，按线路着色
- 输出 A/B/C 线路站点列表
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import networkx as nx
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 读取路网邻接矩阵
# ============================================================
print("=" * 60)
print("读取路网矩阵...")
road_path = r"d:\作业\py展示\Hangzhou-mobility-data-set\Metro_roadMap.csv"
adj_df = pd.read_csv(road_path, index_col=0)
print(f"矩阵形状: {adj_df.shape} (81站 × 81站)")
print(f"连接边总数: {adj_df.values.sum() // 2} 条无向边")

# ============================================================
# 2. 从刷卡数据获取 stationID → lineID 映射
# ============================================================
print("\n读取刷卡数据获取站点-线路映射...")
record_path = r"d:\作业\py展示\Hangzhou-mobility-data-set\record_2019-01-01.csv"
df = pd.read_csv(
    record_path,
    dtype={'stationID': 'int16', 'lineID': 'category'},
    usecols=['stationID', 'lineID'],
)
# 每个站点取出现最多的线路（理论上每个站点只属于一条线路）
station_line_series = df.groupby('stationID')['lineID'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
station_line = station_line_series.to_dict()  # 转为普通 dict
del df  # 释放内存

print(f"站点-线路映射完成，共 {len(station_line)} 个站点")

# ============================================================
# 3. 构建 networkx 图
# ============================================================
G = nx.Graph()
station_ids = list(adj_df.index.astype(int))

# 添加节点
for sid in station_ids:
    line = station_line.get(sid, '?')
    G.add_node(sid, line=line)

# 添加边
for i, row_i in enumerate(station_ids):
    for j, row_j in enumerate(station_ids):
        if j > i and adj_df.iloc[i, j] == 1:
            G.add_edge(row_i, row_j)

print(f"图构建完成: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

# ============================================================
# 4. 线路颜色映射
# ============================================================
line_colors = {
    'A': '#FF6B6B',  # 红
    'B': '#4D96FF',  # 蓝
    'C': '#6BCB77',  # 绿
    '?': '#CCCCCC',
}
node_colors = [line_colors.get(G.nodes[n].get('line', '?'), '#CCCCCC') for n in G.nodes()]

# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(22, 10))
fig.suptitle('杭州地铁线路拓扑结构', fontsize=18, fontweight='bold')

# --- 5a: 拓扑图 ---
ax1 = axes[0]

# 计算布局：优先使用 Kamada-Kawai 让连通分量自然展开
pos = nx.kamada_kawai_layout(G)

nx.draw_networkx_edges(G, pos, ax=ax1, edge_color='gray', alpha=0.4, width=1.2)
nx.draw_networkx_nodes(G, pos, ax=ax1, node_color=node_colors, node_size=180,
                       edgecolors='black', linewidths=0.5)
# 标签
labels = {n: str(n) for n in G.nodes()}
nx.draw_networkx_labels(G, pos, ax=ax1, labels=labels, font_size=7, font_color='black')

# 图例
from matplotlib.lines import Line2D
line_station_counts = {line: sum(1 for v in station_line.values() if v == line) for line in ['A', 'B', 'C']}
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=line_colors['A'],
           markersize=12, label=f'A线 ({line_station_counts["A"]}站)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=line_colors['B'],
           markersize=12, label=f'B线 ({line_station_counts["B"]}站)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=line_colors['C'],
           markersize=12, label=f'C线 ({line_station_counts["C"]}站)'),
]
ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)
ax1.set_title('站点连接拓扑（按线路着色）', fontsize=14, fontweight='bold')
ax1.axis('off')

# --- 5b: 站点-线路对照表 ---
ax2 = axes[1]
ax2.axis('off')
ax2.set_title('站点 → 线路 对照表', fontsize=14, fontweight='bold', y=1.02)

# 分线路整理
lines_data = {'A': [], 'B': [], 'C': []}
for sid in sorted(station_ids):
    line = station_line.get(sid, '?')
    if line in lines_data:
        lines_data[line].append(sid)

# 绘制表格
table_data = []
max_len = max(len(v) for v in lines_data.values())
for i in range(max_len):
    row = []
    for line in ['A', 'B', 'C']:
        stations = lines_data[line]
        row.append(str(stations[i]) if i < len(stations) else '')
    table_data.append(row)

# 表格背景色
col_colors = [line_colors['A'], line_colors['B'], line_colors['C']]
cell_colors = []
for i in range(max_len):
    row_colors = []
    for j, line in enumerate(['A', 'B', 'C']):
        if i < len(lines_data[line]):
            row_colors.append(line_colors[line] + '40')  # 加透明度
        else:
            row_colors.append('#FAFAFA')
    cell_colors.append(row_colors)

table = ax2.table(
    cellText=table_data,
    colLabels=['A线站点', 'B线站点', 'C线站点'],
    cellLoc='center',
    loc='center',
    cellColours=cell_colors,
    colColours=[line_colors['A'] + '80', line_colors['B'] + '80', line_colors['C'] + '80'],
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.3)

for key, cell in table.get_celld().items():
    cell.set_edgecolor('#DDDDDD')
    if key[0] == 0:  # 表头
        cell.set_text_props(fontweight='bold', color='white')

# ============================================================
# 6. 保存图片
# ============================================================
os.makedirs(r"d:\作业\py展示\pic", exist_ok=True)
output_path = r"d:\作业\py展示\pic\metro_topology.png"
plt.tight_layout()
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n图表已保存至: {output_path}")

# ============================================================
# 7. 打印各线路站点列表
# ============================================================
print("\n" + "=" * 60)
for line in ['A', 'B', 'C']:
    stations = sorted(lines_data[line])
    print(f"\n【{line}线】共 {len(stations)} 站:")
    # 每10个换行
    for i in range(0, len(stations), 10):
        print("  " + ", ".join(str(s) for s in stations[i:i+10]))

# ============================================================
# 8. 连通分量分析
# ============================================================
print("\n" + "=" * 60)
components = list(nx.connected_components(G))
print(f"连通分量数: {len(components)}")
for i, comp in enumerate(components):
    comp_list = sorted(comp)
    lines_in_comp = set(station_line.get(s, '?') for s in comp_list)
    print(f"  分量{i+1}: {len(comp_list)}站, 涉及线路: {lines_in_comp}")
    print(f"    站点: {comp_list}")

# 孤立节点
isolated = [n for n in G.nodes() if G.degree(n) == 0]
if isolated:
    print(f"\n⚠ 孤立站点(无连接): {isolated}")

plt.show()
print("\n完成！")
