# -*- coding: utf-8 -*-
"""
全三线探索性数据分析
- 各线路客流量对比
- 站点流量排名
- 线路间拓扑关系
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DATA_PATH = r"d:\作业\py展示\AST\data\all_lines_full.csv"
MAP_PATH  = r"d:\作业\py展示\AST\data\station_line_map.csv"
ADJ_PATH  = r"d:\作业\py展示\AST\data\adj_matrix.csv"
OUT_DIR   = r"d:\作业\py展示\AST\results"
os.makedirs(OUT_DIR, exist_ok=True)

PICK_DATE = "2019-01-07"

print("加载数据...")
df = pd.read_csv(DATA_PATH, parse_dates=['time_slot'])
df_map = pd.read_csv(MAP_PATH)
adj = pd.read_csv(ADJ_PATH, index_col=0)

line_colors = {'A': '#FF6B6B', 'B': '#4D96FF', 'C': '#6BCB77'}

# ============================================================
# 图1: 三线每日总客流量趋势
# ============================================================
fig1, ax1 = plt.subplots(figsize=(14, 5))
daily = df.groupby(['date_str', 'lineID'])[['inNums', 'outNums']].sum().reset_index()
daily['total'] = daily['inNums'] + daily['outNums']

for line in ['A', 'B', 'C']:
    ld = daily[daily['lineID'] == line]
    ax1.plot(range(len(ld)), ld['total'], 'o-', color=line_colors[line],
             linewidth=2, markersize=4, label=f'{line}线')

ax1.set_xticks(range(0, 25, 2))
ax1.set_xticklabels([f'01-{d+1:02d}' for d in range(0, 25, 2)], rotation=45, fontsize=8)
ax1.set_ylabel('每日总客流量', fontsize=11)
ax1.set_title('三线每日总客流量变化 (01-01~01-25)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, '01_daily_flow.png'), dpi=150)
plt.close(fig1)
print("  [1] 每日流量趋势 → 01_daily_flow.png")

# ============================================================
# 图2: 各线路10分钟客流量趋势（选一天）
# ============================================================
fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
fig2.suptitle(f'三线 10分钟进站/出站趋势 ({PICK_DATE})', fontsize=14, fontweight='bold')

df_day = df[df['date_str'] == PICK_DATE]
for line in ['A', 'B', 'C']:
    ld = df_day[df_day['lineID'] == line].groupby('time_slot')
    ax2a.plot(ld['time_slot'].first(), ld['inNums'].sum(),
              color=line_colors[line], linewidth=1.5, label=f'{line}线')
    ax2b.plot(ld['time_slot'].first(), ld['outNums'].sum(),
              color=line_colors[line], linewidth=1.5, label=f'{line}线')

ax2a.set_ylabel('进站人次/10min', fontsize=10)
ax2a.legend(fontsize=9)
ax2a.grid(True, alpha=0.3)
ax2a.set_title('进站', fontsize=12)

ax2b.set_ylabel('出站人次/10min', fontsize=10)
ax2b.legend(fontsize=9)
ax2b.grid(True, alpha=0.3)
ax2b.set_title('出站', fontsize=12)

ax2b.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax2b.xaxis.set_major_locator(mdates.HourLocator(interval=3))
plt.setp(ax2b.xaxis.get_majorticklabels(), rotation=30, fontsize=8)
fig2.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, '02_10min_flow.png'), dpi=150)
plt.close(fig2)
print("  [2] 三线10分钟趋势 → 02_10min_flow.png")

# ============================================================
# 图3: 站点总流量排名 Top30
# ============================================================
fig3, ax3 = plt.subplots(figsize=(12, 8))
station_total = df_day.groupby('stationID')[['inNums', 'outNums']].sum()
station_total['total'] = station_total['inNums'] + station_total['outNums']
station_total = station_total.sort_values('total', ascending=True).tail(30)

colors = [line_colors.get(
    df_map.set_index('stationID').loc[sid, 'lineID'], 'gray'
) for sid in station_total.index]

ax3.barh(range(30), station_total['total'], color=colors, alpha=0.85, edgecolor='white')
ax3.set_yticks(range(30))
ax3.set_yticklabels([f'{s}({df_map.set_index("stationID").loc[s,"lineID"]})'
                      for s in station_total.index], fontsize=9)
ax3.set_xlabel('总客流量', fontsize=11)
ax3.set_title(f'Top 30 繁忙站点 ({PICK_DATE})', fontsize=14, fontweight='bold')
# 图例
from matplotlib.patches import Patch
ax3.legend(handles=[Patch(color=c, label=l) for l, c in line_colors.items()],
           fontsize=10, loc='lower right')
fig3.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, '03_top30_stations.png'), dpi=150)
plt.close(fig3)
print("  [3] Top30站点排名 → 03_top30_stations.png")

# ============================================================
# 图4: 路网热力图
# ============================================================
fig4, ax4 = plt.subplots(figsize=(12, 10))
im = ax4.imshow(adj.values, cmap='Blues', aspect='auto', interpolation='none')
ax4.set_title('全线路网邻接矩阵 (81站)', fontsize=14, fontweight='bold')
ax4.set_xlabel('站点 ID')
ax4.set_ylabel('站点 ID')

# 标注线路分界线
b_line_end = max(line_stations_ := {'A': 14, 'B': 34, 'C': 32}, key=lambda k: 0) or 0
# 用 station_line 映射
from collections import Counter
line_order = sorted(set(df_map['lineID']))
cum = 0
for line in line_order:
    cnt = (df_map['lineID'] == line).sum()
    ax4.axhline(cum + cnt - 0.5, color='red', linewidth=1.5, linestyle='--')
    ax4.axvline(cum + cnt - 0.5, color='red', linewidth=1.5, linestyle='--')
    ax4.text(cum + cnt/2, len(df_map)+1, f'{line}线', ha='center', fontsize=10,
             fontweight='bold', color='red')
    cum += cnt

plt.colorbar(im, ax=ax4, shrink=0.85, label='连接 (0/1)')
fig4.tight_layout()
fig4.savefig(os.path.join(OUT_DIR, '04_adj_heatmap.png'), dpi=150)
plt.close(fig4)
print("  [4] 路网热力图 → 04_adj_heatmap.png")

# ============================================================
# 统计摘要
# ============================================================
print("\n" + "=" * 60)
print("统计摘要")
print("=" * 60)
for line in ['A', 'B', 'C']:
    ld = df[df['lineID'] == line]
    print(f"  {line}线: {ld['stationID'].nunique()}站, "
          f"日均进站 {ld['inNums'].sum()/25:,.0f}, "
          f"日均出站 {ld['outNums'].sum()/25:,.0f}")

print(f"\n✅ EDA 完成: 4 张图 → {OUT_DIR}/")
