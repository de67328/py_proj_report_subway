# -*- coding: utf-8 -*-
"""
ACF/PACF 分析 — 确定最优输入窗口大小
- 分析各站点客流序列的自相关函数
- 找出显著自相关的时间滞后
- 可视化输出到 AST/results/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DATA_PATH = r"d:\作业\py展示\AST\data\all_lines_full.csv"
MAP_PATH  = r"d:\作业\py展示\AST\data\station_line_map.csv"
OUT_DIR   = r"d:\作业\py展示\AST\results"
os.makedirs(OUT_DIR, exist_ok=True)

MAX_LAGS = 144 * 2  # 最多看 2 天（2880 分钟 = 288 个时段）

print("加载数据...")
df = pd.read_csv(DATA_PATH, parse_dates=['time_slot'])
df_map = pd.read_csv(MAP_PATH)

# 只用训练集数据
df_train = df[df['date_str'] <= "2019-01-22"].copy()

# ============================================================
# 1. 选取代表性站点（每条线选流量最大站点）
# ============================================================
station_flow = df_train.groupby('stationID')['inNums'].mean()
top_stations = {}
for line in ['A', 'B', 'C']:
    line_stations = df_map[df_map['lineID'] == line]['stationID'].values
    line_flow = station_flow[station_flow.index.isin(line_stations)]
    top_stations[line] = line_flow.idxmax()

print(f"代表站点: A线={top_stations['A']}, B线={top_stations['B']}, C线={top_stations['C']}")

# ============================================================
# 2. 计算 ACF
# ============================================================
fig, axes = plt.subplots(3, 2, figsize=(18, 14))
fig.suptitle('客流序列 ACF/PACF 分析 (训练集 01-01~01-22)', fontsize=16, fontweight='bold')

line_colors = {'A': '#FF6B6B', 'B': '#4D96FF', 'C': '#6BCB77'}

for i, (line, sid) in enumerate(top_stations.items()):
    # 提取该站点的时间序列
    station_series = (
        df_train[df_train['stationID'] == sid]
        .sort_values('time_slot')['inNums']
        .values
    )

    # 计算 ACF
    acf_vals = acf(station_series, nlags=MAX_LAGS)
    pacf_vals = pacf(station_series, nlags=MAX_LAGS // 2)  # PACF 只看短滞后

    color = line_colors[line]

    # ACF 子图
    ax_acf = axes[i, 0]
    ax_acf.stem(range(len(acf_vals)), acf_vals, linefmt=color, markerfmt='o', basefmt='gray')
    ax_acf.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    # 95% 置信区间
    conf = 1.96 / np.sqrt(len(station_series))
    ax_acf.axhline(conf, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    ax_acf.axhline(-conf, color='red', linestyle='--', linewidth=0.8, alpha=0.5)

    # 标注关键滞后
    ax_acf.axvline(6, color='orange', linestyle=':', linewidth=1, alpha=0.7)
    ax_acf.axvline(12, color='orange', linestyle=':', linewidth=1, alpha=0.7)
    ax_acf.axvline(18, color='orange', linestyle=':', linewidth=1, alpha=0.7)
    ax_acf.axvline(144, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_acf.axvline(288, color='green', linestyle='--', linewidth=1.5, alpha=0.7)

    ax_acf.set_xlabel('滞后 (×10min)')
    ax_acf.set_ylabel('ACF')
    ax_acf.set_title(f'{line}线 站点{sid} 进站 ACF', fontsize=12, fontweight='bold')
    ax_acf.set_xlim(0, MAX_LAGS)

    # PACF 子图
    ax_pacf = axes[i, 1]
    ax_pacf.stem(range(len(pacf_vals)), pacf_vals, linefmt=color, markerfmt='o', basefmt='gray')
    ax_pacf.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax_pacf.axhline(conf, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    ax_pacf.axhline(-conf, color='red', linestyle='--', linewidth=0.8, alpha=0.5)

    ax_pacf.axvline(6, color='orange', linestyle=':', linewidth=1, alpha=0.7)
    ax_pacf.axvline(12, color='orange', linestyle=':', linewidth=1, alpha=0.7)
    ax_pacf.axvline(18, color='orange', linestyle=':', linewidth=1, alpha=0.7)

    ax_pacf.set_xlabel('滞后 (×10min)')
    ax_pacf.set_ylabel('PACF')
    ax_pacf.set_title(f'{line}线 站点{sid} 进站 PACF', fontsize=12, fontweight='bold')
    ax_pacf.set_xlim(0, MAX_LAGS // 2)

    # 打印关键滞后值
    print(f"\n{line}线 站点{sid}:")
    print(f"  ACF(6)={acf_vals[6]:.3f}  ACF(12)={acf_vals[12]:.3f}  "
          f"ACF(18)={acf_vals[18]:.3f}")
    print(f"  ACF(144)={acf_vals[144]:.3f}  ACF(288)={acf_vals[288]:.3f}")
    print(f"  PACF(6)={pacf_vals[6]:.3f}  PACF(12)={pacf_vals[12]:.3f}")

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'acf_pacf_analysis.png'), dpi=150)
plt.close(fig)
print(f"\n✅ ACF/PACF 图已保存")

# ============================================================
# 3. 汇总：所有站点的平均 ACF
# ============================================================
fig2, ax2 = plt.subplots(figsize=(14, 5))
fig2.suptitle('全80站平均 ACF（训练集）', fontsize=14, fontweight='bold')

all_acfs = []
for sid in sorted(df_train['stationID'].unique()):
    series = (df_train[df_train['stationID'] == sid]
              .sort_values('time_slot')['inNums'].values)
    if len(series) > MAX_LAGS:
        acf_vals = acf(series, nlags=MAX_LAGS)
        all_acfs.append(acf_vals)

mean_acf = np.mean(all_acfs, axis=0)
std_acf = np.std(all_acfs, axis=0)
lags = np.arange(len(mean_acf))

ax2.plot(lags, mean_acf, 'b-', linewidth=1.5, label='均值 ACF')
ax2.fill_between(lags, mean_acf - std_acf, mean_acf + std_acf,
                 alpha=0.2, color='blue')
ax2.axhline(0, color='gray', linestyle='--', linewidth=0.5)
conf = 1.96 / np.sqrt(len(all_acfs[0]))
ax2.axhline(conf, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
ax2.axhline(-conf, color='red', linestyle='--', linewidth=0.8, alpha=0.5)

# 标注
ax2.axvline(6, color='orange', linestyle=':', linewidth=1.5, label='T=6 (1h)')
ax2.axvline(12, color='orange', linestyle='--', linewidth=1.5, label='T=12 (2h)')
ax2.axvline(18, color='orange', linestyle='-.', linewidth=1.5, label='T=18 (3h)')
ax2.axvline(144, color='green', linestyle='--', linewidth=2, label='T=144 (1天)')

ax2.set_xlabel('滞后 (×10min)', fontsize=11)
ax2.set_ylabel('ACF', fontsize=11)
ax2.legend(fontsize=9, loc='upper right')
ax2.set_xlim(0, 200)

fig2.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, 'acf_mean_all_stations.png'), dpi=150)
plt.close(fig2)
print("✅ 全站均值 ACF 图已保存")

# ============================================================
# 4. 结论输出
# ============================================================
print("\n" + "=" * 60)
print("  ACF 分析结论")
print("=" * 60)
print(f"  ACF(6)  均值 = {mean_acf[6]:.4f}  — 1小时前相关程度")
print(f"  ACF(12) 均值 = {mean_acf[12]:.4f}  — 2小时前相关程度")
print(f"  ACF(18) 均值 = {mean_acf[18]:.4f}  — 3小时前相关程度")
print(f"  ACF(144)均值 = {mean_acf[144]:.4f}  — 1天前相关程度")

# 推荐窗口
print(f"\n  推荐 T_hist:")
if mean_acf[12] > 0.5:
    print(f"    12 (2小时) — ACF={mean_acf[12]:.3f}，强相关且窗口适中")
if mean_acf[18] > 0.4:
    print(f"    18 (3小时) — ACF={mean_acf[18]:.3f}，可考虑增大")
if mean_acf[144] > 0.3:
    print(f"    144 (1天)  — ACF={mean_acf[144]:.3f}，日周期性显著")
