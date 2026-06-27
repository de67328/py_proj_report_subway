# -*- coding: utf-8 -*-
"""
B线单日可视化 — 每张图独立输出
- 选取一天 (2019-01-07 周一) 的 B 线聚合数据
- 多角度展示各站点人流量随时间变化
- 每张图单独保存为 PNG
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

# ============================================================
# 配置
# ============================================================
DATA_PATH = r"d:\作业\py展示\data\b_line_full.csv"
PICK_DATE = "2019-01-07"  # 周一，工作日典型日
OUTPUT_DIR = r"d:\作业\py展示\pic"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 读取 & 筛选
# ============================================================
print(f"读取 {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH, parse_dates=['time_slot'])
df_day = df[df['date_str'] == PICK_DATE].copy()
print(f"选中日期: {PICK_DATE}, 共 {len(df_day)} 条记录")
print(f"总进站: {df_day['inNums'].sum():,}  总出站: {df_day['outNums'].sum():,}")

# ---- 预计算 ----
total_in  = df_day.groupby('time_slot')['inNums'].sum()
total_out = df_day.groupby('time_slot')['outNums'].sum()
station_total = df_day.groupby('stationID')[['inNums', 'outNums']].sum()
station_total['total'] = station_total['inNums'] + station_total['outNums']
station_total = station_total.sort_values('total', ascending=True)
is_peak = df_day['is_peak'] == 1
top5 = station_total.tail(5).index
top1 = station_total.index[-1]

# 站点颜色
cmap = plt.cm.tab20
station_colors = {sid: cmap(i % 20) for i, sid in enumerate(sorted(df_day['stationID'].unique()))}


# ============================================================
# 辅助：格式化 x 轴为时:分
# ============================================================
def fmt_xaxis(ax, interval=3):
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, fontsize=8)


# ============================================================
# 图1: 全线总进站/出站 趋势
# ============================================================
fig1, ax1 = plt.subplots(figsize=(14, 5))
ax1.fill_between(total_in.index, total_in.values, alpha=0.4, color='#4ECDC4', label='进站')
ax1.fill_between(total_out.index, total_out.values, alpha=0.4, color='#FF6B6B', label='出站')
ax1.plot(total_in.index, total_in.values, color='#2BA89E', linewidth=1.5)
ax1.plot(total_out.index, total_out.values, color='#D94F4F', linewidth=1.5)
ax1.set_title(f'B线全线 进站/出站 10分钟趋势 ({PICK_DATE} 周一)', fontsize=14, fontweight='bold')
ax1.set_ylabel('人次')
ax1.legend(loc='upper left', fontsize=10)
fmt_xaxis(ax1)
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig(os.path.join(OUTPUT_DIR, '01_full_trend.png'), dpi=150)
plt.close(fig1)
print("  [1/7] 全线趋势 → 01_full_trend.png")

# ============================================================
# 图2: 各站点总进站/出站排名
# ============================================================
fig2, ax2 = plt.subplots(figsize=(12, 10))
y_pos = range(len(station_total))
bar_h = 0.35
ax2.barh([p - bar_h/2 for p in y_pos], station_total['inNums'],
         height=bar_h, color='#4ECDC4', label='进站', alpha=0.85)
ax2.barh([p + bar_h/2 for p in y_pos], station_total['outNums'],
         height=bar_h, color='#FF6B6B', label='出站', alpha=0.85)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(station_total.index, fontsize=9)
ax2.set_title(f'B线 各站点总进站/出站排名 ({PICK_DATE})', fontsize=14, fontweight='bold')
ax2.set_xlabel('总人次')
ax2.legend(loc='lower right', fontsize=10)
fig2.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, '02_station_rank.png'), dpi=150)
plt.close(fig2)
print("  [2/7] 站点排名 → 02_station_rank.png")

# ============================================================
# 图3: 进站热力图 (站点 × 时段)
# ============================================================
fig3, ax3 = plt.subplots(figsize=(16, 8))
pivot_in = df_day.pivot_table(values='inNums', index='stationID', columns='slot_id', aggfunc='sum')
im3 = ax3.imshow(pivot_in.values, aspect='auto', cmap='YlOrRd', interpolation='bilinear')
ax3.set_title(f'B线 进站热力图 站点×时段 ({PICK_DATE})', fontsize=14, fontweight='bold')
ax3.set_ylabel('站点 ID', fontsize=11)
ax3.set_xlabel('时段 (每格10分钟)', fontsize=11)
ax3.set_yticks(range(0, 34, 2))
ax3.set_yticklabels(pivot_in.index[::2])
ax3.set_xticks(range(0, 144, 18))
ax3.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 3)])
plt.colorbar(im3, ax=ax3, shrink=0.85, label='进站人数')
fig3.tight_layout()
fig3.savefig(os.path.join(OUTPUT_DIR, '03_heatmap_in.png'), dpi=150)
plt.close(fig3)
print("  [3/7] 进站热力图 → 03_heatmap_in.png")

# ============================================================
# 图4: 出站热力图 (站点 × 时段)
# ============================================================
fig4, ax4 = plt.subplots(figsize=(16, 8))
pivot_out = df_day.pivot_table(values='outNums', index='stationID', columns='slot_id', aggfunc='sum')
im4 = ax4.imshow(pivot_out.values, aspect='auto', cmap='PuBuGn', interpolation='bilinear')
ax4.set_title(f'B线 出站热力图 站点×时段 ({PICK_DATE})', fontsize=14, fontweight='bold')
ax4.set_ylabel('站点 ID', fontsize=11)
ax4.set_xlabel('时段 (每格10分钟)', fontsize=11)
ax4.set_yticks(range(0, 34, 2))
ax4.set_yticklabels(pivot_out.index[::2])
ax4.set_xticks(range(0, 144, 18))
ax4.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 3)])
plt.colorbar(im4, ax=ax4, shrink=0.85, label='出站人数')
fig4.tight_layout()
fig4.savefig(os.path.join(OUTPUT_DIR, '04_heatmap_out.png'), dpi=150)
plt.close(fig4)
print("  [4/7] 出站热力图 → 04_heatmap_out.png")

# ============================================================
# 图5: Top5 站点 进站趋势对比
# ============================================================
fig5, ax5 = plt.subplots(figsize=(14, 6))
for sid in top5:
    s_data = df_day[df_day['stationID'] == sid].set_index('time_slot')['inNums']
    ax5.plot(s_data.index, s_data.values, linewidth=1.8,
             label=f'站点 {sid}', color=station_colors.get(sid, 'gray'), alpha=0.9)
ax5.set_title(f'B线 Top 5 繁忙站点 进站趋势对比 ({PICK_DATE})', fontsize=14, fontweight='bold')
ax5.set_ylabel('进站人次 (10min)')
ax5.legend(loc='upper left', fontsize=10, ncol=2)
fmt_xaxis(ax5)
ax5.grid(True, alpha=0.3)
fig5.tight_layout()
fig5.savefig(os.path.join(OUTPUT_DIR, '05_top5_in_trend.png'), dpi=150)
plt.close(fig5)
print("  [5/7] Top5进站趋势 → 05_top5_in_trend.png")

# ============================================================
# 图6: 最繁忙站点 进站 vs 出站
# ============================================================
fig6, ax6 = plt.subplots(figsize=(14, 6))
s_data = df_day[df_day['stationID'] == top1].set_index('time_slot')
ax6.fill_between(s_data.index, s_data['inNums'], alpha=0.4, color='#4ECDC4', label='进站')
ax6.fill_between(s_data.index, s_data['outNums'], alpha=0.4, color='#FF6B6B', label='出站')
ax6.plot(s_data.index, s_data['inNums'], color='#2BA89E', linewidth=1.5)
ax6.plot(s_data.index, s_data['outNums'], color='#D94F4F', linewidth=1.5)
ax6.set_title(f'B线 最繁忙站点 {top1}  进站 vs 出站 ({PICK_DATE})', fontsize=14, fontweight='bold')
ax6.set_ylabel('人次 (10min)')
ax6.legend(loc='upper left', fontsize=10)
fmt_xaxis(ax6)
ax6.grid(True, alpha=0.3)
fig6.tight_layout()
fig6.savefig(os.path.join(OUTPUT_DIR, '06_top1_in_vs_out.png'), dpi=150)
plt.close(fig6)
print("  [6/7] 最忙站点进出对比 → 06_top1_in_vs_out.png")

# ============================================================
# 图7: 高峰/非高峰 进站 vs 出站 散点
# ============================================================
fig7, ax7 = plt.subplots(figsize=(8, 8))
ax7.scatter(df_day.loc[is_peak, 'inNums'], df_day.loc[is_peak, 'outNums'],
            alpha=0.5, s=10, c='#FF6B6B', label='高峰时段', edgecolors='none')
ax7.scatter(df_day.loc[~is_peak, 'inNums'], df_day.loc[~is_peak, 'outNums'],
            alpha=0.35, s=6, c='#4D96FF', label='非高峰时段', edgecolors='none')
mx = max(df_day['inNums'].max(), df_day['outNums'].max())
ax7.plot([0, mx], [0, mx], 'k--', alpha=0.3, linewidth=0.8)
ax7.set_xlim(0, mx * 1.05)
ax7.set_ylim(0, mx * 1.05)
ax7.set_title(f'B线 进站 vs 出站 散点 ({PICK_DATE})', fontsize=14, fontweight='bold')
ax7.set_xlabel('进站人数 (10min)', fontsize=11)
ax7.set_ylabel('出站人数 (10min)', fontsize=11)
ax7.legend(fontsize=10)
fig7.tight_layout()
fig7.savefig(os.path.join(OUTPUT_DIR, '07_scatter_in_vs_out.png'), dpi=150)
plt.close(fig7)
print("  [7/7] 进出散点 → 07_scatter_in_vs_out.png")

# ============================================================
# 统计摘要
# ============================================================
print("\n" + "=" * 60)
print(f"B线 {PICK_DATE} 统计摘要")
print("=" * 60)
print(f"  总进站:     {df_day['inNums'].sum():>12,}")
print(f"  总出站:     {df_day['outNums'].sum():>12,}")
print(f"  高峰进站:   {df_day.loc[is_peak, 'inNums'].sum():>12,} ({df_day.loc[is_peak, 'inNums'].sum()/df_day['inNums'].sum()*100:.1f}%)")
print(f"  高峰出站:   {df_day.loc[is_peak, 'outNums'].sum():>12,} ({df_day.loc[is_peak, 'outNums'].sum()/df_day['outNums'].sum()*100:.1f}%)")
print(f"\nTop 5 繁忙站点 (按总客流):")
for sid in station_total.tail(5).index[::-1]:
    row = station_total.loc[sid]
    print(f"  站点 {sid:2d}:  进站 {row['inNums']:>8,}  出站 {row['outNums']:>8,}  合计 {row['total']:>8,}")
print(f"\n✅ 7 张图已全部保存至 {OUTPUT_DIR}/")
print("完成！")
