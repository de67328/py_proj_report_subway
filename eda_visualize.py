# -*- coding: utf-8 -*-
"""
地铁人流量预测 - 初步可视化 EDA
对 record_2019-01-01.csv 进行探索性数据分析
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 使用 TkAgg 后端，支持交互式窗口
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import Counter

# ============================================================
# 设置中文字体
# ============================================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 数据加载
# ============================================================
print("=" * 60)
print("正在加载数据...")
DATA_PATH = r"d:\作业\py展示\record_2019-01-01.csv"

# 只读取必要列以节省内存
df = pd.read_csv(
    DATA_PATH,
    dtype={
        'lineID': 'category',
        'stationID': 'int16',
        'deviceID': 'int16',
        'status': 'int8',
        'userID': 'str',
        'payType': 'int8',
    },
    parse_dates=['time'],
)
print(f"数据加载完成！共 {len(df):,} 条记录")
print(f"内存占用: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# ============================================================
# 2. 数据概览
# ============================================================
print("\n" + "=" * 60)
print("数据基本信息：")
print(df.info(memory_usage='deep'))
print("\n前5行数据：")
print(df.head())
print("\n缺失值统计：")
print(df.isnull().sum())
print("\n数据描述：")
print(df.describe())

# ============================================================
# 3. 各字段分布
# ============================================================
print("\nlineID 分布：")
print(df['lineID'].value_counts())
print("\nstatus 分布 (0=进站, 1=出站)：")
print(df['status'].value_counts())
print("\npayType 分布：")
print(df['payType'].value_counts())
print(f"\n唯一 stationID 数量: {df['stationID'].nunique()}")
print(f"唯一 userID 数量: {df['userID'].nunique()}")

# ============================================================
# 4. 可视化 - 创建子图布局
# ============================================================
fig = plt.figure(figsize=(20, 18))
fig.suptitle('地铁刷卡数据 EDA 可视化 (2019-01-01)', fontsize=18, fontweight='bold', y=0.98)

# --- 4.1 进站/出站饼图 ---
ax1 = fig.add_subplot(3, 3, 1)
status_counts = df['status'].value_counts()
labels = ['进站 (0)', '出站 (1)']
colors = ['#4ECDC4', '#FF6B6B']
ax1.pie(status_counts, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90, explode=(0.02, 0.02))
ax1.set_title('进站 vs 出站 比例', fontsize=13, fontweight='bold')

# --- 4.2 各线路刷卡量 ---
ax2 = fig.add_subplot(3, 3, 2)
line_counts = df['lineID'].value_counts().sort_index()
bars = ax2.bar(line_counts.index.astype(str), line_counts.values,
               color=['#FFD93D', '#6BCB77', '#4D96FF'])
ax2.set_title('各线路刷卡量', fontsize=13, fontweight='bold')
ax2.set_xlabel('线路')
ax2.set_ylabel('刷卡次数')
for bar, val in zip(bars, line_counts.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
             f'{val:,}', ha='center', fontsize=9)

# --- 4.3 每小时刷卡量趋势（分进站/出站） ---
ax3 = fig.add_subplot(3, 3, (3, 5))  # 跨两列
df['hour'] = df['time'].dt.hour
hourly_status = df.groupby(['hour', 'status']).size().unstack(fill_value=0)
hourly_status.columns = ['进站', '出站']
hourly_status.plot(ax=ax3, marker='o', linewidth=2, markersize=5)
ax3.set_title('各时段进站/出站人次变化', fontsize=13, fontweight='bold')
ax3.set_xlabel('小时')
ax3.set_ylabel('刷卡次数')
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.3)
ax3.set_xticks(range(0, 24))

# --- 4.4 payType 分布 ---
ax4 = fig.add_subplot(3, 3, 6)
pay_counts = df['payType'].value_counts().sort_index()
ax4.bar(pay_counts.index.astype(str), pay_counts.values, color='#C39BD3')
ax4.set_title('支付类型(payType)分布', fontsize=13, fontweight='bold')
ax4.set_xlabel('payType')
ax4.set_ylabel('刷卡次数')
for i, v in enumerate(pay_counts.values):
    ax4.text(i, v + 5000, f'{v:,}', ha='center', fontsize=9)

# --- 4.5 Top 15 站点刷卡量 ---
ax5 = fig.add_subplot(3, 3, 7)
top_stations = df['stationID'].value_counts().head(15)
colors_top = plt.cm.viridis(np.linspace(0.2, 0.9, 15))
bars = ax5.barh(range(15), top_stations.values, color=colors_top)
ax5.set_yticks(range(15))
ax5.set_yticklabels([f'站点 {s}' for s in top_stations.index])
ax5.invert_yaxis()
ax5.set_title('Top 15 繁忙站点', fontsize=13, fontweight='bold')
ax5.set_xlabel('刷卡次数')
for bar, val in zip(bars, top_stations.values):
    ax5.text(bar.get_width() + 1000, bar.get_y() + bar.get_height()/2,
             f'{val:,}', va='center', fontsize=8)

# --- 4.6 10分钟粒度聚合示例 ---
ax6 = fig.add_subplot(3, 3, 8)
# 选取客流量最大的站点
top_station = df['stationID'].value_counts().idxmax()
station_df = df[df['stationID'] == top_station].copy()
station_df['time_10min'] = station_df['time'].dt.floor('10min')
time_series = station_df.groupby(['time_10min', 'status']).size().unstack(fill_value=0)
if time_series.shape[1] >= 2:
    time_series.columns = ['进站', '出站']
elif time_series.shape[1] == 1:
    time_series.columns = ['进站'] if 0 in time_series.columns else ['出站']
ax6.plot(time_series.index, time_series.iloc[:, 0] if time_series.shape[1]>=1 else [],
         label=time_series.columns[0] if time_series.shape[1]>=1 else '', alpha=0.7, linewidth=1)
if time_series.shape[1] >= 2:
    ax6.plot(time_series.index, time_series.iloc[:, 1], label=time_series.columns[1], alpha=0.7, linewidth=1)
ax6.set_title(f'站点 {top_station} 10分钟粒度人流量', fontsize=13, fontweight='bold')
ax6.set_xlabel('时间')
ax6.set_ylabel('人数')
ax6.legend(fontsize=8)
ax6.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax6.xaxis.set_major_locator(mdates.HourLocator(interval=3))
plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, fontsize=7)
ax6.grid(True, alpha=0.3)

# --- 4.7 进站/出站小时热力图 ---
ax7 = fig.add_subplot(3, 3, 9)
hour_station_in = df[df['status'] == 0].groupby(['stationID', 'hour']).size().unstack(fill_value=0)
# 取前20个站点
top20 = df['stationID'].value_counts().head(20).index
hour_station_in_top = hour_station_in.loc[hour_station_in.index.isin(top20)]
im = ax7.imshow(hour_station_in_top.values, aspect='auto', cmap='YlOrRd')
ax7.set_title('Top20站点 × 小时 进站热力图', fontsize=13, fontweight='bold')
ax7.set_xlabel('小时')
ax7.set_ylabel('站点ID')
ax7.set_yticks(range(len(hour_station_in_top)))
ax7.set_yticklabels(hour_station_in_top.index)
plt.colorbar(im, ax=ax7, shrink=0.8)

# ============================================================
# 5. 关键统计输出
# ============================================================
print("\n" + "=" * 60)
print("关键统计汇总：")
print(f"  总记录数:        {len(df):>12,}")
print(f"  进站次数:        {status_counts.get(0,0):>12,}")
print(f"  出站次数:        {status_counts.get(1,0):>12,}")
print(f"  唯一站点数:      {df['stationID'].nunique():>12}")
print(f"  唯一用户数:      {df['userID'].nunique():>12}")
print(f"  线路数:          {df['lineID'].nunique():>12}")
print(f"  时间范围:        {df['time'].min()} ~ {df['time'].max()}")
print(f"  平均每小时进站:  {status_counts.get(0,0)/24:>12,.0f}")
print(f"  平均每小时出站:  {status_counts.get(1,0)/24:>12,.0f}")

# ============================================================
# 6. 保存图片 + 显示
# ============================================================
plt.tight_layout(rect=[0, 0, 1, 0.96])
output_path = r"d:\作业\py展示\eda_visualize_output.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n图表已保存至: {output_path}")
plt.show()
print("\n可视化完成！")
