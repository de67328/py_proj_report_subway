# -*- coding: utf-8 -*-
"""
全三线数据预处理
- 读取 25 天原始刷卡数据
- 筛选 A/B/C 三线全部站点
- 按 10 分钟粒度聚合进站/出站人数
- 输出到 AST/data/ 目录
"""

import pandas as pd
import numpy as np
import os
from datetime import timedelta

# ============================================================
# 配置
# ============================================================
RAW_DIR   = r"d:\作业\py展示\Hangzhou-mobility-data-set"
OUT_DIR   = r"d:\作业\py展示\AST\data"
ROAD_MAP  = os.path.join(RAW_DIR, "Metro_roadMap.csv")

START_DATE = "2019-01-01"
END_DATE   = "2019-01-25"

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 1. 确定每条线路的站点
# ============================================================
print("=" * 60)
print("全三线数据预处理")
print("=" * 60)

# 用一天数据获取 stationID → lineID 映射
sample = pd.read_csv(
    os.path.join(RAW_DIR, "record_2019-01-01.csv"),
    dtype={'stationID': 'int16', 'lineID': 'category'},
    usecols=['stationID', 'lineID'],
)

station_line = (
    sample.groupby('stationID')['lineID']
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
    .to_dict()
)
del sample

line_stations = {}
for line in ['A', 'B', 'C']:
    line_stations[line] = sorted([s for s, l in station_line.items() if l == line])
    print(f"  {line}线: {len(line_stations[line])} 站 → {line_stations[line]}")

ALL_STATIONS = sorted(station_line.keys())
N_STATIONS = len(ALL_STATIONS)
print(f"  总计: {N_STATIONS} 站")

# ============================================================
# 2. 逐天聚合
# ============================================================
all_days = []
date_range = pd.date_range(START_DATE, END_DATE, freq='D')

for date in date_range:
    date_str = date.strftime('%Y-%m-%d')
    fpath = os.path.join(RAW_DIR, f"record_{date_str}.csv")
    if not os.path.exists(fpath):
        print(f"  ⚠ 跳过: {date_str}")
        continue

    print(f"  处理 {date_str} ...", end=' ', flush=True)

    df = pd.read_csv(fpath, usecols=['time', 'stationID', 'status'],
                     dtype={'stationID': 'int16', 'status': 'int8'},
                     parse_dates=['time'])

    df = df[df['stationID'].isin(ALL_STATIONS)]
    df['time_slot'] = df['time'].dt.floor('10min')

    agg = (df.groupby(['stationID', 'time_slot', 'status'])
             .size()
             .unstack(fill_value=0))
    agg.columns = ['inNums', 'outNums']
    agg = agg.reset_index()
    all_days.append(agg)

    del df
    print(f"✓ {len(agg)} 条")

# ============================================================
# 3. 合并 & 填充缺失
# ============================================================
print("\n合并 & 填充缺失时段...")
df_all = pd.concat(all_days, ignore_index=True)
del all_days

all_slots = pd.date_range(f"{START_DATE} 00:00:00", f"{END_DATE} 23:50:00", freq='10min')
full_index = pd.MultiIndex.from_product([ALL_STATIONS, all_slots],
                                        names=['stationID', 'time_slot'])
df_full = pd.DataFrame(index=full_index).reset_index()
df_full['inNums'] = 0
df_full['outNums'] = 0

df_full = df_full.set_index(['stationID', 'time_slot'])
df_all_idx = df_all.set_index(['stationID', 'time_slot'])
df_full.update(df_all_idx)
df_all = df_full.reset_index()
del df_full, df_all_idx

# ============================================================
# 4. 添加特征
# ============================================================
print("构造特征...")
df_all['date_str'] = df_all['time_slot'].dt.strftime('%Y-%m-%d')
df_all['hour']     = df_all['time_slot'].dt.hour.astype('int8')
df_all['minute']   = df_all['time_slot'].dt.minute.astype('int8')
df_all['weekday']  = df_all['time_slot'].dt.weekday.astype('int8')
df_all['is_weekend'] = (df_all['weekday'] >= 5).astype('int8')
df_all['is_peak'] = (
    ((df_all['hour'] >= 7) & (df_all['hour'] < 9)) |
    ((df_all['hour'] >= 17) & (df_all['hour'] < 19))
).astype('int8')
df_all['slot_id'] = (df_all['hour'] * 6 + df_all['minute'] // 10).astype('int8')

# 线路标签
df_all['lineID'] = df_all['stationID'].map(station_line).astype('category')

# ============================================================
# 5. 保存
# ============================================================
out_path = os.path.join(OUT_DIR, "all_lines_full.csv")
df_all.to_csv(out_path, index=False, encoding='utf-8')
print(f"\n✅ 已保存: {out_path}")
print(f"   形状: {df_all.shape}")
print(f"   站点: {df_all['stationID'].nunique()}")
print(f"   天数: {df_all['date_str'].nunique()}")
print(f"   总进站: {df_all['inNums'].sum():,}")
print(f"   总出站: {df_all['outNums'].sum():,}")

# 保存站点-线路映射
map_df = pd.DataFrame([
    {'stationID': s, 'lineID': station_line[s]}
    for s in ALL_STATIONS
])
map_df.to_csv(os.path.join(OUT_DIR, "station_line_map.csv"), index=False)

# 保存路网子矩阵
adj_full = pd.read_csv(ROAD_MAP, index_col=0)
adj_full.index = adj_full.index.astype(int)
adj_full.columns = adj_full.columns.astype(int)
adj_all = adj_full.loc[ALL_STATIONS, ALL_STATIONS]
adj_all.to_csv(os.path.join(OUT_DIR, "adj_matrix.csv"))
print(f"  路网矩阵: {adj_all.shape}")
print(f"  边数: {adj_all.values.sum() // 2}")
