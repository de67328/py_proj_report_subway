# -*- coding: utf-8 -*-
"""
B线数据预处理脚本
- 读取 2019-01-01 ~ 2019-01-25 共25天刷卡数据
- 筛选 B 线站点 (stationID 0-33)
- 按 10分钟粒度聚合进站/出站人数
- 添加时间特征
- 拆分训练集 (day 1-22) 和验证集 (day 23-25)
- 输出到 data/ 目录
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# ============================================================
# 配置
# ============================================================
DATA_DIR = r"d:\作业\py展示\Hangzhou-mobility-data-set"
OUTPUT_DIR = r"d:\作业\py展示\data"
B_LINE_STATIONS = list(range(0, 34))  # B线站点 0-33
START_DATE = "2019-01-01"
END_DATE = "2019-01-25"
TRAIN_END_DATE = "2019-01-22"  # 训练集截止日期

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. 逐天读取并聚合
# ============================================================
print("=" * 60)
print("B线数据预处理：逐天读取 & 10分钟聚合")
print("=" * 60)

all_days = []  # 存放每天聚合后的 DataFrame
date_range = pd.date_range(START_DATE, END_DATE, freq='D')

for date in date_range:
    date_str = date.strftime('%Y-%m-%d')
    file_path = os.path.join(DATA_DIR, f"record_{date_str}.csv")

    if not os.path.exists(file_path):
        print(f"  ⚠ 文件不存在，跳过: {file_path}")
        continue

    print(f"  处理 {date_str} ...", end=' ', flush=True)

    # 只读取必要列
    df = pd.read_csv(
        file_path,
        dtype={
            'lineID': 'category',
            'stationID': 'int16',
            'status': 'int8',
        },
        usecols=['time', 'lineID', 'stationID', 'status'],
        parse_dates=['time'],
    )

    # 筛选 B 线
    df = df[df['stationID'].isin(B_LINE_STATIONS)]

    # 生成 10 分钟时间段标签
    df['time_slot'] = df['time'].dt.floor('10min')

    # 按 (stationID, time_slot, status) 聚合计数
    agg = df.groupby(['stationID', 'time_slot', 'status']).size().unstack(fill_value=0)
    agg.columns = ['inNums', 'outNums']  # status: 0=进站, 1=出站
    agg = agg.reset_index()

    all_days.append(agg)
    del df
    print(f"✓ {len(agg)} 条聚合记录")

# ============================================================
# 2. 合并所有天
# ============================================================
print("\n合并所有天数据...")
df_all = pd.concat(all_days, ignore_index=True)
del all_days

print(f"合并后总记录数: {len(df_all):,}")

# ============================================================
# 3. 填充缺失的 (stationID, time_slot) 组合
# ============================================================
print("\n填充缺失时段（部分时段无刷卡数据）...")

# 生成完整的时间-站点笛卡尔积
all_slots = pd.date_range(
    f"{START_DATE} 00:00:00",
    f"{END_DATE} 23:50:00",
    freq='10min'
)

# 创建完整索引
full_index = pd.MultiIndex.from_product(
    [B_LINE_STATIONS, all_slots],
    names=['stationID', 'time_slot']
)
df_full = pd.DataFrame(index=full_index).reset_index()
df_full['inNums'] = 0
df_full['outNums'] = 0

# 合并已有数据 (保留已有，其余为0)
df_full = df_full.set_index(['stationID', 'time_slot'])
df_all_idx = df_all.set_index(['stationID', 'time_slot'])

# 只更新有数据的行
df_full.update(df_all_idx)
df_all = df_full.reset_index()

del df_full, df_all_idx

print(f"填充后总记录数: {len(df_all):,} (34站 × {len(all_slots)}时段)")

# ============================================================
# 4. 添加时间特征
# ============================================================
print("\n构造时间特征...")

df_all['date'] = df_all['time_slot'].dt.date
df_all['hour'] = df_all['time_slot'].dt.hour.astype('int8')
df_all['minute'] = df_all['time_slot'].dt.minute.astype('int8')
df_all['weekday'] = df_all['time_slot'].dt.weekday.astype('int8')  # 0=周一
df_all['is_weekend'] = (df_all['weekday'] >= 5).astype('int8')

# 早晚高峰标记: 7:00-9:00, 17:00-19:00
df_all['is_peak'] = (
    ((df_all['hour'] >= 7) & (df_all['hour'] < 9)) |
    ((df_all['hour'] >= 17) & (df_all['hour'] < 19))
).astype('int8')

# 时段编号 (0-143, 每天144个10分钟时段)
df_all['slot_id'] = (df_all['hour'] * 6 + df_all['minute'] // 10).astype('int8')

print(f"特征列: {list(df_all.columns)}")

# ============================================================
# 5. 拆分训练集 & 验证集
# ============================================================
print("\n拆分训练集和验证集...")

# 统一用字符串比较避免 date vs Timestamp 类型冲突
df_all['date_str'] = df_all['time_slot'].dt.strftime('%Y-%m-%d')
train_mask = df_all['date_str'] <= TRAIN_END_DATE
df_train = df_all[train_mask].copy()
df_valid = df_all[~train_mask].copy()
train_end_dt = pd.Timestamp(TRAIN_END_DATE)

print(f"  训练集: {len(df_train):,} 条 ({START_DATE} ~ {TRAIN_END_DATE})")
print(f"  验证集: {len(df_valid):,} 条 ({train_end_dt + timedelta(days=1):%Y-%m-%d} ~ {END_DATE})")

# ============================================================
# 6. 保存
# ============================================================
print("\n保存数据...")

df_train.to_csv(
    os.path.join(OUTPUT_DIR, 'b_line_train.csv'),
    index=False, encoding='utf-8'
)
df_valid.to_csv(
    os.path.join(OUTPUT_DIR, 'b_line_valid.csv'),
    index=False, encoding='utf-8'
)

# 同时保存完整数据集（方便后续做滞后特征）
df_all.to_csv(
    os.path.join(OUTPUT_DIR, 'b_line_full.csv'),
    index=False, encoding='utf-8'
)

print(f"\n✅ 数据已保存至 {OUTPUT_DIR}/")
print(f"   b_line_train.csv — 训练集")
print(f"   b_line_valid.csv — 验证集")
print(f"   b_line_full.csv  — 完整25天数据")

# ============================================================
# 7. 数据摘要
# ============================================================
print("\n" + "=" * 60)
print("数据摘要")
print("=" * 60)
print(f"\n训练集统计:")
print(f"  站点数: {df_train['stationID'].nunique()}")
print(f"  天数:   {df_train['date'].nunique()}")
print(f"  总进站: {df_train['inNums'].sum():,}")
print(f"  总出站: {df_train['outNums'].sum():,}")

print(f"\n验证集统计:")
print(f"  站点数: {df_valid['stationID'].nunique()}")
print(f"  天数:   {df_valid['date'].nunique()}")
print(f"  总进站: {df_valid['inNums'].sum():,}")
print(f"  总出站: {df_valid['outNums'].sum():,}")

print(f"\n前5行样例:")
print(df_train.head().to_string())

print("\n完成！")
