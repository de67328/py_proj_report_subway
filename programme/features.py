# -*- coding: utf-8 -*-
"""
特征工程模块 — Closed-loop
- 读取 B 线完整聚合数据
- 构造时间特征、滞后特征、滚动特征、站点编码
- 拆分训练集 (day 1~22) 和验证集 (day 23~25)
- 返回 X_train, y_train, X_valid, y_valid
"""

import pandas as pd
import numpy as np
import os

DATA_PATH = r"d:\作业\py展示\data\b_line_full.csv"
TRAIN_END = "2019-01-22"


def build_features(df=None) -> tuple:
    """
    主入口：构建所有特征，返回 (X_train, y_train, X_valid, y_valid, feature_names)
    y_train/y_valid 包含两列: inNums, outNums
    """
    if df is None:
        df = _load_data()

    df = df.sort_values(['stationID', 'time_slot']).copy()

    # ---- 1. 基础时间特征（已在预处理中做好，直接可用） ----
    # hour, minute, weekday, is_weekend, is_peak, slot_id

    # ---- 2. 滞后特征 (同站点、同时段、前N天) ----
    df = _add_lag_features(df, days_list=[1, 2, 3, 7])

    # ---- 3. 滚动特征 (同站点、前N个时段) ----
    df = _add_rolling_features(df, slots_list=[1, 2, 3])

    # ---- 4. 站点 One-Hot 编码 ----
    df = _add_station_dummies(df)

    # ---- 5. 交互特征 (高峰 × 站点大类) ----
    # 站点按平均客流量分档
    station_avg = df.groupby('stationID')['inNums'].transform('mean')
    df['station_crowd_level'] = pd.cut(station_avg, bins=3, labels=[0, 1, 2]).astype('int8')
    df['peak_x_crowd'] = df['is_peak'] * df['station_crowd_level']

    # ---- 6. 拆分 X / y ----
    feature_cols = [c for c in df.columns if c not in
                    ('stationID', 'time_slot', 'date', 'date_str',
                     'inNums', 'outNums', 'station_crowd_level')]

    train_mask = df['date_str'] <= TRAIN_END

    X_train = df.loc[train_mask, feature_cols].copy()
    X_valid = df.loc[~train_mask, feature_cols].copy()
    y_train = df.loc[train_mask, ['inNums', 'outNums']].copy()
    y_valid = df.loc[~train_mask, ['inNums', 'outNums']].copy()

    # 处理缺失值（前3天没有 lag_3d / lag_7d 等）
    X_train = X_train.fillna(0)
    X_valid = X_valid.fillna(0)

    print(f"特征工程完成: X_train={X_train.shape}, X_valid={X_valid.shape}")
    print(f"特征列 ({len(feature_cols)}): {feature_cols}")

    return X_train, y_train, X_valid, y_valid, feature_cols


# ============================================================
# 内部函数
# ============================================================

def _load_data():
    print(f"加载数据: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=['time_slot'])
    return df


def _add_lag_features(df, days_list):
    """同站点、同时段、前N天的 inNums/outNums"""
    df = df.copy()
    # 每天144个时段
    SLOTS_PER_DAY = 144

    for d in days_list:
        shift_steps = d * SLOTS_PER_DAY
        for col in ['inNums', 'outNums']:
            df[f'{col}_lag_{d}d'] = (
                df.groupby('stationID')[col]
                .shift(shift_steps)
            )
    return df


def _add_rolling_features(df, slots_list):
    """同站点、前N个时段的滚动均值"""
    df = df.copy()
    for n in slots_list:
        for col in ['inNums', 'outNums']:
            df[f'{col}_roll_{n}'] = (
                df.groupby('stationID')[col]
                .transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
            )
    return df


def _add_station_dummies(df):
    """站点 One-Hot 编码"""
    dummies = pd.get_dummies(df['stationID'], prefix='st', dtype='int8')
    df = pd.concat([df, dummies], axis=1)
    return df


# ============================================================
# 独立运行测试
# ============================================================
if __name__ == '__main__':
    X_train, y_train, X_valid, y_valid, feature_cols = build_features()
    print(f"\n训练集: X={X_train.shape}, y={y_train.shape}")
    print(f"验证集: X={X_valid.shape}, y={y_valid.shape}")
    print(f"\nX_train 前3行:")
    print(X_train.head(3).to_string())
    print(f"\ny_train 统计:")
    print(y_train.describe())
