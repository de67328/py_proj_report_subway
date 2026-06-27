# -*- coding: utf-8 -*-
"""
主流程 — B线地铁人流量预测模型对比
- 统一特征工程 → 线性模型 + 树模型 → MAE 对比
"""

import sys
import os

# 确保 programme 目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from features import build_features, _load_data
from Linear_methods import run_linear_models
from Xgb_methods import run_tree_models
from results_viz import visualize_results, visualize_method_comparison


def main():
    print("=" * 70)
    print("   B线地铁人流量预测 — 模型对比实验")
    print("   Closed-loop: 训练 01-01~01-22 | 验证 01-23~01-25")
    print("=" * 70)

    # ============================================================
    # Step 1: 特征工程
    # ============================================================
    print("\n[Step 1] 特征工程 ...")
    X_train, y_train, X_valid, y_valid, feature_names = build_features()

    # ============================================================
    # Step 2: 线性模型
    # ============================================================
    print("\n[Step 2] 线性模型训练与评估 ...")
    linear_results, linear_models, linear_preds = run_linear_models(
        X_train, y_train, X_valid, y_valid
    )

    # ============================================================
    # Step 3: 树模型
    # ============================================================
    print("\n[Step 3] 树模型训练与评估 ...")
    tree_results, tree_models, tree_preds, best_preds = run_tree_models(
        X_train, y_train, X_valid, y_valid
    )

    # ============================================================
    # Step 4: 汇总对比
    # ============================================================
    all_results = linear_results + tree_results

    print("\n" + "=" * 70)
    print("   模型对比结果 (Closed-loop, B线)")
    print("=" * 70)
    print(f"{'模型':<22} {'MAE_in':>8} {'MAE_out':>8} {'MAE_avg':>8} {'耗时(s)':>8}")
    print("-" * 58)

    # 按 MAE_avg 排序
    all_results.sort(key=lambda x: x['mae_avg'])

    for r in all_results:
        print(f"{r['name']:<22} {r['mae_in']:>8.2f} {r['mae_out']:>8.2f} "
              f"{r['mae_avg']:>8.2f} {r['time']:>8.2f}")

    # ---- 标杆：用前一天同时段作为预测（naive baseline） ----
    print("\n[Baseline] 前一天同时段 ...", end=' ', flush=True)
    # 从特征中取 lag_1d 作为预测
    if 'inNums_lag_1d' in X_valid.columns:
        baseline_in = X_valid['inNums_lag_1d'].fillna(0).values
        baseline_out = X_valid['outNums_lag_1d'].fillna(0).values
        mae_b_in = np.mean(np.abs(y_valid['inNums'].values - baseline_in))
        mae_b_out = np.mean(np.abs(y_valid['outNums'].values - baseline_out))
        print(f"MAE_in={mae_b_in:.2f}  MAE_out={mae_b_out:.2f}  "
              f"MAE_avg={(mae_b_in+mae_b_out)/2:.2f}")
        print(f"{'Naive(前一天)':<22} {mae_b_in:>8.2f} {mae_b_out:>8.2f} "
              f"{(mae_b_in+mae_b_out)/2:>8.2f} {'-':>8}")

    print("-" * 58)
    print("\n✅ 模型对比完成！")

    # ============================================================
    # Step 5: 最佳模型结果可视化
    # ============================================================
    if best_preds:
        print("\n[Step 5] 最佳模型 (XGBoost) 结果可视化 ...")
        # 从完整数据中取验证集的 time_slot 和 stationID
        df_full = _load_data()
        df_full = df_full.sort_values(['stationID', 'time_slot'])
        valid_mask = df_full['date_str'] > "2019-01-22"
        valid_slots = df_full.loc[valid_mask, 'time_slot'].values
        valid_stations = df_full.loc[valid_mask, 'stationID'].values
        del df_full

        visualize_results(
            y_valid,
            best_preds['pred_in'],
            best_preds['pred_out'],
            valid_slots,
            valid_stations
        )

    # ============================================================
    # Step 6: 方法间对比折线图
    # ============================================================
    print("\n[Step 6] 方法间对比折线图 ...")
    df_full2 = _load_data()
    df_full2 = df_full2.sort_values(['stationID', 'time_slot'])
    valid_mask2 = df_full2['date_str'] > "2019-01-22"
    valid_slots2 = df_full2.loc[valid_mask2, 'time_slot'].values
    valid_stations2 = df_full2.loc[valid_mask2, 'stationID'].values
    del df_full2

    visualize_method_comparison(
        y_valid, linear_preds, tree_preds,
        valid_slots2, valid_stations2
    )

    print("\n✅ 全部完成！")


if __name__ == '__main__':
    main()
