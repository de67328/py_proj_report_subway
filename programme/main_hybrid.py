# -*- coding: utf-8 -*-
"""
main_hybrid.py — GCN 嵌入 + 树模型 混合方案
- 先运行 gcn_embeddings.py 提取嵌入（如果还没有）
- 用 build_features_hybrid 构建含 GCN 嵌入的特征
- 仅训练树模型（XGBoost / LightGBM / GBDT）
- 模型保存至 models/hybrid/
- 与 no_ast 版本对比
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from features import build_features, build_features_hybrid, _load_data
from Xgb_methods import run_tree_models
from results_viz import visualize_results

MODEL_DIR_HYBRID = r"d:\作业\py展示\programme\models\hybrid\tree"
EMBED_PATH = r"d:\作业\py展示\data\gcn_embeddings.csv"


def main():
    print("=" * 70)
    print("   B线地铁人流量预测 — GCN嵌入 + 树模型 (Hybrid)")
    print("   Closed-loop: 训练 01-01~01-22 | 验证 01-23~01-25")
    print("=" * 70)

    # ---- Step 0: 确保 GCN 嵌入存在 ----
    if not os.path.exists(EMBED_PATH):
        print("\n[Step 0] GCN 嵌入不存在，正在提取...")
        from gcn_embeddings import extract_and_save_embeddings
        extract_and_save_embeddings()
    else:
        print(f"\n[Step 0] GCN 嵌入已存在: {EMBED_PATH}")

    # ---- Step 1: 特征工程（含 GCN 嵌入） ----
    print("\n[Step 1] 特征工程 (含 GCN 嵌入) ...")
    X_train, y_train, X_valid, y_valid, feature_cols = build_features_hybrid()

    # ---- Step 2: 树模型训练 ----
    print("\n[Step 2] 树模型训练 (含 GCN 嵌入) ...")
    hybrid_results, _, _, best_preds = run_tree_models(
        X_train, y_train, X_valid, y_valid, model_dir=MODEL_DIR_HYBRID
    )

    # ---- Step 3: 也跑一版无 GCN 的树模型作对照 ----
    print("\n[Step 3] 对照: 树模型 (无 GCN) ...")
    X_train_base, y_train_b, X_valid_base, y_valid_b, _ = build_features()
    base_results, _, _, base_best = run_tree_models(
        X_train_base, y_train_b, X_valid_base, y_valid_b,
        model_dir=r"d:\作业\py展示\programme\models\no_ast\tree"
    )

    # ---- Step 4: 对比 ----
    print("\n" + "=" * 70)
    print("   Hybrid (GCN+Tree) vs Baseline (Tree only) 对比")
    print("=" * 70)
    print(f"{'模型':<25} {'MAE_in':>8} {'MAE_out':>8} {'MAE_avg':>8} {'类型':>12}")
    print("-" * 68)

    for r in sorted(hybrid_results, key=lambda x: x['mae_avg']):
        print(f"{'[Hybrid] '+r['name']:<25} {r['mae_in']:>8.2f} {r['mae_out']:>8.2f} "
              f"{r['mae_avg']:>8.2f} {'GCN+Tree':>12}")

    for r in sorted(base_results, key=lambda x: x['mae_avg']):
        print(f"{'[Baseline] '+r['name']:<25} {r['mae_in']:>8.2f} {r['mae_out']:>8.2f} "
              f"{r['mae_avg']:>8.2f} {'Tree only':>12}")

    # 计算提升
    print("-" * 68)
    for hr in hybrid_results:
        for br in base_results:
            if hr['name'] == br['name']:
                delta = br['mae_avg'] - hr['mae_avg']
                pct = delta / br['mae_avg'] * 100
                sign = '↓' if delta > 0 else '↑'
                print(f"  {hr['name']}: {br['mae_avg']:.2f} → {hr['mae_avg']:.2f} "
                      f"({sign}{abs(pct):.1f}%)")

    # ---- Step 5: 可视化 ----
    if best_preds:
        print("\n[Step 5] 最佳 Hybrid 模型结果可视化 ...")
        df_full = _load_data()
        df_full = df_full.sort_values(['stationID', 'time_slot'])
        valid_mask = df_full['date_str'] > "2019-01-22"
        valid_slots = df_full.loc[valid_mask, 'time_slot'].values
        valid_stations = df_full.loc[valid_mask, 'stationID'].values
        del df_full
        visualize_results(y_valid, best_preds['pred_in'], best_preds['pred_out'],
                          valid_slots, valid_stations)

    print("\n✅ main_hybrid 完成！")


if __name__ == '__main__':
    main()
