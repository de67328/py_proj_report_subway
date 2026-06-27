# -*- coding: utf-8 -*-
"""
main_tree_ast.py — 考虑 ASTGCN 的版本
- 仅训练树模型（XGBoost / LightGBM / GBDT）
- 同时训练 ASTGCN 图神经网络
- 模型保存至 models/tree_ast/
- 对比树模型与 ASTGCN
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import joblib
import glob
import os
from features import build_features, _load_data
from Xgb_methods import run_tree_models
from results_viz import visualize_results
from sklearn.metrics import mean_absolute_error

MODEL_DIR_TREE = r"d:\作业\py展示\programme\models\tree_ast\tree"

# ASTGCN 可选
_ASTGCN_AVAIL = True
try:
    from astgcn_train import train_astgcn
    from graph_utils import build_graph_data
except ImportError as e:
    _ASTGCN_AVAIL = False
    print(f"  ⚠ ASTGCN 不可用: {e}")


def _check_models_exist(model_dir, expected_count=1):
    if not os.path.isdir(model_dir):
        return False
    return len(glob.glob(os.path.join(model_dir, '*.pkl'))) >= expected_count


def _load_and_predict_tree(model_dir, X_train, y_train, X_valid, y_valid):
    """从磁盘加载树模型并预测"""
    print(f"  📂 加载已有树模型: {model_dir}")
    results, models_dict, preds_dict, best_preds = [], {}, {}, {}
    in_files = sorted(glob.glob(os.path.join(model_dir, '*_in.pkl')))
    out_files = sorted(glob.glob(os.path.join(model_dir, '*_out.pkl')))
    for in_f, out_f in zip(in_files, out_files):
        name = os.path.basename(in_f).replace('_in.pkl', '')
        model_in, model_out = joblib.load(in_f), joblib.load(out_f)
        pred_in = np.maximum(model_in.predict(X_valid), 0)
        pred_out = np.maximum(model_out.predict(X_valid), 0)
        mae_in = mean_absolute_error(y_valid['inNums'], pred_in)
        mae_out = mean_absolute_error(y_valid['outNums'], pred_out)
        results.append({'name': name, 'mae_in': mae_in, 'mae_out': mae_out,
                        'mae_avg': (mae_in+mae_out)/2, 'time': 0})
        models_dict[name] = (model_in, model_out)
        preds_dict[name] = (pred_in, pred_out)
        if not best_preds:
            best_preds = {'pred_in': pred_in, 'pred_out': pred_out}
        print(f"    {name}: MAE_in={mae_in:.2f} MAE_out={mae_out:.2f}")
    return results, models_dict, preds_dict, best_preds


def main():
    print("=" * 70)
    print("   B线地铁人流量预测 — 树模型 + ASTGCN 对比")
    print("   Closed-loop: 训练 01-01~01-22 | 验证 01-23~01-25")
    print("=" * 70)

    # ---- Step 1: 特征工程 (供树模型使用) ----
    print("\n[Step 1] 特征工程 (树模型) ...")
    X_train, y_train, X_valid, y_valid, feature_names = build_features()

    # ---- Step 2: 树模型 ----
    if _check_models_exist(MODEL_DIR_TREE, expected_count=3):
        print("\n[Step 2] 加载已有树模型 ...")
        tree_results, tree_models, tree_preds, best_preds = _load_and_predict_tree(
            MODEL_DIR_TREE, X_train, y_train, X_valid, y_valid
        )
    else:
        print("\n[Step 2] 树模型训练 ...")
        tree_results, tree_models, tree_preds, best_preds = run_tree_models(
            X_train, y_train, X_valid, y_valid, model_dir=MODEL_DIR_TREE
        )

    # ---- Step 3: ASTGCN ----
    astgcn_result = None
    if _ASTGCN_AVAIL:
        print("\n[Step 3] ASTGCN 图神经网络训练 ...")
        astgcn_result, _ = train_astgcn()
    else:
        print("\n[Step 3] ASTGCN 跳过（环境不可用）")

    # ---- Step 4: 汇总对比 ----
    all_results = tree_results[:]  # 只含树模型
    if astgcn_result:
        all_results.append(astgcn_result)

    print("\n" + "=" * 70)
    print("   模型对比结果 (树模型 + ASTGCN)")
    print("=" * 70)
    print(f"{'模型':<22} {'MAE_in':>8} {'MAE_out':>8} {'MAE_avg':>8} {'耗时(s)':>8}")
    print("-" * 58)

    all_results.sort(key=lambda x: x['mae_avg'])
    for r in all_results:
        extra = ''
        if 'params' in r:
            extra = f"  ({r['params']:,} params)"
        print(f"{r['name']:<22} {r['mae_in']:>8.2f} {r['mae_out']:>8.2f} "
              f"{r['mae_avg']:>8.2f} {r['time']:>8.1f}{extra}")

    # ---- Naive baseline ----
    print("\n[Baseline] 前一天同时段 ...", end=' ', flush=True)
    if 'inNums_lag_1d' in X_valid.columns:
        baseline_in = X_valid['inNums_lag_1d'].fillna(0).values
        baseline_out = X_valid['outNums_lag_1d'].fillna(0).values
        mae_b_in = np.mean(np.abs(y_valid['inNums'].values - baseline_in))
        mae_b_out = np.mean(np.abs(y_valid['outNums'].values - baseline_out))
        print(f"MAE_in={mae_b_in:.2f}  MAE_out={mae_b_out:.2f}  MAE_avg={(mae_b_in+mae_b_out)/2:.2f}")
        print(f"{'Naive(前一天)':<22} {mae_b_in:>8.2f} {mae_b_out:>8.2f} {(mae_b_in+mae_b_out)/2:>8.2f} {'-':>8}")
    print("-" * 58)

    # ---- Step 5: XGBoost 结果可视化 ----
    if best_preds:
        print("\n[Step 5] 最佳模型 (XGBoost) 结果可视化 ...")
        df_full = _load_data()
        df_full = df_full.sort_values(['stationID', 'time_slot'])
        valid_mask = df_full['date_str'] > "2019-01-22"
        valid_slots = df_full.loc[valid_mask, 'time_slot'].values
        valid_stations = df_full.loc[valid_mask, 'stationID'].values
        del df_full
        visualize_results(y_valid, best_preds['pred_in'], best_preds['pred_out'],
                          valid_slots, valid_stations)

    print("\n✅ main_tree_ast 完成！")


if __name__ == '__main__':
    main()
