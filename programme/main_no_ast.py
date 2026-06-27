# -*- coding: utf-8 -*-
"""
main_no_ast.py — 不考虑 ASTGCN 的版本
- 线性模型 + 树模型
- 模型保存至 models/no_ast/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import joblib
import glob
from features import build_features, _load_data
from Linear_methods import run_linear_models
from Xgb_methods import run_tree_models
from results_viz import visualize_results, visualize_method_comparison
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

MODEL_DIR_LINEAR = r"d:\作业\py展示\programme\models\no_ast\linear"
MODEL_DIR_TREE   = r"d:\作业\py展示\programme\models\no_ast\tree"


def _check_models_exist(model_dir, expected_count=1):
    """检查目录下是否有 .pkl 模型文件"""
    if not os.path.isdir(model_dir):
        return False
    pkl_files = glob.glob(os.path.join(model_dir, '*.pkl'))
    return len(pkl_files) >= expected_count


def _load_and_predict_linear(model_dir, X_train, y_train, X_valid, y_valid):
    """从磁盘加载线性模型并预测，返回 (results, models_dict, preds_dict)"""
    print(f"  📂 加载已有线性模型: {model_dir}")
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    X_train_s = scaler.transform(X_train)
    X_valid_s = scaler.transform(X_valid)

    results = []
    models_dict = {}
    preds_dict = {}

    model_files = sorted(glob.glob(os.path.join(model_dir, '*.pkl')))
    for f in model_files:
        fname = os.path.basename(f)
        if fname == 'scaler.pkl':
            continue
        name = fname.replace('.pkl', '')
        obj = joblib.load(f)

        t0 = __import__('time').time()
        if isinstance(obj, dict) and 'pca' in obj:  # PCA+Ridge
            X_v = obj['pca'].transform(X_valid_s)
            pred = np.maximum(obj['ridge'].predict(X_v), 0)
        else:
            pred = np.maximum(obj.predict(X_valid_s), 0)
        t = __import__('time').time() - t0

        mae_in = mean_absolute_error(y_valid['inNums'], pred[:, 0])
        mae_out = mean_absolute_error(y_valid['outNums'], pred[:, 1])
        results.append({'name': name, 'mae_in': mae_in, 'mae_out': mae_out,
                        'mae_avg': (mae_in+mae_out)/2, 'time': t})
        models_dict[name] = obj
        preds_dict[name] = (pred[:, 0], pred[:, 1])
        print(f"    {name}: MAE_in={mae_in:.2f} MAE_out={mae_out:.2f}")

    return results, models_dict, preds_dict


def _load_and_predict_tree(model_dir, X_train, y_train, X_valid, y_valid):
    """从磁盘加载树模型并预测，返回 (results, models_dict, preds_dict, best_preds)"""
    print(f"  📂 加载已有树模型: {model_dir}")
    results = []
    models_dict = {}
    preds_dict = {}
    best_preds = {}

    # 按名称分组 (XGBoost_in + XGBoost_out = 一个模型)
    model_files = sorted(glob.glob(os.path.join(model_dir, '*.pkl')))
    in_files = [f for f in model_files if f.endswith('_in.pkl')]
    out_files = [f for f in model_files if f.endswith('_out.pkl')]

    for in_f, out_f in zip(in_files, out_files):
        name = os.path.basename(in_f).replace('_in.pkl', '')
        model_in = joblib.load(in_f)
        model_out = joblib.load(out_f)

        t0 = __import__('time').time()
        pred_in = np.maximum(model_in.predict(X_valid), 0)
        pred_out = np.maximum(model_out.predict(X_valid), 0)
        t = __import__('time').time() - t0

        mae_in = mean_absolute_error(y_valid['inNums'], pred_in)
        mae_out = mean_absolute_error(y_valid['outNums'], pred_out)
        results.append({'name': name, 'mae_in': mae_in, 'mae_out': mae_out,
                        'mae_avg': (mae_in+mae_out)/2, 'time': t})
        models_dict[name] = (model_in, model_out)
        preds_dict[name] = (pred_in, pred_out)
        print(f"    {name}: MAE_in={mae_in:.2f} MAE_out={mae_out:.2f}")

        # 第一个作为 best
        if not best_preds:
            best_preds = {'pred_in': pred_in, 'pred_out': pred_out}

    return results, models_dict, preds_dict, best_preds


def main():
    print("=" * 70)
    print("   B线地铁人流量预测 — 模型对比 (不含ASTGCN)")
    print("   Closed-loop: 训练 01-01~01-22 | 验证 01-23~01-25")
    print("=" * 70)

    # ---- Step 1: 特征工程 ----
    print("\n[Step 1] 特征工程 ...")
    X_train, y_train, X_valid, y_valid, feature_names = build_features()

    # ---- Step 2: 线性模型 ----
    if _check_models_exist(MODEL_DIR_LINEAR, expected_count=7):
        print("\n[Step 2] 加载已有线性模型 ...")
        linear_results, linear_models, linear_preds = _load_and_predict_linear(
            MODEL_DIR_LINEAR, X_train, y_train, X_valid, y_valid
        )
    else:
        print("\n[Step 2] 线性模型训练与评估 ...")
        linear_results, linear_models, linear_preds = run_linear_models(
            X_train, y_train, X_valid, y_valid, model_dir=MODEL_DIR_LINEAR
        )

    # ---- Step 3: 树模型 ----
    if _check_models_exist(MODEL_DIR_TREE, expected_count=3):
        print("\n[Step 3] 加载已有树模型 ...")
        tree_results, tree_models, tree_preds, best_preds = _load_and_predict_tree(
            MODEL_DIR_TREE, X_train, y_train, X_valid, y_valid
        )
    else:
        print("\n[Step 3] 树模型训练与评估 ...")
        tree_results, tree_models, tree_preds, best_preds = run_tree_models(
            X_train, y_train, X_valid, y_valid, model_dir=MODEL_DIR_TREE
        )

    # ---- Step 4: 汇总对比 ----
    all_results = linear_results + tree_results

    print("\n" + "=" * 70)
    print("   模型对比结果 (不含ASTGCN)")
    print("=" * 70)
    print(f"{'模型':<22} {'MAE_in':>8} {'MAE_out':>8} {'MAE_avg':>8} {'耗时(s)':>8}")
    print("-" * 58)

    all_results.sort(key=lambda x: x['mae_avg'])
    for r in all_results:
        print(f"{r['name']:<22} {r['mae_in']:>8.2f} {r['mae_out']:>8.2f} "
              f"{r['mae_avg']:>8.2f} {r['time']:>8.2f}")

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

    # ---- Step 6: 方法间对比折线图 ----
    print("\n[Step 6] 方法间对比折线图 ...")
    df_full2 = _load_data()
    df_full2 = df_full2.sort_values(['stationID', 'time_slot'])
    valid_mask2 = df_full2['date_str'] > "2019-01-22"
    valid_slots2 = df_full2.loc[valid_mask2, 'time_slot'].values
    valid_stations2 = df_full2.loc[valid_mask2, 'stationID'].values
    del df_full2
    visualize_method_comparison(y_valid, linear_preds, tree_preds,
                                valid_slots2, valid_stations2)

    print("\n✅ main_no_ast 完成！")


if __name__ == '__main__':
    main()
