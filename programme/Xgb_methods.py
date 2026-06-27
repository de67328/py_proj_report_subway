# -*- coding: utf-8 -*-
"""
树模型模块
- XGBoost
- LightGBM
- GBDT (sklearn)
返回 (results, models_dict, preds_dict, best_preds)
"""

import numpy as np
import joblib
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import time

MODEL_DIR = r"d:\作业\py展示\programme\models\tree"

_XGB_AVAIL = False
_LGB_AVAIL = False

try:
    import xgboost as xgb
    _XGB_AVAIL = True
except ImportError:
    print("  ⚠ xgboost 未安装，跳过 XGBoost")

try:
    import lightgbm as lgb
    _LGB_AVAIL = True
except ImportError:
    print("  ⚠ lightgbm 未安装，跳过 LightGBM")


def run_tree_models(X_train, y_train, X_valid, y_valid):
    """运行所有树模型，返回 (results, models_dict, preds_dict, best_preds)"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    results = []
    models_dict = {}   # name → (model_in, model_out)
    preds_dict = {}    # name → (pred_in, pred_out)
    best_preds = {}

    # ---- 1. XGBoost ----
    if _XGB_AVAIL:
        name = 'XGBoost'
        print(f"  [树] {name} ...", end=' ', flush=True)
        t0 = time.time()
        model_in = xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1
        )
        model_in.fit(X_train, y_train['inNums'])
        pred_in = np.maximum(model_in.predict(X_valid), 0)

        model_out = xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1
        )
        model_out.fit(X_train, y_train['outNums'])
        pred_out = np.maximum(model_out.predict(X_valid), 0)

        t = time.time() - t0
        mae_in = mean_absolute_error(y_valid['inNums'], pred_in)
        mae_out = mean_absolute_error(y_valid['outNums'], pred_out)
        results.append({'name': name, 'mae_in': mae_in, 'mae_out': mae_out,
                        'mae_avg': (mae_in + mae_out) / 2, 'time': t})
        models_dict[name] = (model_in, model_out)
        preds_dict[name] = (pred_in, pred_out)
        best_preds = {'pred_in': pred_in, 'pred_out': pred_out}
        # 保存
        joblib.dump(model_in, os.path.join(MODEL_DIR, f'{name}_in.pkl'))
        joblib.dump(model_out, os.path.join(MODEL_DIR, f'{name}_out.pkl'))
        print(f"MAE={mae_in:.2f}/{mae_out:.2f} ✓")

    # ---- 2. LightGBM ----
    if _LGB_AVAIL:
        name = 'LightGBM'
        print(f"  [树] {name} ...", end=' ', flush=True)
        t0 = time.time()
        model_in = lgb.LGBMRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbose=-1
        )
        model_in.fit(X_train, y_train['inNums'])
        pred_in = np.maximum(model_in.predict(X_valid), 0)

        model_out = lgb.LGBMRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbose=-1
        )
        model_out.fit(X_train, y_train['outNums'])
        pred_out = np.maximum(model_out.predict(X_valid), 0)

        t = time.time() - t0
        mae_in = mean_absolute_error(y_valid['inNums'], pred_in)
        mae_out = mean_absolute_error(y_valid['outNums'], pred_out)
        results.append({'name': name, 'mae_in': mae_in, 'mae_out': mae_out,
                        'mae_avg': (mae_in + mae_out) / 2, 'time': t})
        models_dict[name] = (model_in, model_out)
        preds_dict[name] = (pred_in, pred_out)
        joblib.dump(model_in, os.path.join(MODEL_DIR, f'{name}_in.pkl'))
        joblib.dump(model_out, os.path.join(MODEL_DIR, f'{name}_out.pkl'))
        print(f"MAE={mae_in:.2f}/{mae_out:.2f} ✓")

    # ---- 3. GBDT (sklearn) ----
    name = 'GBDT_sklearn'
    print(f"  [树] {name} ...", end=' ', flush=True)
    t0 = time.time()
    model_in = GradientBoostingRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
    model_in.fit(X_train, y_train['inNums'])
    pred_in = np.maximum(model_in.predict(X_valid), 0)

    model_out = GradientBoostingRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
    model_out.fit(X_train, y_train['outNums'])
    pred_out = np.maximum(model_out.predict(X_valid), 0)

    t = time.time() - t0
    mae_in = mean_absolute_error(y_valid['inNums'], pred_in)
    mae_out = mean_absolute_error(y_valid['outNums'], pred_out)
    results.append({'name': name, 'mae_in': mae_in, 'mae_out': mae_out,
                    'mae_avg': (mae_in + mae_out) / 2, 'time': t})
    models_dict[name] = (model_in, model_out)
    preds_dict[name] = (pred_in, pred_out)
    joblib.dump(model_in, os.path.join(MODEL_DIR, f'{name}_in.pkl'))
    joblib.dump(model_out, os.path.join(MODEL_DIR, f'{name}_out.pkl'))
    print(f"MAE={mae_in:.2f}/{mae_out:.2f} ✓")

    print(f"  📁 模型已保存至 {MODEL_DIR}/")
    return results, models_dict, preds_dict, best_preds
