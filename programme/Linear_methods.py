# -*- coding: utf-8 -*-
"""
线性模型模块
- Ridge (L2)
- Lasso (L1)
- ElasticNet (L1+L2)
- PCA + Ridge
- PLS (Partial Least Squares)
返回 (results, models_dict, preds_dict)
"""

import numpy as np
import joblib
import os
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import time

MODEL_DIR = r"d:\作业\py展示\programme\models\linear"


def _safe_fit_predict(model, X_train, y_train, X_valid, y_valid):
    """安全训练+预测，返回 (y_pred, train_time)"""
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    y_pred = model.predict(X_valid)
    y_pred = np.maximum(y_pred, 0)
    return y_pred, elapsed


def run_linear_models(X_train, y_train, X_valid, y_valid, model_dir=None):
    """运行所有线性模型，返回 (results, models_dict, preds_dict)"""
    if model_dir is None:
        model_dir = MODEL_DIR
    os.makedirs(model_dir, exist_ok=True)
    results = []
    models_dict = {}   # name → model (for reloading)
    preds_dict = {}    # name → (pred_in, pred_out)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_valid_s = scaler.transform(X_valid)

    # ---- 1. Ridge (L2) ----
    name = 'Ridge_L2'
    print(f"  [线性] {name} ...", end=' ', flush=True)
    model = Ridge(alpha=1.0)
    y_pred, t = _safe_fit_predict(model, X_train_s, y_train, X_valid_s, y_valid)
    mae_in = mean_absolute_error(y_valid['inNums'], y_pred[:, 0])
    mae_out = mean_absolute_error(y_valid['outNums'], y_pred[:, 1])
    results.append({'name': name, 'mae_in': mae_in, 'mae_out': mae_out,
                    'mae_avg': (mae_in + mae_out) / 2, 'time': t})
    models_dict[name] = model
    preds_dict[name] = (y_pred[:, 0], y_pred[:, 1])
    joblib.dump(model, os.path.join(model_dir, f'{name}.pkl'))
    print(f"MAE={mae_in:.2f}/{mae_out:.2f} ✓")

    # ---- 2. Lasso (L1) ----
    name = 'Lasso_L1'
    print(f"  [线性] {name} ...", end=' ', flush=True)
    model = Lasso(alpha=0.01, max_iter=5000)
    y_pred, t = _safe_fit_predict(model, X_train_s, y_train, X_valid_s, y_valid)
    mae_in = mean_absolute_error(y_valid['inNums'], y_pred[:, 0])
    mae_out = mean_absolute_error(y_valid['outNums'], y_pred[:, 1])
    results.append({'name': name, 'mae_in': mae_in, 'mae_out': mae_out,
                    'mae_avg': (mae_in + mae_out) / 2, 'time': t})
    models_dict[name] = model
    preds_dict[name] = (y_pred[:, 0], y_pred[:, 1])
    joblib.dump(model, os.path.join(model_dir, f'{name}.pkl'))
    print(f"MAE={mae_in:.2f}/{mae_out:.2f} ✓")

    # ---- 3. ElasticNet ----
    name = 'ElasticNet'
    print(f"  [线性] {name} ...", end=' ', flush=True)
    model = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000)
    y_pred, t = _safe_fit_predict(model, X_train_s, y_train, X_valid_s, y_valid)
    mae_in = mean_absolute_error(y_valid['inNums'], y_pred[:, 0])
    mae_out = mean_absolute_error(y_valid['outNums'], y_pred[:, 1])
    results.append({'name': name, 'mae_in': mae_in, 'mae_out': mae_out,
                    'mae_avg': (mae_in + mae_out) / 2, 'time': t})
    models_dict[name] = model
    preds_dict[name] = (y_pred[:, 0], y_pred[:, 1])
    joblib.dump(model, os.path.join(model_dir, f'{name}.pkl'))
    print(f"MAE={mae_in:.2f}/{mae_out:.2f} ✓")

    # ---- 4. PCA + Ridge (保存 PCA + Ridge 两个对象) ----
    for n_comp in [20, 40]:
        name = f'PCA{n_comp}+Ridge'
        print(f"  [线性] {name} ...", end=' ', flush=True)
        pca = PCA(n_components=n_comp, random_state=42)
        X_train_pca = pca.fit_transform(X_train_s)
        X_valid_pca = pca.transform(X_valid_s)
        model = Ridge(alpha=1.0)
        t0 = time.time()
        model.fit(X_train_pca, y_train)
        t = time.time() - t0
        y_pred = np.maximum(model.predict(X_valid_pca), 0)
        mae_in = mean_absolute_error(y_valid['inNums'], y_pred[:, 0])
        mae_out = mean_absolute_error(y_valid['outNums'], y_pred[:, 1])
        results.append({'name': name, 'mae_in': mae_in, 'mae_out': mae_out,
                        'mae_avg': (mae_in + mae_out) / 2, 'time': t})
        models_dict[name] = {'pca': pca, 'ridge': model}
        preds_dict[name] = (y_pred[:, 0], y_pred[:, 1])
        joblib.dump({'pca': pca, 'ridge': model}, os.path.join(model_dir, f'{name}.pkl'))
        print(f"MAE={mae_in:.2f}/{mae_out:.2f} ✓")

    # ---- 5. PLS ----
    for n_comp in [10, 20]:
        name = f'PLS{n_comp}'
        print(f"  [线性] {name} ...", end=' ', flush=True)
        model = PLSRegression(n_components=n_comp, scale=True)
        t0 = time.time()
        model.fit(X_train_s, y_train)
        t = time.time() - t0
        y_pred = np.maximum(model.predict(X_valid_s), 0)
        mae_in = mean_absolute_error(y_valid['inNums'], y_pred[:, 0])
        mae_out = mean_absolute_error(y_valid['outNums'], y_pred[:, 1])
        results.append({'name': name, 'mae_in': mae_in, 'mae_out': mae_out,
                        'mae_avg': (mae_in + mae_out) / 2, 'time': t})
        models_dict[name] = model
        preds_dict[name] = (y_pred[:, 0], y_pred[:, 1])
        joblib.dump(model, os.path.join(model_dir, f'{name}.pkl'))
        print(f"MAE={mae_in:.2f}/{mae_out:.2f} ✓")

    # 保存 scaler
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    print(f"  📁 模型已保存至 {model_dir}/")

    return results, models_dict, preds_dict
