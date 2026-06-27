# -*- coding: utf-8 -*-
"""计算 XGBoost 模型的 R² — 全线及各站点"""
import sys
sys.path.insert(0, r'd:\作业\py展示\programme')

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from features import build_features, _load_data
import xgboost as xgb

# 加载
X_train, y_train, X_valid, y_valid, _ = build_features()

# XGBoost
model_in = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
model_in.fit(X_train, y_train['inNums'])
pred_in = np.maximum(model_in.predict(X_valid), 0)

model_out = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
model_out.fit(X_train, y_train['outNums'])
pred_out = np.maximum(model_out.predict(X_valid), 0)

actual_in = y_valid['inNums'].values
actual_out = y_valid['outNums'].values

# 全线 R2
r2_in_all = r2_score(actual_in, pred_in)
r2_out_all = r2_score(actual_out, pred_out)
print(f"All-line R2 - in: {r2_in_all:.4f}  out: {r2_out_all:.4f}  avg: {(r2_in_all+r2_out_all)/2:.4f}")

# 按站点
df_full = _load_data()
df_full = df_full.sort_values(['stationID', 'time_slot'])
valid_mask = df_full['date_str'] > "2019-01-22"
valid_stations = df_full.loc[valid_mask, 'stationID'].values

df_r2 = pd.DataFrame({
    'station': valid_stations,
    'actual_in': actual_in, 'pred_in': pred_in,
    'actual_out': actual_out, 'pred_out': pred_out,
})

print("\nPer-station R2:")
print(f"{'st':>4} {'R2_in':>8} {'R2_out':>8} {'R2_avg':>8}")
print("-" * 32)
per_station = {}
for sid in sorted(df_r2['station'].unique()):
    mask = df_r2['station'] == sid
    r2i = r2_score(df_r2.loc[mask, 'actual_in'], df_r2.loc[mask, 'pred_in'])
    r2o = r2_score(df_r2.loc[mask, 'actual_out'], df_r2.loc[mask, 'pred_out'])
    per_station[sid] = (r2i, r2o)
    print(f"  {sid:>3} {r2i:>8.4f} {r2o:>8.4f} {(r2i+r2o)/2:>8.4f}")

r2i_vals = [v[0] for v in per_station.values()]
r2o_vals = [v[1] for v in per_station.values()]
print(f"\nStation R2 range: in [{min(r2i_vals):.4f}, {max(r2i_vals):.4f}]  out [{min(r2o_vals):.4f}, {max(r2o_vals):.4f}]")
print(f"Station R2 mean:  in {np.mean(r2i_vals):.4f}  out {np.mean(r2o_vals):.4f}")
neg_in = sum(1 for v in r2i_vals if v < 0)
neg_out = sum(1 for v in r2o_vals if v < 0)
print(f"Negative R2 stations: in={neg_in}  out={neg_out}")
