# py_proj_report_subway

地铁人流量预测 — 机器学习小组展示项目

## 项目简介
基于杭州地铁刷卡数据，预测 B 线各站点未来以 10 分钟为单位的进站/出站人次。

## 技术方案
- **数据范围**：B 线 34 站，25 天数据
- **特征工程**：时间特征 + 滞后特征 + 滚动特征 + 站点编码 (55维)
- **线性模型**：Ridge(L2) / Lasso(L1) / ElasticNet / PCA+Ridge / PLS
- **树模型**：XGBoost / LightGBM / GBDT
- **评估指标**：MAE（与赛题一致）

## 最佳结果
| 模型 | MAE_in | MAE_out | MAE_avg |
|------|--------|---------|---------|
| XGBoost | 18.69 | 15.13 | **16.91** |
| GBDT | 18.75 | 15.20 | 16.98 |
| LightGBM | 18.92 | 15.37 | 17.15 |
| Naive(前一天) | 19.34 | 17.92 | 18.63 |

## 项目结构
```
├── programme/                # 核心代码
│   ├── main.py               # 主流程
│   ├── features.py           # 特征工程
│   ├── Linear_methods.py     # 线性模型
│   ├── Xgb_methods.py        # 树模型
│   └── results_viz.py        # 结果可视化
├── data/                     # 预处理后数据
├── pic/                      # EDA可视化
├── results_viz/              # 预测结果可视化
├── project_memory.md         # 项目笔记
├── eda_visualize.py          # 原始数据EDA
├── b_line_visualize.py       # B线单日可视化
├── metro_topology_viz.py     # 线路拓扑图
└── preprocess_b_line.py      # 数据预处理
```

## 运行方式
```bash
cd programme
python main.py
```
