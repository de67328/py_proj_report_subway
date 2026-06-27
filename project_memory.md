# 地铁人流量预测 - 项目记忆

## 赛题概述
- **目标**：预测未来一天各站点以10分钟为单位的进站(inNums)和出站(outNums)人次
- **训练数据**：2019-01-01 至 2019-01-25，共25天，约7000万条刷卡记录
- **线路**：3条线路（A/B/C），81个地铁站（stationID 0-80，54号缺失）
- **评估指标**：MAE（入站MAE + 出站MAE的平均值）
- **建模范围**：仅 B 线（站点 0-33，34站），Closed-loop 设定

## 数据字段说明
| 字段 | 含义 |
|------|------|
| time | 刷卡时间 |
| lineID | 线路ID (A/B/C) |
| stationID | 站点ID (0-80) |
| deviceID | 设备ID |
| status | 0=进站, 1=出站 |
| userID | 用户ID (payType=3时非唯一) |
| payType | 支付类型 (3=非唯一用户) |

## 预测输出格式
- 每10分钟一个时间段：[startTime, endTime)
- 需预测：inNums（进站人数）, outNums（出站人数）
- 示例：`0,2019-01-29 00:40:00,2019-01-29 00:50:00,100.0,200.0`

## 当前进度
- [x] 理解赛题
- [x] 数据EDA与可视化（record_2019-01-01）
- [x] 线路拓扑可视化（pic/metro_topology.png）
- [x] B线数据预处理（data/b_line_*.csv）
- [x] B线单日可视化（pic/01~07_*.png）
- [ ] 特征工程
- [ ] 模型训练与对比
- [ ] 预测与提交

## 技术方案

### Closed-loop 设定
使用 1月1日~22日 训练，1月23日~25日 验证，所有特征仅依赖过去数据。

### 特征工程（统一，所有模型共用）
- **时间特征**：hour, minute, weekday, is_weekend, is_peak, slot_id
- **滞后特征**：前1/2/3/7天同时段同站点流量（lag_1d, lag_2d, lag_3d, lag_7d）
- **滚动特征**：同站点前1/2/3个时段的流量（rolling_1, rolling_2, rolling_3）
- **站点编码**：One-Hot Encoding（34站）

### 模型对比方案

| 类别 | 模型 | 说明 |
|------|------|------|
| 线性 | Ridge (L2) | 岭回归 |
| 线性 | Lasso (L1) | 可做特征选择 |
| 线性 | ElasticNet | L1+L2 混合 |
| 线性 | PCA + Ridge | 降维后回归 |
| 线性 | PLS | 偏最小二乘 |
| 树模型 | XGBoost | 梯度提升 |
| 树模型 | LightGBM | 轻量级GBDT |
| 树模型 | CatBoost | 类别特征友好 |
| 树模型 | GBDT (sklearn) | 经典梯度提升 |

### 评估
- MAE（与赛题一致）
- 分别评估 inNums 和 outNums，取平均

## 目录结构
```
├── Hangzhou-mobility-data-set/   # 原始数据
├── data/                         # 预处理后数据
│   ├── b_line_train.csv
│   ├── b_line_valid.csv
│   └── b_line_full.csv
├── pic/                          # 可视化输出
├── programme/                    # 核心代码
│   ├── main.py                   # 主流程
│   ├── features.py               # 特征工程
│   ├── Linear_methods.py         # 线性模型
│   └── Xgb_methods.py            # 树模型
├── project_memory.md
├── eda_visualize.py
├── metro_topology_viz.py
├── preprocess_b_line.py
└── b_line_visualize.py
```
