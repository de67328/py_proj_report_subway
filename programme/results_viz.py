# -*- coding: utf-8 -*-
"""
预测结果可视化 — 对比实际 vs 预测
- 选取最佳模型 XGBoost 的结果
- 多张独立图输出到 results_viz/ 目录
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import time

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = r"d:\作业\py展示\results_viz"


def fmt_xaxis(ax, interval=4):
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, fontsize=7)


def visualize_results(y_valid, pred_in, pred_out, time_slots, station_ids):
    """核心可视化：实际 vs 预测"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    actual_in = y_valid['inNums'].values
    actual_out = y_valid['outNums'].values

    # ---- 图1: 全线汇总 — 实际 vs 预测 时间序列 ----
    fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig1.suptitle('B线全线 实际 vs 预测 (验证集 01-23~01-25)', fontsize=14, fontweight='bold')

    # 按 time_slot 汇总
    df_viz = pd.DataFrame({
        'time': time_slots,
        'actual_in': actual_in, 'pred_in': pred_in,
        'actual_out': actual_out, 'pred_out': pred_out,
    })
    hourly_in = df_viz.groupby('time')[['actual_in', 'pred_in']].sum()
    hourly_out = df_viz.groupby('time')[['actual_out', 'pred_out']].sum()

    ax1a.fill_between(hourly_in.index, hourly_in['actual_in'], alpha=0.3, color='#4ECDC4', label='实际')
    ax1a.plot(hourly_in.index, hourly_in['actual_in'], color='#2BA89E', linewidth=1.2)
    ax1a.plot(hourly_in.index, hourly_in['pred_in'], color='#FF6B6B', linewidth=1.5, label='预测', alpha=0.9)
    ax1a.set_ylabel('进站人次/10min', fontsize=11)
    ax1a.legend(loc='upper right', fontsize=9)
    ax1a.grid(True, alpha=0.3)
    ax1a.set_title('进站', fontsize=12, fontweight='bold')

    ax1b.fill_between(hourly_out.index, hourly_out['actual_out'], alpha=0.3, color='#4ECDC4', label='实际')
    ax1b.plot(hourly_out.index, hourly_out['actual_out'], color='#2BA89E', linewidth=1.2)
    ax1b.plot(hourly_out.index, hourly_out['pred_out'], color='#FF6B6B', linewidth=1.5, label='预测', alpha=0.9)
    ax1b.set_ylabel('出站人次/10min', fontsize=11)
    ax1b.legend(loc='upper right', fontsize=9)
    ax1b.grid(True, alpha=0.3)
    ax1b.set_title('出站', fontsize=12, fontweight='bold')

    fmt_xaxis(ax1b)
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUTPUT_DIR, '01_full_actual_vs_pred.png'), dpi=150)
    plt.close(fig1)
    print("  [1/6] 全线实际vs预测 → 01_full_actual_vs_pred.png")

    # ---- 图2: 散点图 — 实际 vs 预测 ----
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle('实际值 vs 预测值 散点图', fontsize=14, fontweight='bold')

    mx_in = max(actual_in.max(), pred_in.max())
    ax2a.scatter(actual_in, pred_in, alpha=0.15, s=3, c='#4D96FF', edgecolors='none')
    ax2a.plot([0, mx_in], [0, mx_in], 'r--', alpha=0.5, linewidth=1)
    ax2a.set_xlim(0, mx_in * 1.02)
    ax2a.set_ylim(0, mx_in * 1.02)
    ax2a.set_xlabel('实际进站', fontsize=11)
    ax2a.set_ylabel('预测进站', fontsize=11)
    ax2a.set_title('进站', fontsize=12, fontweight='bold')
    ax2a.grid(True, alpha=0.3)

    mx_out = max(actual_out.max(), pred_out.max())
    ax2b.scatter(actual_out, pred_out, alpha=0.15, s=3, c='#FF6B6B', edgecolors='none')
    ax2b.plot([0, mx_out], [0, mx_out], 'r--', alpha=0.5, linewidth=1)
    ax2b.set_xlim(0, mx_out * 1.02)
    ax2b.set_ylim(0, mx_out * 1.02)
    ax2b.set_xlabel('实际出站', fontsize=11)
    ax2b.set_ylabel('预测出站', fontsize=11)
    ax2b.set_title('出站', fontsize=12, fontweight='bold')
    ax2b.grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, '02_scatter.png'), dpi=150)
    plt.close(fig2)
    print("  [2/6] 散点图 → 02_scatter.png")

    # ---- 图3: 误差分布直方图 ----
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))
    fig3.suptitle('预测误差分布', fontsize=14, fontweight='bold')

    err_in = pred_in - actual_in
    err_out = pred_out - actual_out

    ax3a.hist(err_in, bins=80, color='#4D96FF', alpha=0.7, edgecolor='white')
    ax3a.axvline(0, color='red', linestyle='--', linewidth=1.5)
    ax3a.axvline(err_in.mean(), color='orange', linestyle='-', linewidth=1.5, label=f'均值={err_in.mean():.2f}')
    ax3a.set_xlabel('预测 - 实际 (进站)', fontsize=11)
    ax3a.set_ylabel('频次', fontsize=11)
    ax3a.set_title('进站误差分布', fontsize=12, fontweight='bold')
    ax3a.legend(fontsize=9)

    ax3b.hist(err_out, bins=80, color='#FF6B6B', alpha=0.7, edgecolor='white')
    ax3b.axvline(0, color='red', linestyle='--', linewidth=1.5)
    ax3b.axvline(err_out.mean(), color='orange', linestyle='-', linewidth=1.5, label=f'均值={err_out.mean():.2f}')
    ax3b.set_xlabel('预测 - 实际 (出站)', fontsize=11)
    ax3b.set_ylabel('频次', fontsize=11)
    ax3b.set_title('出站误差分布', fontsize=12, fontweight='bold')
    ax3b.legend(fontsize=9)

    fig3.tight_layout()
    fig3.savefig(os.path.join(OUTPUT_DIR, '03_error_dist.png'), dpi=150)
    plt.close(fig3)
    print("  [3/6] 误差分布 → 03_error_dist.png")

    # ---- 图4: 各站点 MAE ----
    fig4, ax4 = plt.subplots(figsize=(12, 7))
    df_viz['station'] = station_ids
    station_mae = df_viz.groupby('station').apply(
        lambda g: pd.Series({
            'mae_in': np.mean(np.abs(g['actual_in'] - g['pred_in'])),
            'mae_out': np.mean(np.abs(g['actual_out'] - g['pred_out'])),
        }),
        include_groups=False
    ).sort_values('mae_in', ascending=True)

    y = range(len(station_mae))
    ax4.barh([p - 0.15 for p in y], station_mae['mae_in'], height=0.3,
             color='#4D96FF', alpha=0.8, label='进站 MAE')
    ax4.barh([p + 0.15 for p in y], station_mae['mae_out'], height=0.3,
             color='#FF6B6B', alpha=0.8, label='出站 MAE')
    ax4.set_yticks(y)
    ax4.set_yticklabels(station_mae.index, fontsize=9)
    ax4.set_xlabel('MAE', fontsize=11)
    ax4.set_title('各站点 MAE (XGBoost)', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='x')

    fig4.tight_layout()
    fig4.savefig(os.path.join(OUTPUT_DIR, '04_station_mae.png'), dpi=150)
    plt.close(fig4)
    print("  [4/6] 站点MAE → 04_station_mae.png")

    # ---- 图5: 选取3个典型站点，画实际 vs 预测时序 ----
    # 找 MAE 最好、中等、最差的站点
    station_mae_sorted = station_mae.assign(total=station_mae['mae_in'] + station_mae['mae_out']).sort_values('total')
    pick_stations = [
        station_mae_sorted.index[0],        # 最好
        station_mae_sorted.index[len(station_mae_sorted)//2],  # 中等
        station_mae_sorted.index[-1],        # 最差
    ]

    fig5, axes5 = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig5.suptitle('不同站点 实际 vs 预测 对比 (XGBoost)', fontsize=14, fontweight='bold')

    labels_map = {pick_stations[0]: '最佳', pick_stations[1]: '中等', pick_stations[2]: '最差'}
    for i, sid in enumerate(pick_stations):
        ax = axes5[i]
        mask = df_viz['station'] == sid
        ts = df_viz.loc[mask, 'time']
        ax.plot(ts, df_viz.loc[mask, 'actual_in'], color='#2BA89E', linewidth=1.2, label='实际进站')
        ax.plot(ts, df_viz.loc[mask, 'pred_in'], color='#FF6B6B', linewidth=1.2, label='预测进站', alpha=0.7)
        ax.set_ylabel('人次', fontsize=10)
        ax.set_title(f'站点 {sid} ({labels_map[sid]})', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    fmt_xaxis(axes5[-1])
    fig5.tight_layout()
    fig5.savefig(os.path.join(OUTPUT_DIR, '05_typical_stations.png'), dpi=150)
    plt.close(fig5)
    print("  [5/6] 典型站点对比 → 05_typical_stations.png")

    # ---- 图6: 全天分时段 MAE (按小时汇总) ----
    fig6, ax6 = plt.subplots(figsize=(14, 5))
    df_viz['hour'] = df_viz['time'].dt.hour
    hourly_mae = df_viz.groupby('hour').apply(
        lambda g: pd.Series({
            'mae_in': np.mean(np.abs(g['actual_in'] - g['pred_in'])),
            'mae_out': np.mean(np.abs(g['actual_out'] - g['pred_out'])),
        }),
        include_groups=False
    )

    ax6.bar(hourly_mae.index - 0.15, hourly_mae['mae_in'], width=0.3,
            color='#4D96FF', alpha=0.8, label='进站 MAE')
    ax6.bar(hourly_mae.index + 0.15, hourly_mae['mae_out'], width=0.3,
            color='#FF6B6B', alpha=0.8, label='出站 MAE')
    ax6.set_xlabel('小时', fontsize=11)
    ax6.set_ylabel('MAE', fontsize=11)
    ax6.set_title('各时段 MAE (XGBoost)', fontsize=14, fontweight='bold')
    ax6.set_xticks(range(0, 24))
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3, axis='y')

    fig6.tight_layout()
    fig6.savefig(os.path.join(OUTPUT_DIR, '06_hourly_mae.png'), dpi=150)
    plt.close(fig6)
    print("  [6/6] 分时段MAE → 06_hourly_mae.png")

    print(f"\n✅ 6张图已保存至 {OUTPUT_DIR}/")


def visualize_method_comparison(y_valid, linear_preds, tree_preds, time_slots, station_ids):
    """
    方法间对比折线图：
    - 图1: 所有线性方法 vs 实际值 (选取最繁忙站点)
    - 图2: 所有树方法 vs 实际值
    每张图包含多种方法的折线，不同颜色标注
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 找到最繁忙站点的数据索引
    df_viz = pd.DataFrame({
        'time': time_slots,
        'actual_in': y_valid['inNums'].values,
        'actual_out': y_valid['outNums'].values,
        'station': station_ids,
    })
    top_station = df_viz.groupby('station')['actual_in'].sum().idxmax()
    mask = df_viz['station'] == top_station
    ts = df_viz.loc[mask, 'time'].values
    actual_in_st = df_viz.loc[mask, 'actual_in'].values

    # 颜色方案
    lin_colors = ['#FF6B6B', '#4D96FF', '#6BCB77', '#FFD93D', '#C39BD3', '#FF8C42', '#45B7D1']
    tree_colors = ['#FF6B6B', '#4D96FF', '#6BCB77']

    # ---- 图1: 线性方法对比折线 (进站) ----
    fig1, ax1 = plt.subplots(figsize=(16, 6))
    ax1.plot(ts, actual_in_st, 'k-', linewidth=2.5, label='实际值', zorder=10)

    for i, (name, (pred_in, _)) in enumerate(linear_preds.items()):
        pred_st = pred_in[mask.values]
        ax1.plot(ts, pred_st, linewidth=1.3, color=lin_colors[i % len(lin_colors)],
                 label=name, alpha=0.85)

    ax1.set_title(f'线性方法对比 — 站点{top_station} 进站预测 ({", ".join(linear_preds.keys())})',
                  fontsize=13, fontweight='bold')
    ax1.set_ylabel('进站人次/10min', fontsize=11)
    ax1.legend(loc='upper right', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    fmt_xaxis(ax1)
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUTPUT_DIR, '07_linear_comparison.png'), dpi=150)
    plt.close(fig1)
    print("  [7/8] 线性方法对比 → 07_linear_comparison.png")

    # ---- 图2: 树方法对比折线 (进站) ----
    fig2, ax2 = plt.subplots(figsize=(16, 6))
    ax2.plot(ts, actual_in_st, 'k-', linewidth=2.5, label='实际值', zorder=10)

    for i, (name, (pred_in, _)) in enumerate(tree_preds.items()):
        pred_st = pred_in[mask.values]
        ax2.plot(ts, pred_st, linewidth=1.5, color=tree_colors[i % len(tree_colors)],
                 label=name, alpha=0.85)

    ax2.set_title(f'树方法对比 — 站点{top_station} 进站预测 ({", ".join(tree_preds.keys())})',
                  fontsize=13, fontweight='bold')
    ax2.set_ylabel('进站人次/10min', fontsize=11)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    fmt_xaxis(ax2)
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, '08_tree_comparison.png'), dpi=150)
    plt.close(fig2)
    print("  [8/8] 树方法对比 → 08_tree_comparison.png")

    print(f"✅ 对比图已保存至 {OUTPUT_DIR}/")


def visualize_astgcn_comparison(y_valid, xgb_preds, astgcn_preds):
    """
    ASTGCN vs XGBoost 对比图
    - 散点图: XGBoost预测 vs ASTGCN预测 vs 实际值
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    actual_in = y_valid['inNums'].values
    actual_out = y_valid['outNums'].values

    # ASTGCN 只预测了部分样本（滑动窗口），取交集
    n_astgcn = len(astgcn_preds['pred_in'])
    n_xgb = len(xgb_preds['pred_in'])
    n_common = min(n_astgcn, n_xgb)

    xgb_in = xgb_preds['pred_in'][:n_common]
    xgb_out = xgb_preds['pred_out'][:n_common]
    astgcn_in = astgcn_preds['pred_in'][:n_common]
    astgcn_out = astgcn_preds['pred_out'][:n_common]
    actual_in_c = actual_in[:n_common]
    actual_out_c = actual_out[:n_common]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('ASTGCN vs XGBoost 预测对比', fontsize=15, fontweight='bold')

    # 左上: XGBoost散点
    mx = max(actual_in_c.max(), xgb_in.max()) * 1.05
    ax1.scatter(actual_in_c, xgb_in, alpha=0.2, s=4, c='#4D96FF', edgecolors='none', label='XGBoost')
    ax1.plot([0, mx], [0, mx], 'r--', alpha=0.5)
    ax1.set_xlim(0, mx); ax1.set_ylim(0, mx)
    ax1.set_xlabel('实际进站'); ax1.set_ylabel('预测进站')
    ax1.set_title('XGBoost 进站', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 右上: ASTGCN散点
    mx = max(actual_in_c.max(), astgcn_in.max()) * 1.05
    ax2.scatter(actual_in_c, astgcn_in, alpha=0.2, s=4, c='#FF6B6B', edgecolors='none', label='ASTGCN')
    ax2.plot([0, mx], [0, mx], 'r--', alpha=0.5)
    ax2.set_xlim(0, mx); ax2.set_ylim(0, mx)
    ax2.set_xlabel('实际进站'); ax2.set_ylabel('预测进站')
    ax2.set_title('ASTGCN 进站', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 左下: 误差分布对比
    err_xgb = xgb_in - actual_in_c
    err_astgcn = astgcn_in - actual_in_c
    ax3.hist(err_xgb, bins=60, alpha=0.5, color='#4D96FF', label=f'XGBoost (MAE={np.mean(np.abs(err_xgb)):.1f})')
    ax3.hist(err_astgcn, bins=60, alpha=0.5, color='#FF6B6B', label=f'ASTGCN (MAE={np.mean(np.abs(err_astgcn)):.1f})')
    ax3.axvline(0, color='black', linestyle='--', linewidth=1)
    ax3.set_xlabel('预测 - 实际 (进站)')
    ax3.set_ylabel('频次')
    ax3.set_title('进站误差分布对比', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)

    # 右下: 出站误差分布对比
    err_xgb_o = xgb_out - actual_out_c
    err_astgcn_o = astgcn_out - actual_out_c
    ax4.hist(err_xgb_o, bins=60, alpha=0.5, color='#4D96FF', label=f'XGBoost (MAE={np.mean(np.abs(err_xgb_o)):.1f})')
    ax4.hist(err_astgcn_o, bins=60, alpha=0.5, color='#FF6B6B', label=f'ASTGCN (MAE={np.mean(np.abs(err_astgcn_o)):.1f})')
    ax4.axvline(0, color='black', linestyle='--', linewidth=1)
    ax4.set_xlabel('预测 - 实际 (出站)')
    ax4.set_ylabel('频次')
    ax4.set_title('出站误差分布对比', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '09_astgcn_vs_xgb.png'), dpi=150)
    plt.close(fig)
    print("  ASTGCN对比图 → 09_astgcn_vs_xgb.png")


# ============================================================
# 独立运行测试
# ============================================================
if __name__ == '__main__':
    print("此模块由 main.py 调用，不单独运行。")
    print("请运行 python main.py")
