# -*- coding: utf-8 -*-
"""
ASTGCN 主入口
- 数据预处理（如需要）
- EDA（可选）
- 训练
- 可视化
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description='ASTGCN 全三线地铁人流量预测')
    parser.add_argument('--preprocess', action='store_true', help='先运行数据预处理')
    parser.add_argument('--eda', action='store_true', help='先运行 EDA')
    parser.add_argument('--skip-train', action='store_true', help='跳过训练（仅可视化）')
    args = parser.parse_args()

    # ---- 预处理 ----
    if args.preprocess:
        print("\n[Preprocess] 数据预处理...")
        from preprocess import main as preprocess_main
        # 直接执行预处理脚本
        import preprocess
        # preprocess 脚本已经在 import 时执行，这里不需要额外调用

    # ---- EDA ----
    if args.eda:
        print("\n[EDA] 探索性数据分析...")
        import subprocess
        subprocess.run([sys.executable, os.path.join(
            os.path.dirname(__file__), 'eda_all_lines.py'
        )])

    # ---- 训练 ----
    if not args.skip_train:
        from train import train
        from visualize import (plot_training_curves, plot_predictions,
                               plot_gcn_embeddings)

        result, history, model, (preds, targets) = train()

        print("\n" + "=" * 60)
        print("  最终结果")
        print("=" * 60)
        print(f"  模型: {result['name']}")
        print(f"  设备: {result['device']}")
        print(f"  参数: {result['params']:,}")
        print(f"  MAE_in:  {result['mae_in']:.2f}")
        print(f"  MAE_out: {result['mae_out']:.2f}")
        print(f"  MAE_avg: {result['mae_avg']:.2f}")
        print(f"  耗时:    {result['time']:.0f}s")
        print(f"  最佳轮:  {result['best_epoch']}")

        # ---- 可视化 ----
        print("\n[Visualization] 生成图表...")
        plot_training_curves(history)
        plot_predictions(preds, targets)

        # GCN 嵌入可视化
        from graph_utils import build_graph_data
        _, _, adj, _ = build_graph_data()
        adj_t = adj.to(next(model.parameters()).device)
        plot_gcn_embeddings(model, adj_t, None)

        print(f"\n✅ 全部完成！结果保存在 AST/results/ 和 AST/models/")


if __name__ == '__main__':
    main()
