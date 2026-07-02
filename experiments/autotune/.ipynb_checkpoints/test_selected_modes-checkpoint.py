#!/usr/bin/env python
"""
测试指定三种模式在全部测试窗口（50个）的表现
模式：no_rule, semantic_top50_theta_max, rl_top50_semantic_best
输出报告含折线图（各模式每个窗口的MASE对比）
"""

import os
import sys
import json
import math
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.autotune.utils import (
    load_config, load_window_data, compute_all_metrics
)


class SelectedModesTester:
    def __init__(self, run_dir: str, round_num: int, config_path: str = None):
        self.run_dir = run_dir
        self.round_num = round_num
        self.config = load_config(config_path)
        self.output_dir = self.config.get('output_dir', 'storage/autotune_results')
        self.run_dir_out = run_dir + "_half"
        self.predictions_dir = os.path.join(self.run_dir_out, "semantic_vs_rl_results_all", "predictions")

        # 指定三种模式
        self.modes = ['no_rule', 'semantic_top50_theta_max', 'rl_top50_semantic_best']

        # 加载全部测试窗口（不截取）
        self.test_df = self._load_test_df()
        self._window_id_to_data = self._build_window_data_map()

        self.results = {}

    def _load_test_df(self) -> pd.DataFrame:
        csv_path = os.path.join(self.output_dir, "collected_windows.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ 未找到采集数据: {csv_path}")

        df = pd.read_csv(csv_path)

        # 获取测试集（全部）
        if 'split' in df.columns:
            test_df = df[df['split'] == 'test'].copy()
        else:
            n = len(df)
            a_end = int(n * 0.5)
            b_mask = pd.Series([False] * n)
            b_mask.iloc[a_end:] = True
            df['split'] = ['A'] * a_end + ['B'] * (n - a_end)
            b_df = df[b_mask].copy().sort_values('window_id').reset_index(drop=True)
            n_b = len(b_df)
            test_size = int(n_b * 0.5)  # 取一半作为测试集
            test_df = b_df.iloc[:test_size].copy()

        test_df = test_df.sort_values('window_id').reset_index(drop=True)
        print(f"📊 使用全部测试窗口，共 {len(test_df)} 个")
        return test_df

    def _build_window_data_map(self) -> Dict:
        window_map = {}
        for _, row in self.test_df.iterrows():
            wid = row.get('window_id')
            wpath = row.get('window_data_path')
            if wid is not None and wpath and os.path.exists(wpath):
                try:
                    wdata = load_window_data(wpath)
                    window_map[wid] = {
                        'test': wdata['test'],
                        'mase_scale': wdata.get('mase_scale', 1.0),
                        'horizon': wdata.get('horizon', 7),
                    }
                except:
                    pass
        print(f"📊 成功加载 {len(window_map)} 个窗口的真实值")
        return window_map

    def _restore_mode_data(self, mode: str) -> Optional[Dict]:
        mode_pred_dir = os.path.join(self.predictions_dir, mode)
        if not os.path.exists(mode_pred_dir):
            print(f"⚠️ 模式 {mode} 的预测目录不存在: {mode_pred_dir}")
            return None

        npy_files = [f for f in os.listdir(mode_pred_dir) if f.startswith('window_') and f.endswith('.npy')]
        if not npy_files:
            print(f"⚠️ 模式 {mode} 没有预测文件")
            return None

        mases, maes, rmses, smapes, owas = [], [], [], [], []
        window_mases = {}

        for npy_file in npy_files:
            try:
                wid_str = npy_file.replace('window_', '').replace('.npy', '')
                wid = int(wid_str)
                if wid not in self._window_id_to_data:
                    continue
                win_data = self._window_id_to_data[wid]
                pred_path = os.path.join(mode_pred_dir, npy_file)
                pred = np.load(pred_path)
                if len(pred) != len(win_data['test']):
                    continue
                metrics = compute_all_metrics(pred, win_data['test'], win_data['mase_scale'])
                mase = metrics.get('mase', float('inf'))
                if mase != float('inf') and not np.isnan(mase):
                    mases.append(mase)
                    maes.append(metrics.get('mae', 0))
                    rmses.append(metrics.get('rmse', 0))
                    smapes.append(metrics.get('smape', 0))
                    owas.append(metrics.get('owa', 0))
                    window_mases[wid] = mase
            except Exception as e:
                print(f"   ⚠️ 处理 {npy_file} 失败: {e}")
                continue

        if not mases:
            return None

        return {
            'mases': mases,
            'maes': maes,
            'rmses': rmses,
            'smapes': smapes,
            'owas': owas,
            'window_mases': window_mases,
            'window_count': len(mases)
        }

    def run(self):
        print("\n" + "=" * 80)
        print("🧪 测试三种模式（全部窗口）")
        print(f"模式: {', '.join(self.modes)}")
        print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        for mode in self.modes:
            data = self._restore_mode_data(mode)
            if data:
                self.results[mode] = data
                print(f"✅ {mode}: 成功加载 {data['window_count']} 个窗口")
            else:
                print(f"❌ {mode}: 无数据")

        if not self.results:
            print("❌ 没有任何模式的数据，请检查预测文件是否存在")
            return

        # 生成文本报告
        self._generate_report()
        # 生成折线图
        self._plot_line_chart()

    def _generate_report(self):
        no_rule_data = self.results.get('no_rule')
        if not no_rule_data:
            print("⚠️ 缺少 no_rule 数据，无法计算百分比")
            return
        no_rule_mase = np.mean(no_rule_data['mases'])

        rows = []
        for mode in self.modes:
            data = self.results.get(mode)
            if not data:
                continue
            avg_mase = np.mean(data['mases'])
            avg_mae = np.mean(data['maes'])
            avg_rmse = np.mean(data['rmses'])
            avg_smape = np.mean(data['smapes'])
            avg_owa = np.mean(data['owas'])
            window_count = data['window_count']
            rows.append({
                'mode': mode,
                'window_count': window_count,
                'avg_mase': avg_mase,
                'avg_mae': avg_mae,
                'avg_rmse': avg_rmse,
                'avg_smape': avg_smape,
                'avg_owa': avg_owa,
            })

        lines = []
        lines.append("=" * 120)
        lines.append("📊 三种模式对比报告（全部窗口）")
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"运行目录: {self.run_dir_out}")
        lines.append(f"窗口数: {len(self.test_df)} 个")
        lines.append("=" * 120)
        lines.append("")
        lines.append("模式说明:")
        lines.append("  - no_rule: 无参考策略")
        lines.append("  - semantic_top50_theta_max: 语义前50%中θ最大的策略作为参考 -> LLM生成策略")
        lines.append("  - rl_top50_semantic_best: θ前50%中语义最匹配的策略作为参考 -> LLM生成策略")
        lines.append("")
        lines.append("★ 预测值来源: llog/cs2_half/semantic_vs_rl_results_all/predictions/")
        lines.append("")
        lines.append(f"{'模式':<28} | {'窗口数':<8} | {'MASE':<20} | {'MAE':<12} | {'RMSE':<12} | {'SMAPE':<12} | {'OWA':<12}")
        lines.append("-" * 160)

        for r in rows:
            mase = r['avg_mase']
            if r['mode'] == 'no_rule':
                change_str = "(基准)"
            else:
                pct = (mase - no_rule_mase) / no_rule_mase * 100
                change_str = f"({pct:+.3f}%)"
            mase_display = f"{mase:.6f} {change_str}"
            lines.append(
                f"{r['mode']:<28} | {r['window_count']:<8} | {mase_display:<20} | {r['avg_mae']:<12.6f} | "
                f"{r['avg_rmse']:<12.6f} | {r['avg_smape']:<12.6f} | {r['avg_owa']:<12.6f}"
            )

        lines.append("-" * 160)
        lines.append("")
        lines.append("📊 汇总:")
        for r in rows:
            if r['mode'] == 'no_rule':
                continue
            mase = r['avg_mase']
            pct = (mase - no_rule_mase) / no_rule_mase * 100
            sign = "变差" if pct > 0 else "变好"
            lines.append(f"  - {r['mode']}: MASE = {mase:.6f} (相对 no_rule {sign} {abs(pct):.3f}%)")

        # 保存文本报告
        output_file = os.path.join(self.run_dir_out, "selected_modes_report.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"\n✅ 文本报告已保存至: {output_file}")
        print("\n".join(lines))

    def _plot_line_chart(self):
        """绘制三种模式的窗口MASE折线图"""
        # 提取各模式的 window_mases
        mode_window_data = {}
        for mode in self.modes:
            data = self.results.get(mode)
            if data and data.get('window_mases'):
                # 按窗口ID排序
                sorted_items = sorted(data['window_mases'].items())
                wids = [w for w, _ in sorted_items]
                mases = [m for _, m in sorted_items]
                mode_window_data[mode] = (wids, mases)

        if len(mode_window_data) < 2:
            print("⚠️ 数据不足，无法绘制折线图")
            return

        # 颜色和标签
        color_map = {
            'no_rule': '#808080',
            'semantic_top50_theta_max': '#D4693A',
            'rl_top50_semantic_best': '#3F7E5C',
        }
        display_names = {
            'no_rule': 'no_rule (基准)',
            'semantic_top50_theta_max': 'semantic_top50_theta_max',
            'rl_top50_semantic_best': 'rl_top50_semantic_best',
        }

        fig, ax = plt.subplots(figsize=(14, 7))

        for mode, (wids, mases) in mode_window_data.items():
            color = color_map.get(mode, '#000000')
            label = display_names.get(mode, mode)
            ax.plot(wids, mases, marker='o', color=color, linewidth=2, markersize=5, label=label)

        ax.set_xlabel('窗口ID', fontsize=12)
        ax.set_ylabel('MASE', fontsize=12)
        ax.set_title('三种模式窗口 MASE 对比折线图（全部窗口）', fontsize=14)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

        # 保存图片
        plot_path = os.path.join(self.run_dir_out, "selected_modes_linechart.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 折线图已保存至: {plot_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="测试三种指定模式（全部窗口）")
    parser.add_argument('--resume', type=str, required=True,
                        help='原始运行目录（如 llog/cs2）')
    parser.add_argument('--round', type=int, required=True,
                        help='指定轮次（如 57）')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径')
    args = parser.parse_args()

    if os.path.exists(args.resume):
        run_dir = args.resume
    else:
        run_dir = os.path.join("llog", args.resume)

    if not os.path.exists(run_dir):
        print(f"❌ 目录不存在: {run_dir}")
        return

    tester = SelectedModesTester(run_dir, args.round, args.config)
    tester.run()


if __name__ == '__main__':
    main()