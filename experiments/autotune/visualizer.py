# experiments/autotune/visualizer.py
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from datetime import datetime

from experiments.autotune.utils import ProgressLogger


class ResultVisualizer:
    def __init__(self, config: Dict, logger: ProgressLogger):
        self.config = config
        self.logger = logger
        self.output_dir = config.get('output_dir', 'storage/autotune_results')
        self.vis_config = config.get('visualization', {})
        self.enabled = self.vis_config.get('enabled', True)
        self.save_plots = self.vis_config.get('save_plots', True)
        self.show_plots = self.vis_config.get('show_plots', False)

    def visualize(self, validation_results: Dict, dataset_name: str, output_dir: str = None):
        if not self.enabled:
            return

        if not validation_results or validation_results.get('total_windows', 0) == 0:
            self.logger.log("⚠️ 无有效验证结果，跳过可视化")
            return

        self.logger.log(f"\n📊 生成可视化: {dataset_name}")

        output_dir = output_dir or self.output_dir
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

        window_results = validation_results.get('window_results', [])
        mases = validation_results.get('mases', [])

        if not mases:
            self.logger.log("⚠️ 无有效 MASE 数据，跳过可视化")
            return

        avg_mase = validation_results.get('avg_mase', 0)
        threshold = validation_results.get('threshold', avg_mase * 1.2)
        hard_count = validation_results.get('hard_count', 0)
        total_windows = validation_results.get('total_windows', 0)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        window_ids = [r.get('window_id', f'w{i}') for i, r in enumerate(window_results)] if window_results else list(range(len(mases)))
        colors = ['red' if m > threshold else 'steelblue' for m in mases]

        axes[0].bar(range(len(mases)), mases, color=colors, alpha=0.8)
        axes[0].axhline(y=avg_mase, color='green', linestyle='--', label=f'平均 MASE = {avg_mase:.4f}')
        axes[0].axhline(y=threshold, color='orange', linestyle='--', label=f'阈值 = {threshold:.4f}')
        axes[0].set_xlabel('窗口 ID')
        axes[0].set_ylabel('MASE')
        axes[0].set_title(f'{dataset_name} - 验证集窗口 MASE 分布\n(困难窗口: {hard_count}/{total_windows})')
        axes[0].set_xticks(range(len(mases)))
        axes[0].set_xticklabels(window_ids, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)

        bins = min(10, len(mases))
        if bins > 0:
            axes[1].hist(mases, bins=bins, color='steelblue', alpha=0.7, edgecolor='black')
            axes[1].axvline(x=avg_mase, color='green', linestyle='--', label=f'平均 MASE = {avg_mase:.4f}')
            axes[1].axvline(x=threshold, color='orange', linestyle='--', label=f'阈值 = {threshold:.4f}')
            axes[1].set_xlabel('MASE')
            axes[1].set_ylabel('频数')
            axes[1].set_title(f'{dataset_name} - MASE 分布直方图')
            axes[1].legend()
            axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if self.save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fig_path = os.path.join(vis_dir, f'{dataset_name}_validation_{timestamp}.png')
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            self.logger.log(f"   📁 图表已保存: {fig_path}")

        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)

        stats = {
            'dataset': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'avg_mase': avg_mase,
            'std_mase': np.std(mases) if mases else 0,
            'min_mase': min(mases) if mases else 0,
            'max_mase': max(mases) if mases else 0,
            'threshold': threshold,
            'hard_count': hard_count,
            'total_windows': total_windows,
            'hard_ratio': hard_count / max(1, total_windows),
            'threshold_multiplier': validation_results.get('threshold_multiplier', 1.2),
            'worst_3_window_ids': validation_results.get('worst_3_window_ids', []),
            'window_details': window_results
        }

        stats_path = os.path.join(vis_dir, f'{dataset_name}_validation_stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        self.logger.log(f"   📁 统计信息已保存: {stats_path}")
        self.logger.log(f"   📊 验证摘要: avg_mase={avg_mase:.4f}, 困难窗口={hard_count}/{total_windows}")

    def generate_report(self, validation_results: Dict, dataset_name: str, output_dir: str = None):
        if not validation_results or validation_results.get('total_windows', 0) == 0:
            self.logger.log("⚠️ 无有效验证结果，跳过报告生成")
            return

        output_dir = output_dir or self.output_dir
        report_dir = os.path.join(output_dir, 'reports')
        os.makedirs(report_dir, exist_ok=True)

        avg_mase = validation_results.get('avg_mase', 0)
        hard_count = validation_results.get('hard_count', 0)
        total_windows = validation_results.get('total_windows', 0)
        worst_3 = validation_results.get('worst_3_window_ids', [])

        report_lines = [
            "=" * 60,
            f"📊 {dataset_name} 规则验证报告",
            "=" * 60,
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"📈 平均 MASE: {avg_mase:.4f}",
            f"🎯 困难窗口数: {hard_count}/{total_windows} ({hard_count/max(1,total_windows):.2%})",
            f"🔴 MASE 最差 3 个窗口 ID: {worst_3}",
            "",
            "=" * 60,
            "📋 各窗口详情:",
            "-" * 60,
        ]

        for r in validation_results.get('window_results', []):
            report_lines.append(
                f"  窗口 {r.get('window_id', 'unknown')}: MASE={r.get('mase', 0):.4f}, "
                f"origin={r.get('origin', 0)}, train_size={r.get('train_size', 0)}"
            )

        report_path = os.path.join(report_dir, f'{dataset_name}_validation_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        self.logger.log(f"   📁 报告已保存: {report_path}")