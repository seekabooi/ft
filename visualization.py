import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# 设置中文字体（如果系统支持）或使用英文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

class Visualizer:
    def __init__(self, dataset_name, storage_dir='storage'):
        self.dataset = dataset_name
        self.storage = storage_dir
        self.eval_csv = os.path.join(storage_dir, f'eval_{dataset_name}.csv')
        # 寻找最新的日志文件（按名称匹配）
        self.log_file = self._find_latest_log(dataset_name)

    def _find_latest_log(self, dataset_name):
        log_dir = os.path.join(self.storage, 'logs')
        if not os.path.exists(log_dir):
            return None
        candidates = []
        for f in os.listdir(log_dir):
            if f.startswith(f'agent_{dataset_name}') and f.endswith('.log'):
                candidates.append(os.path.join(log_dir, f))
        if not candidates:
            return None
        # 按修改时间最新
        latest = max(candidates, key=os.path.getmtime)
        return latest

    def load_predictions(self):
        if not os.path.exists(self.eval_csv):
            raise FileNotFoundError(f'未找到评估结果文件: {self.eval_csv}')
        df = pd.read_csv(self.eval_csv)
        return df['actual'].values, df['prediction'].values

    def load_weights_from_log(self):
        """
        从日志中提取每一步的技能权重。
        日志格式：每行一个 JSON，含 event 字段。
        我们关心 'llm_plan_new' 和 'llm_plan_reused' 事件中的 'plan' 的 'skill_weights'。
        """
        if not self.log_file or not os.path.exists(self.log_file):
            return None
        weights_by_step = {}
        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                except:
                    continue
                if data.get('event') in ('llm_plan_new', 'llm_plan_reused'):
                    step = data.get('step') or data.get('task_id', '')
                    plan = data.get('plan', {})
                    skill_weights = plan.get('skill_weights', {})
                    if skill_weights:
                        weights_by_step[step] = skill_weights
        return weights_by_step

    def plot_predictions_vs_actuals(self, save_path=None):
        actuals, preds = self.load_predictions()
        plt.figure(figsize=(12, 6))
        x = range(len(actuals))
        plt.plot(x, actuals, 'b-o', label='Actual', markersize=4, linewidth=1)
        plt.plot(x, preds, 'r--s', label='Predicted', markersize=4, linewidth=1)
        plt.title(f'{self.dataset} - Actual vs Predicted')
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def plot_error_distribution(self, save_path=None):
        actuals, preds = self.load_predictions()
        errors = preds - actuals
        plt.figure(figsize=(10, 5))
        plt.hist(errors, bins=20, edgecolor='k', alpha=0.7, density=True)
        # 拟合正态分布曲线
        from scipy import stats
        mu, std = stats.norm.fit(errors)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        plt.plot(x, p, 'r', linewidth=2, label=f'Normal fit (μ={mu:.2f}, σ={std:.2f})')
        plt.axvline(0, color='grey', linestyle='--', alpha=0.5)
        plt.title(f'{self.dataset} - Error Distribution')
        plt.xlabel('Prediction Error')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def plot_skill_weights_heatmap(self, top_n=5, save_path=None):
        """
        绘制技能权重热力图：x轴为步骤，y轴为技能，颜色为权重。
        """
        step_weights = self.load_weights_from_log()
        if not step_weights:
            print('⚠️ 没有找到技能权重日志，跳过热力图')
            return
        # 转换为 DataFrame
        rows = []
        for step, weights in step_weights.items():
            for skill, w in weights.items():
                rows.append([step, skill, w])
        df = pd.DataFrame(rows, columns=['step', 'skill', 'weight'])
        # 转换为宽表
        wide = df.pivot(index='skill', columns='step', values='weight').fillna(0)
        # 按总权重排序，显示使用频率最高的技能
        wide['total'] = wide.sum(axis=1)
        wide = wide.sort_values('total', ascending=False).drop(columns='total')
        top_skills = wide.head(top_n).index
        wide = wide.loc[top_skills]
        # 绘制热力图
        plt.figure(figsize=(14, max(4, len(top_skills))))
        sns.heatmap(wide, cmap='YlOrRd', annot=True, fmt='.2f', linewidths=0.5,
                    cbar_kws={'label': 'Weight'})
        plt.title(f'{self.dataset} - Skill Weights Over Steps (Top {top_n})')
        plt.xlabel('Step')
        plt.ylabel('Skill')
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def generate_all(self, output_dir=None):
        if output_dir is None:
            output_dir = os.path.join(self.storage, 'plots')
        os.makedirs(output_dir, exist_ok=True)
        self.plot_predictions_vs_actuals(os.path.join(output_dir, f'{self.dataset}_predictions.png'))
        self.plot_error_distribution(os.path.join(output_dir, f'{self.dataset}_error_dist.png'))
        self.plot_skill_weights_heatmap(save_path=os.path.join(output_dir, f'{self.dataset}_weights_heatmap.png'))
        print(f'✅ 图表已保存到 {output_dir}/')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='生成评估可视化图表')
    parser.add_argument('--dataset', type=str, required=True, help='数据集名称，如 airline_passengers')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录，默认为 storage/plots')
    args = parser.parse_args()
    viz = Visualizer(args.dataset)
    viz.generate_all(args.output_dir)