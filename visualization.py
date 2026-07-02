import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def plot_prediction_vs_actual(csv_path, dataset_name=None):
    """
    绘制预测值与真实值的误差对比图
    包含：预测曲线、真实曲线、误差柱状图
    """
    if not os.path.exists(csv_path):
        print(f"❌ 文件未找到: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    if 'prediction' not in df.columns or 'actual' not in df.columns:
        print("❌ CSV 必须包含 'prediction' 和 'actual' 列")
        return

    preds = df['prediction'].values
    actuals = df['actual'].values
    errors = preds - actuals
    abs_errors = np.abs(errors)

    # 计算误差指标
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    mape = np.mean(np.abs((errors) / (actuals + 1e-6))) * 100
    bias = np.mean(errors)
    max_error = np.max(abs_errors)
    min_error = np.min(abs_errors)

    # 创建画布
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                   gridspec_kw={'height_ratios': [2, 1]})

    time_steps = range(1, len(preds) + 1)

    # ==================== 上图：预测 vs 真实（带误差区间） ====================
    ax1.plot(time_steps, actuals, 'o-', label='真实值 (Actual)',
             color='#2E86C1', linewidth=2.5, markersize=8)
    ax1.plot(time_steps, preds, 's--', label='预测值 (Prediction)',
             color='#E74C3C', linewidth=2, markersize=8)

    # 填充误差区间（预测与真实之间的区域）
    ax1.fill_between(time_steps, preds, actuals,
                     where=(preds >= actuals),
                     color='#2ECC71', alpha=0.3, label='高估区域 (预测 > 真实)')
    ax1.fill_between(time_steps, preds, actuals,
                     where=(preds < actuals),
                     color='#E74C3C', alpha=0.3, label='低估区域 (预测 < 真实)')

    # 标注每个点的误差值
    for i, (t, p, a, e) in enumerate(zip(time_steps, preds, actuals, errors)):
        offset = 5 if e >= 0 else -5
        color = '#2ECC71' if e >= 0 else '#E74C3C'
        ax1.annotate(f'{e:+.2f}',
                     xy=(t, p),
                     xytext=(t, p + offset if p + offset > a else p - offset),
                     fontsize=8, color=color, ha='center', alpha=0.7,
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6))

    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax1.set_xlabel('预测步数 (Horizon)')
    ax1.set_ylabel('数值')
    ax1.set_title(f'📊 预测值与真实值对比 - {dataset_name or "预测结果"}', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # 在右上角添加统计信息
    stats_text = (
        f'MAE  = {mae:.4f}\n'
        f'RMSE = {rmse:.4f}\n'
        f'MAPE = {mape:.2f}%\n'
        f'Bias = {bias:+.4f}\n'
        f'最大误差 = {max_error:.4f}\n'
        f'最小误差 = {min_error:.4f}'
    )
    ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ==================== 下图：误差柱状图 ====================
    colors = ['#2ECC71' if e >= 0 else '#E74C3C' for e in errors]
    bars = ax2.bar(time_steps, errors, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
    ax2.axhline(y=np.mean(errors), color='red', linestyle='-.',
                linewidth=1.5, label=f'平均偏差 = {bias:+.4f}')

    # 在柱子上方标注误差值
    for bar, e in zip(bars, errors):
        height = bar.get_height()
        ax2.annotate(f'{e:+.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3 if height >= 0 else -3),
                     textcoords='offset points',
                     ha='center', va='bottom' if height >= 0 else 'top',
                     fontsize=9, color='black', alpha=0.7)

    ax2.set_xlabel('预测步数 (Horizon)')
    ax2.set_ylabel('误差 (预测 - 真实)')
    ax2.set_title(f'📉 逐步误差分布 (正:高估 / 负:低估)', fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # 保存图片
    os.makedirs('storage/plots', exist_ok=True)
    save_path = f'storage/plots/error_compare_{dataset_name or "result"}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 误差对比图已保存至: {save_path}")

    # 打印误差统计摘要
    print("\n" + "=" * 60)
    print(f"📊 误差统计摘要 - {dataset_name or '预测结果'}")
    print("=" * 60)
    print(f"  MAE  (平均绝对误差): {mae:.4f}")
    print(f"  RMSE (均方根误差):   {rmse:.4f}")
    print(f"  MAPE (平均绝对百分比误差): {mape:.2f}%")
    print(f"  Bias (平均偏差):      {bias:+.4f} ({'高估' if bias > 0 else '低估'})")
    print(f"  最大绝对误差:        {max_error:.4f} (步 {np.argmax(abs_errors) + 1})")
    print(f"  最小绝对误差:        {min_error:.4f} (步 {np.argmin(abs_errors) + 1})")

    # 分析偏差模式
    high_count = np.sum(errors > 0)
    low_count = np.sum(errors < 0)
    print(f"  高估次数: {high_count} / 低估次数: {low_count}")
    if high_count > low_count * 1.5:
        print("  ⚠️ 模型存在系统性高估倾向")
    elif low_count > high_count * 1.5:
        print("  ⚠️ 模型存在系统性低估倾向")
    else:
        print("  ✅ 误差正负平衡，无明显系统性偏差")

    print("=" * 60)

    plt.show()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='melbourne_temp',
                        help='数据集名称（自动读取 storage/eval_{dataset}.csv）')
    parser.add_argument('--csv', type=str, default=None,
                        help='手动指定CSV路径（优先级高于 --dataset）')
    args = parser.parse_args()

    if args.csv:
        csv_path = args.csv
        name = os.path.splitext(os.path.basename(csv_path))[0]
    else:
        csv_path = f'storage/eval_{args.dataset}.csv'
        name = args.dataset

    plot_prediction_vs_actual(csv_path, dataset_name=name)