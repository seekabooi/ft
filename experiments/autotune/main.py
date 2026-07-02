#!/usr/bin/env python
"""
SPLS 主入口 - v5 多轮训练版（A/B划分版，配置化）
★ 4轮训练：A归纳1轮 + B3/B4/B1 演化3轮
★ 第一轮使用前50%数据（A），B为后50%（分为B1-B4）
★ 演化使用B3、B4、B1（由配置文件指定顺序）
★ 测试集从每个B子集的前1/3抽取组成
★ 每窗口生成2个候选策略
★ 技能级缓存加速
★ 检查点断点续训（支持B子集索引恢复）
★ 多版本对比评估
★ 事后补丁：针对困难窗口池统一打补丁
★ 测试阶段并行加速
★ ★ 支持 --resume 参数从指定运行目录恢复训练
★ ★ 演化轮评估并行化，策略未变化时跳过评估
★ ★ ★ 2026-06-24 新增强制演化触发、测试集最终评估
★ ★ ★ ★ 2026-06-24 禁用每轮 B 子集评估，最终测试使用 10 线程
★ ★ ★ ★ ★ 2026-06-25 模块化拆分 + 全量终端日志输出到 full_output.log
★ ★ ★ ★ ★ ★ 2026-06-25 支持 PolicyGraph 作为核心数据结构
"""

import argparse
import sys
import os
from datetime import datetime

from experiments.autotune.tuner_core import SPLSAutoTuner


class Tee:
    """同时将输出写入终端和日志文件的 'tee' 类"""
    def __init__(self, filename, mode='a'):
        self.file = open(filename, mode, encoding='utf-8')
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def write(self, message):
        self.file.write(message)
        self.file.flush()
        self.stdout.write(message)
        self.stdout.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()


def get_run_dir(args):
    """根据参数确定运行目录（与 SPLSAutoTuner 中的逻辑保持一致）"""
    if args.resume:
        run_dir = args.resume
        if not os.path.exists(run_dir):
            test_path = os.path.join("llog", run_dir)
            if os.path.exists(test_path):
                run_dir = test_path
            else:
                if not run_dir.startswith("run_"):
                    run_path = os.path.join("llog", f"run_{run_dir}")
                    if os.path.exists(run_path):
                        run_dir = run_path
        return run_dir
    else:
        from experiments.autotune.utils import create_run_folder
        return create_run_folder("llog")


def main():
    parser = argparse.ArgumentParser(description="SPLS v5 多轮训练 (A/B划分版)")
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--min_train', type=int, default=None)
    parser.add_argument('--horizon', type=int, default=None)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--compare', action='store_true', help='运行公平 Ablation')
    parser.add_argument('--resume', type=str, default=None,
                        help='从指定的运行目录恢复训练（如 run_20260624_043426 或 llog/run_20260624_043426）')

    args = parser.parse_args()

    run_dir = get_run_dir(args)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir, exist_ok=True)

    full_log_path = os.path.join(run_dir, "full_output.log")
    tee = Tee(full_log_path, 'w')
    sys.stdout = tee
    sys.stderr = tee

    print(f"\n{'='*80}")
    print(f"🚀 程序启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 运行目录: {run_dir}")
    print(f"📄 全量日志文件: {full_log_path}")
    print(f"{'='*80}\n")

    try:
        tuner = SPLSAutoTuner(args.config, verbose=args.verbose, resume_dir=args.resume)
        tuner.run(args.dataset, args.min_train, args.horizon, compare=args.compare)
    except Exception as e:
        print(f"\n❌ 程序异常终止: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sys.stdout = tee.stdout
        sys.stderr = tee.stderr
        tee.close()
        print(f"\n✅ 全量日志已保存至: {full_log_path}")


if __name__ == '__main__':
    main()