#!/usr/bin/env python
"""
缓存管理工具 - 一键清除采集缓存和/或训练检查点
用于修改 config.yaml 中 step_size/horizon/window_sizes 后重新采集

使用方法：
    python clear_cache.py              # 清除采集缓存（保留检查点）
    python clear_cache.py --all        # 清除采集缓存 + 训练检查点
    python clear_cache.py --keep-llog  # 只清除采集缓存，保留所有 llog
    python clear_cache.py --dataset melbourne_temp  # 清除指定数据集缓存
"""

import os
import sys
import shutil
import argparse
from datetime import datetime

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(PROJECT_ROOT, "storage", "autotune_results")
LLOG_DIR = os.path.join(PROJECT_ROOT, "llog")


def print_header():
    print("=" * 70)
    print("🧹 SPLS 缓存管理工具")
    print(f"   时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


def print_file_size(path):
    """获取文件/文件夹大小"""
    if os.path.isfile(path):
        size = os.path.getsize(path)
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        else:
            return f"{size / (1024 * 1024):.1f} MB"

    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    if total < 1024:
        return f"{total} B"
    elif total < 1024 * 1024:
        return f"{total / 1024:.1f} KB"
    else:
        return f"{total / (1024 * 1024):.1f} MB"


def list_caches():
    """列出所有缓存信息"""
    print("\n📋 当前缓存状态:")
    print("-" * 70)

    # 采集缓存
    csv_path = os.path.join(STORAGE_DIR, "collected_windows.csv")
    window_data_dir = os.path.join(STORAGE_DIR, "window_data")

    if os.path.exists(csv_path):
        size = print_file_size(csv_path)
        print(f"   📄 collected_windows.csv: {size}")
    else:
        print(f"   📄 collected_windows.csv: (不存在)")

    if os.path.exists(window_data_dir):
        size = print_file_size(window_data_dir)
        count = len(os.listdir(window_data_dir)) if os.path.exists(window_data_dir) else 0
        print(f"   📁 window_data/: {size} ({count} 个文件)")
    else:
        print(f"   📁 window_data/: (不存在)")

    # 训练检查点
    if os.path.exists(LLOG_DIR):
        run_dirs = [d for d in os.listdir(LLOG_DIR) if d.startswith("run_")]
        if run_dirs:
            print(f"\n   📁 训练运行历史:")
            for d in sorted(run_dirs):
                run_path = os.path.join(LLOG_DIR, d)
                size = print_file_size(run_path)
                has_checkpoint = os.path.exists(os.path.join(run_path, "checkpoint.json"))
                checkpoint_status = "✅ 有检查点" if has_checkpoint else "无检查点"
                print(f"      {d}: {size} ({checkpoint_status})")
        else:
            print(f"\n   📁 训练运行历史: (无)")
    else:
        print(f"\n   📁 llog/: (不存在)")

    print("-" * 70)


def clear_collection_cache(dataset_name: str = None):
    """清除采集缓存"""
    print("\n🗑️ 清除采集缓存...")

    csv_path = os.path.join(STORAGE_DIR, "collected_windows.csv")
    window_data_dir = os.path.join(STORAGE_DIR, "window_data")

    deleted_files = []

    # 删除 collected_windows.csv
    if os.path.exists(csv_path):
        os.remove(csv_path)
        deleted_files.append("collected_windows.csv")
        print(f"   ✅ 已删除: collected_windows.csv")
    else:
        print(f"   ⚠️ 文件不存在: collected_windows.csv")

    # 删除 window_data 目录
    if os.path.exists(window_data_dir):
        if dataset_name:
            # 删除指定数据集的文件
            deleted_count = 0
            for f in os.listdir(window_data_dir):
                if f.startswith(f"{dataset_name}_window_"):
                    os.remove(os.path.join(window_data_dir, f))
                    deleted_count += 1
            if deleted_count > 0:
                print(f"   ✅ 已删除: window_data/ 中 {deleted_count} 个 {dataset_name} 文件")
            else:
                print(f"   ⚠️ 未找到 {dataset_name} 的窗口数据文件")
        else:
            shutil.rmtree(window_data_dir)
            print(f"   ✅ 已删除: window_data/ 目录")

    # 删除缓存目录
    cache_dir = os.path.join(STORAGE_DIR, "cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"   ✅ 已删除: cache/ 目录")

    strategy_cache = os.path.join(STORAGE_DIR, "strategy_cache")
    if os.path.exists(strategy_cache):
        shutil.rmtree(strategy_cache)
        print(f"   ✅ 已删除: strategy_cache/ 目录")

    if deleted_files:
        print(f"\n   ✅ 共删除 {len(deleted_files)} 个缓存")
    else:
        print(f"\n   ℹ️ 没有需要删除的缓存")


def clear_checkpoints(keep_latest: bool = True):
    """清除训练检查点"""
    print("\n🗑️ 清除训练检查点...")

    if not os.path.exists(LLOG_DIR):
        print(f"   ⚠️ llog/ 目录不存在")
        return

    run_dirs = [d for d in os.listdir(LLOG_DIR) if d.startswith("run_")]

    if not run_dirs:
        print(f"   ℹ️ 没有训练运行记录")
        return

    if keep_latest:
        # 保留最新的运行目录
        run_dirs.sort(reverse=True)
        latest = run_dirs[0]
        run_dirs = run_dirs[1:]
        print(f"   📌 保留最新: {latest}")

    for d in run_dirs:
        run_path = os.path.join(LLOG_DIR, d)
        shutil.rmtree(run_path)
        print(f"   ✅ 已删除: {d}")

    if not keep_latest:
        print(f"   ✅ 已删除所有训练运行记录")


def clear_all(keep_latest: bool = True):
    """清除所有缓存"""
    clear_collection_cache()
    clear_checkpoints(keep_latest=keep_latest)


def main():
    parser = argparse.ArgumentParser(description="SPLS 缓存管理工具")
    parser.add_argument("--all", action="store_true", help="清除所有缓存（采集缓存 + 训练检查点）")
    parser.add_argument("--keep-llog", action="store_true", help="只清除采集缓存，保留训练检查点")
    parser.add_argument("--dataset", type=str, default=None, help="指定数据集名称（如 melbourne_temp）")
    parser.add_argument("--list", action="store_true", help="列出当前缓存状态")
    parser.add_argument("--yes", "-y", action="store_true", help="跳过确认提示")

    args = parser.parse_args()

    print_header()

    # 显示当前状态
    list_caches()

    # 默认行为：只清除采集缓存
    if not args.all and not args.keep_llog and not args.list:
        print("\n📋 默认操作: 清除采集缓存（保留训练检查点）")
        print("   提示: 使用 --all 清除所有，使用 --keep-llog 只清除采集缓存")

        if not args.yes:
            confirm = input("\n   ⚠️ 确认清除采集缓存？(y/N): ")
            if confirm.lower() != 'y':
                print("   ❌ 已取消")
                return

        clear_collection_cache(args.dataset)

        print("\n" + "=" * 70)
        print("✅ 缓存清除完成!")
        print("   现在可以重新运行: python -m experiments.autotune.main --dataset melbourne_temp --horizon 12 --verbose --compare")
        print("=" * 70)
        return

    if args.list:
        return

    if args.all:
        print("\n📋 操作: 清除所有缓存（采集缓存 + 训练检查点）")

        if not args.yes:
            confirm = input("\n   ⚠️⚠️ 确认清除所有缓存？(y/N): ")
            if confirm.lower() != 'y':
                print("   ❌ 已取消")
                return

        # 清除采集缓存
        clear_collection_cache(args.dataset)

        # 清除所有检查点（保留最新）
        clear_checkpoints(keep_latest=True)

        print("\n" + "=" * 70)
        print("✅ 所有缓存清除完成!")
        print("   现在可以重新运行: python -m experiments.autotune.main --dataset melbourne_temp --horizon 12 --verbose --compare")
        print("=" * 70)
        return

    if args.keep_llog:
        print("\n📋 操作: 只清除采集缓存（保留所有训练检查点）")

        if not args.yes:
            confirm = input("\n   ⚠️ 确认清除采集缓存？(y/N): ")
            if confirm.lower() != 'y':
                print("   ❌ 已取消")
                return

        clear_collection_cache(args.dataset)

        print("\n" + "=" * 70)
        print("✅ 采集缓存清除完成，训练检查点已保留!")
        print("   现在可以重新运行: python -m experiments.autotune.main --dataset melbourne_temp --horizon 12 --verbose --compare")
        print("=" * 70)
        return


if __name__ == "__main__":
    main()