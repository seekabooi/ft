#!/usr/bin/env python
"""
增强版断点续跑诊断与修复工具
功能：
1. 扫描 llog/ 根目录和所有 run_*/ 子目录下的断点文件
2. 检测 llog_dir 配置是否正确，提示路径错误
3. 自动修复：将误放在 llog/ 根目录的文件迁移到对应的 run_*/ 目录
4. 诊断断点续跑能力，生成详细报告
5. 单条指令完成全部检查和修复

用法：
  python -m experiments.autotune.diagnose_breakpoint --all         # 全功能（检查+修复）
  python -m experiments.autotune.diagnose_breakpoint              # 仅诊断
  python -m experiments.autotune.diagnose_breakpoint --fix-misplaced  # 修复错误位置文件
  python -m experiments.autotune.diagnose_breakpoint --run-dir <dir>  # 指定目录
"""

import os
import sys
import json
import glob
import re
import shutil
import time
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ============================================================
# 工具函数
# ============================================================

def get_all_run_dirs(base_dir: str = "llog") -> List[str]:
    run_dirs = glob.glob(os.path.join(base_dir, "run_*"))
    run_dirs.sort(key=os.path.getmtime, reverse=True)
    return run_dirs


def get_latest_run_dir(base_dir: str = "llog") -> Optional[str]:
    dirs = get_all_run_dirs(base_dir)
    return dirs[0] if dirs else None


def parse_run_timestamp(run_dir: str) -> str:
    basename = os.path.basename(run_dir)
    match = re.search(r'run_(\d{8}_\d{6})', basename)
    return match.group(1) if match else "未知"


def format_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    else:
        return f"{size / (1024 * 1024):.1f} MB"


def read_json_safe(filepath: str) -> Optional[Dict]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def write_json_safe(filepath: str, data: Dict) -> bool:
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"  ❌ 写入失败: {e}")
        return False


# ============================================================
# 核心诊断类
# ============================================================

class BreakpointDiagnostic:
    def __init__(self, run_dir: str, verbose: bool = True):
        self.run_dir = run_dir
        self.verbose = verbose
        self.results = {
            "run_dir": run_dir,
            "timestamp": parse_run_timestamp(run_dir),
            "window_results": {"exists": False, "files": [], "total": 0, "trouble": 0, "normal": 0,
                               "window_ids": [], "trouble_ids": [], "mases": []},
            "trouble_json": {"exists": False, "count": 0, "ids": []},
            "progress_json": {"exists": False, "count": 0, "ids": []},
            "refined_json": {"exists": False},
            "logs": {"exists": False, "files": []},
            "round_dirs": [],
            "issues": [],
            "can_rebuild": False,
            "resume_test": {"can_resume": False, "recovered_ids": [], "pending_ids": [], "message": ""}
        }

    def diagnose(self) -> Dict:
        print("\n" + "=" * 80)
        print(f"🔍 断点续跑诊断: {os.path.basename(self.run_dir)}")
        print(f"📁 路径: {self.run_dir}")
        print("=" * 80)

        self._check_window_results()
        self._check_trouble_json()
        self._check_progress_json()
        self._check_refined_json()
        self._check_logs()
        self._check_round_dirs()
        self._verify_consistency()
        self._test_resume_simulation()
        self._print_summary()
        return self.results

    def _check_window_results(self):
        print("\n📊 1. window_results/ 目录")
        print("-" * 50)
        window_dir = os.path.join(self.run_dir, "window_results")
        if not os.path.exists(window_dir):
            print("   ❌ 目录不存在")
            self.results["window_results"]["exists"] = False
            self.results["issues"].append("window_results/ 目录不存在")
            return

        self.results["window_results"]["exists"] = True
        json_files = glob.glob(os.path.join(window_dir, "window_*.json"))
        json_files.sort(key=lambda x: int(os.path.basename(x).replace("window_", "").replace(".json", "")))

        print(f"   ✅ 目录存在，发现 {len(json_files)} 个窗口结果文件")

        window_ids = []
        trouble_ids = []
        mase_list = []
        for fpath in json_files:
            data = read_json_safe(fpath)
            if data is None:
                continue
            wid = data.get("window_id")
            if wid is None:
                continue
            window_ids.append(wid)
            mase = data.get("best_mase", float('inf'))
            if mase != float('inf'):
                mase_list.append(mase)
            if data.get("is_trouble", False):
                trouble_ids.append(wid)

        self.results["window_results"]["files"] = json_files
        self.results["window_results"]["total"] = len(window_ids)
        self.results["window_results"]["trouble"] = len(trouble_ids)
        self.results["window_results"]["normal"] = len(window_ids) - len(trouble_ids)
        self.results["window_results"]["window_ids"] = sorted(window_ids)
        self.results["window_results"]["trouble_ids"] = sorted(trouble_ids)
        self.results["window_results"]["mases"] = mase_list

        print(f"      窗口 ID: {sorted(window_ids)}")
        print(f"      困难窗口: {len(trouble_ids)} 个 (ID: {sorted(trouble_ids)})")
        print(f"      普通窗口: {len(window_ids) - len(trouble_ids)} 个")
        if mase_list:
            avg = sum(mase_list)/len(mase_list)
            mn, mx = min(mase_list), max(mase_list)
            print(f"      MASE 统计: 平均={avg:.4f}, 最小={mn:.4f}, 最大={mx:.4f}")

    def _check_trouble_json(self):
        print("\n📊 2. trouble_windows.json")
        print("-" * 50)
        trouble_path = os.path.join(self.run_dir, "trouble_windows.json")
        if os.path.exists(trouble_path):
            data = read_json_safe(trouble_path)
            if data is not None and isinstance(data, list):
                ids = [w.get("window_id") for w in data if w.get("window_id") is not None]
                print(f"   ✅ 文件存在，包含 {len(data)} 个困难窗口")
                self.results["trouble_json"]["exists"] = True
                self.results["trouble_json"]["count"] = len(data)
                self.results["trouble_json"]["ids"] = sorted(ids)
                if ids:
                    print(f"      窗口 ID: {sorted(ids)}")
            else:
                print("   ⚠️ 文件存在但格式异常")
                self.results["trouble_json"]["exists"] = True
        else:
            print("   ❌ 文件不存在")
            self.results["trouble_json"]["exists"] = False
            self.results["issues"].append("trouble_windows.json 不存在")

        win_trouble = self.results["window_results"].get("trouble", 0)
        if win_trouble > 0 and not self.results["trouble_json"]["exists"]:
            print(f"   💡 检测到 {win_trouble} 个困难窗口，可以从 window_results/ 重建")
            self.results["can_rebuild"] = True

    def _check_progress_json(self):
        print("\n📊 3. induction_progress.json")
        print("-" * 50)
        progress_path = os.path.join(self.run_dir, "induction_progress.json")
        if os.path.exists(progress_path):
            data = read_json_safe(progress_path)
            if data is not None:
                processed = data.get("processed_ids", [])
                print(f"   ✅ 文件存在")
                print(f"      processed_ids: {len(processed)} 个窗口")
                self.results["progress_json"]["exists"] = True
                self.results["progress_json"]["count"] = len(processed)
                self.results["progress_json"]["ids"] = sorted(processed)
                if processed:
                    print(f"      窗口 ID: {sorted(processed)}")
            else:
                print("   ⚠️ 文件存在但格式异常")
                self.results["progress_json"]["exists"] = True
        else:
            print("   ❌ 文件不存在")
            self.results["progress_json"]["exists"] = False
            self.results["issues"].append("induction_progress.json 不存在")

        win_total = self.results["window_results"].get("total", 0)
        if win_total > 0 and not self.results["progress_json"]["exists"]:
            print(f"   💡 检测到 {win_total} 个窗口结果，可以从 window_results/ 重建")
            self.results["can_rebuild"] = True

    def _check_refined_json(self):
        print("\n📊 4. refined_policies.json")
        print("-" * 50)
        refined_path = os.path.join(self.run_dir, "refined_policies.json")
        if os.path.exists(refined_path):
            data = read_json_safe(refined_path)
            if data is not None:
                policies = data.get("policies", [])
                print(f"   ✅ 文件存在，包含 {len(policies)} 条策略（第1轮已完成）")
                self.results["refined_json"]["exists"] = True
            else:
                print("   ⚠️ 文件存在但格式异常")
                self.results["refined_json"]["exists"] = True
        else:
            print("   ❌ 文件不存在（第1轮尚未完成）")
            self.results["refined_json"]["exists"] = False

        for round_num in range(1, 5):
            round_dir = os.path.join(self.run_dir, f"round_{round_num}")
            if os.path.exists(round_dir):
                opt_path = os.path.join(round_dir, "refined_policies_optimized.json")
                raw_path = os.path.join(round_dir, "refined_policies_raw.json")
                if os.path.exists(opt_path) or os.path.exists(raw_path):
                    self.results["round_dirs"].append(round_num)
        if self.results["round_dirs"]:
            print(f"   📁 已完成的轮次: {self.results['round_dirs']}")

    def _check_logs(self):
        print("\n📊 5. 日志文件")
        print("-" * 50)
        log_files = glob.glob(os.path.join(self.run_dir, "spls_autotune_*.log"))
        log_files.sort(key=os.path.getmtime, reverse=True)
        if not log_files:
            print("   ❌ 未找到日志文件")
            self.results["logs"]["exists"] = False
            return
        self.results["logs"]["exists"] = True
        self.results["logs"]["files"] = log_files
        print(f"   ✅ 找到 {len(log_files)} 个日志文件:")
        for lf in log_files[:5]:
            size = os.path.getsize(lf)
            basename = os.path.basename(lf)
            print(f"      - {basename} ({format_size(size)})")
        if log_files:
            latest_log = log_files[0]
            try:
                with open(latest_log, 'r', encoding='utf-8') as f:
                    content = f.read()
                trouble_matches = re.findall(r"📌 困难窗口 (\d+)", content)
                if trouble_matches:
                    print(f"   📌 日志中标记的困难窗口: {sorted(set(map(int, trouble_matches)))}")
                if "📊 第1轮归纳完成统计" in content:
                    print("   ✅ 日志包含第1轮完成统计")
                else:
                    print("   ⚠️ 日志中未找到第1轮完成统计（可能未完成）")
                if "从独立结果文件恢复" in content:
                    print("   ✅ 日志包含断点恢复记录")
            except Exception as e:
                print(f"   ⚠️ 读取日志失败: {e}")

    def _check_round_dirs(self):
        pass

    def _verify_consistency(self):
        print("\n📊 6. 一致性验证")
        print("-" * 50)
        win_ids = set(self.results["window_results"].get("window_ids", []))
        trouble_ids = set(self.results["window_results"].get("trouble_ids", []))
        progress_ids = set(self.results["progress_json"].get("ids", []))
        trouble_json_ids = set(self.results["trouble_json"].get("ids", []))

        if win_ids and trouble_ids:
            if trouble_json_ids != trouble_ids:
                print(f"   ⚠️ 困难窗口不一致: window_results 有 {len(trouble_ids)} 个，trouble_windows.json 有 {len(trouble_json_ids)} 个")
            else:
                print(f"   ✅ 困难窗口一致: {len(trouble_ids)} 个")

        if win_ids and progress_ids:
            if win_ids == progress_ids:
                print(f"   ✅ 进度文件与窗口结果完全一致: {len(win_ids)} 个窗口")
            else:
                missing = win_ids - progress_ids
                extra = progress_ids - win_ids
                if missing:
                    print(f"   ⚠️ 进度文件缺失窗口: {sorted(missing)}")
                if extra:
                    print(f"   ⚠️ 进度文件多余窗口: {sorted(extra)}")
        elif win_ids and not progress_ids:
            print(f"   ⚠️ 有 {len(win_ids)} 个窗口结果，但进度文件不存在")

        print("\n   🔄 断点续跑评估:")
        if not self.results["window_results"]["exists"]:
            print("      ❌ 无独立结果文件，无法断点续跑")
        elif self.results["refined_json"]["exists"]:
            print("      ✅ 第1轮已完成，可以进入第2轮")
        elif not self.results["progress_json"]["exists"] and self.results["window_results"]["total"] > 0:
            print(f"      ⚠️ 有 {self.results['window_results']['total']} 个窗口结果，但进度文件缺失")
            print("      💡 运行 --write 参数重建进度文件，然后重启程序")
        else:
            print(f"      ✅ 有 {self.results['window_results']['total']} 个窗口结果，可以恢复续跑")

    def _test_resume_simulation(self):
        print("\n📊 7. 断点恢复测试 (模拟加载)")
        print("-" * 50)
        win_ids = set(self.results["window_results"].get("window_ids", []))
        progress_ids = set(self.results["progress_json"].get("ids", []))
        if not win_ids:
            print("   ❌ 没有窗口结果文件，无法测试恢复")
            self.results["resume_test"]["can_resume"] = False
            self.results["resume_test"]["message"] = "无窗口结果"
            return

        recovered = set(win_ids)  # 至少从 window_results 恢复
        if progress_ids:
            recovered.update(progress_ids)

        max_id = max(win_ids) if win_ids else 0
        total_windows = max(69, max_id)
        pending = set(range(max_id + 1, total_windows + 1)) if max_id < total_windows else set()

        recovered_ids = sorted(recovered)
        pending_ids = sorted(pending - recovered)

        can_resume = len(recovered) > 0
        if can_resume:
            print(f"   ✅ 可以恢复 {len(recovered)} 个已完成的窗口:")
            print(f"      已恢复窗口: {recovered_ids[:20]}{'...' if len(recovered_ids) > 20 else ''}")
            if pending_ids:
                print(f"      待处理窗口: {pending_ids[:20]}{'...' if len(pending_ids) > 20 else ''} (共 {len(pending_ids)} 个)")
                print(f"      下一个窗口: {min(pending_ids) if pending_ids else '无'}")
            else:
                print("      ✅ 所有窗口已完成！")
        else:
            print("   ❌ 无法恢复任何窗口，需要从头开始")

        self.results["resume_test"]["can_resume"] = can_resume
        self.results["resume_test"]["recovered_ids"] = recovered_ids
        self.results["resume_test"]["pending_ids"] = pending_ids
        self.results["resume_test"]["message"] = f"恢复 {len(recovered_ids)} 个窗口" if can_resume else "无法恢复"

        if can_resume and pending_ids:
            print("\n   📋 恢复后执行计划:")
            print(f"      - 跳过窗口: {recovered_ids[:20]}{'...' if len(recovered_ids) > 20 else ''}")
            print(f"      - 从窗口 {min(pending_ids)} 继续处理")

    def _print_summary(self):
        print("\n" + "=" * 80)
        print("📋 诊断摘要")
        print("=" * 80)
        items = [
            ("运行目录", os.path.basename(self.run_dir)),
            ("运行时间", self.results["timestamp"]),
            ("window_results/", f"{self.results['window_results']['total']} 个窗口 ({self.results['window_results']['trouble']} 个困难)"),
            ("trouble_windows.json", f"{'✅ 存在' if self.results['trouble_json']['exists'] else '❌ 不存在'} ({self.results['trouble_json']['count']} 个)"),
            ("induction_progress.json", f"{'✅ 存在' if self.results['progress_json']['exists'] else '❌ 不存在'} ({self.results['progress_json']['count']} 个)"),
            ("refined_policies.json", "✅ 存在" if self.results["refined_json"]["exists"] else "❌ 不存在（未完成）"),
            ("已完成轮次", f"{self.results['round_dirs']}" if self.results["round_dirs"] else "无"),
            ("断点恢复测试", f"{'✅ 可以通过' if self.results['resume_test']['can_resume'] else '❌ 无法恢复'} ({self.results['resume_test']['message']})"),
        ]
        for label, value in items:
            print(f"   {label:<25} {value}")

        if self.results["issues"]:
            print("\n   ⚠️ 发现的问题:")
            for issue in self.results["issues"]:
                print(f"      - {issue}")

        if self.results["can_rebuild"]:
            print("\n   💡 可以执行修复:")
            print("      python -m experiments.autotune.diagnose_breakpoint --write")
            print("      或")
            print(f"      python -m experiments.autotune.diagnose_breakpoint --run-dir {os.path.basename(self.run_dir)} --write")

        if self.results["resume_test"]["can_resume"] and self.results["resume_test"]["pending_ids"]:
            print("\n   🚀 断点续跑就绪！可以执行:")
            print("      python -m experiments.autotune.main --dataset melbourne_temp --horizon 12 --verbose --compare")
            print(f"      将从窗口 {min(self.results['resume_test']['pending_ids'])} 继续")


# ============================================================
# 增强功能：多位置扫描与修复
# ============================================================

def scan_all_locations():
    """扫描 llog/ 根目录和所有 run_*/ 子目录，查找断点文件"""
    llog_dir = "llog"
    if not os.path.exists(llog_dir):
        return {}

    results = {}
    # 扫描根目录
    root_window_results = os.path.join(llog_dir, "window_results")
    if os.path.exists(root_window_results):
        results["llog_root"] = {
            "path": root_window_results,
            "files": glob.glob(os.path.join(root_window_results, "window_*.json"))
        }

    # 扫描 run_* 子目录
    run_dirs = get_all_run_dirs()
    for rd in run_dirs:
        wd = os.path.join(rd, "window_results")
        if os.path.exists(wd):
            results[rd] = {
                "path": wd,
                "files": glob.glob(os.path.join(wd, "window_*.json"))
            }
    return results


def diagnose_path_misplacement():
    """检测是否有文件放错位置"""
    llog_dir = "llog"
    issues = []
    # 检查根目录是否存在 window_results
    root_win = os.path.join(llog_dir, "window_results")
    if os.path.exists(root_win):
        files = glob.glob(os.path.join(root_win, "window_*.json"))
        if files:
            issues.append({
                "type": "misplaced",
                "source": root_win,
                "files": files,
                "message": f"发现 {len(files)} 个窗口结果文件在 llog/ 根目录，应放在 run_*/ 子目录下"
            })
    # 检查是否存在 run_* 目录但没有 window_results
    run_dirs = get_all_run_dirs()
    if not run_dirs:
        issues.append({
            "type": "no_run_dir",
            "message": "没有找到任何 run_* 目录，可能尚未运行或运行目录被删除"
        })
    else:
        for rd in run_dirs:
            wd = os.path.join(rd, "window_results")
            if not os.path.exists(wd):
                # 可能还没有生成
                pass
    return issues


def fix_misplaced_files(dry_run=True):
    """将 llog/ 根目录下的断点文件移动到最新的 run_*/ 目录"""
    llog_dir = "llog"
    root_win = os.path.join(llog_dir, "window_results")
    if not os.path.exists(root_win):
        print("ℹ️ 没有发现 llog/window_results/，无需修复")
        return False

    run_dirs = get_all_run_dirs()
    if not run_dirs:
        print("❌ 没有找到 run_* 目录，无法确定目标位置")
        return False

    latest_run = run_dirs[0]  # 最新的
    target_dir = os.path.join(latest_run, "window_results")
    files = glob.glob(os.path.join(root_win, "*.json"))
    if not files:
        print("ℹ️ 没有需要迁移的文件")
        return True

    print(f"📦 准备将 {len(files)} 个文件从 {root_win} 迁移到 {target_dir}")
    if dry_run:
        print("   [模拟运行] 将执行以下操作：")
        for f in files:
            print(f"      mv {f} {target_dir}/")
        print("   若要实际执行，请添加 --apply-fix 参数")
        return True

    os.makedirs(target_dir, exist_ok=True)
    for f in files:
        dest = os.path.join(target_dir, os.path.basename(f))
        shutil.move(f, dest)
        print(f"   ✅ 已移动 {os.path.basename(f)}")
    # 删除空目录
    try:
        os.rmdir(root_win)
        print("   🧹 已删除空的 llog/window_results/")
    except OSError:
        pass
    return True


# ============================================================
# 重建功能
# ============================================================

def rebuild_from_window_results(run_dir: str, verbose: bool = True) -> bool:
    window_dir = os.path.join(run_dir, "window_results")
    if not os.path.exists(window_dir):
        print(f"❌ {window_dir} 不存在")
        return False

    json_files = glob.glob(os.path.join(window_dir, "window_*.json"))
    if not json_files:
        print(f"❌ 没有找到 window_*.json 文件")
        return False

    trouble_pool = []
    window_best = []
    processed_ids = []

    for fpath in json_files:
        data = read_json_safe(fpath)
        if data is None:
            continue
        wid = data.get("window_id")
        if wid is None:
            continue
        processed_ids.append(wid)
        best_strategy = data.get("best_strategy")
        best_mase = data.get("best_mase", float('inf'))
        if best_strategy is not None and best_mase != float('inf'):
            item = best_strategy.copy()
            item['_window_id'] = wid
            item['_origin'] = data.get("origin", 0)
            item['_mase'] = best_mase
            item['_features'] = data.get("features", {})
            window_best.append(item)
            if data.get("is_trouble", False):
                trouble_pool.append({
                    "window_id": wid,
                    "origin": data.get("origin", 0),
                    "window_size": data.get("window_size", 600),
                    "mase": best_mase,
                    "window_data_path": data.get("window_data_path", ""),
                    "best_strategy_name": best_strategy.get("name", "unknown"),
                    "collected_at": data.get("saved_at", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                })

    if not window_best:
        print("❌ 没有有效的窗口策略数据")
        return False

    trouble_path = os.path.join(run_dir, "trouble_windows.json")
    if write_json_safe(trouble_path, trouble_pool):
        print(f"   ✅ 已写入 trouble_windows.json ({len(trouble_pool)} 个困难窗口)")
    else:
        print(f"   ❌ 写入 {trouble_path} 失败")

    progress_path = os.path.join(run_dir, "induction_progress.json")
    progress_data = {
        "window_best": window_best,
        "processed_ids": sorted(processed_ids),
        "timestamp": time.time()
    }
    if write_json_safe(progress_path, progress_data):
        print(f"   ✅ 已写入 induction_progress.json ({len(processed_ids)} 个窗口)")
    else:
        print(f"   ❌ 写入 {progress_path} 失败")

    return True


# ============================================================
# 主函数
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="增强版断点续跑诊断与修复工具")
    parser.add_argument("--run-dir", type=str, help="指定运行目录（默认自动检测最新）")
    parser.add_argument("--write", action="store_true", help="重建缺失的元数据文件")
    parser.add_argument("--fix-misplaced", action="store_true", help="将放错位置的文件迁移到正确的运行目录")
    parser.add_argument("--apply-fix", action="store_true", help="实际执行迁移（需与 --fix-misplaced 配合）")
    parser.add_argument("--all", action="store_true", help="执行所有检查和修复（包括迁移）")
    parser.add_argument("--no-verbose", action="store_true", help="精简输出")

    args = parser.parse_args()

    if args.all:
        args.write = True
        args.fix_misplaced = True
        args.apply_fix = True

    print("=" * 80)
    print("🔍 增强版断点续跑诊断与修复工具")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 先检测位置问题
    if args.fix_misplaced:
        print("\n📦 检查文件位置...")
        issues = diagnose_path_misplacement()
        misplaced = [i for i in issues if i.get("type") == "misplaced"]
        if misplaced:
            print("⚠️ 发现文件位置错误:")
            for m in misplaced:
                print(f"   - {m['message']}")
            if args.apply_fix:
                print("\n🔄 正在迁移文件...")
                fix_misplaced_files(dry_run=False)
            else:
                print("\n💡 若要迁移，请加上 --apply-fix 参数")
                print("   python -m experiments.autotune.diagnose_breakpoint --fix-misplaced --apply-fix")
        else:
            print("✅ 文件位置正常")

    # 确定运行目录
    if args.run_dir:
        run_dir = args.run_dir
        if not os.path.exists(run_dir):
            test_path = os.path.join("llog", run_dir)
            if os.path.exists(test_path):
                run_dir = test_path
            else:
                print(f"❌ 目录不存在: {run_dir}")
                return
    else:
        run_dir = get_latest_run_dir()
        if run_dir is None:
            print("❌ 未找到任何运行目录 (llog/run_*)")
            return
        print(f"📁 自动检测到最新运行目录: {run_dir}")

    # 执行诊断
    diag = BreakpointDiagnostic(run_dir, not args.no_verbose)
    results = diag.diagnose()

    # 修复缺失的元数据
    if args.write and results["can_rebuild"]:
        print("\n" + "=" * 80)
        print("🔧 执行修复...")
        print("=" * 80)
        rebuild_from_window_results(run_dir, verbose=True)
        print("\n✅ 修复完成！")
        # 重新诊断验证
        print("\n🔄 重新诊断验证...")
        diag2 = BreakpointDiagnostic(run_dir, not args.no_verbose)
        diag2.diagnose()

    print("\n" + "=" * 80)
    print("✅ 诊断完成")
    print("=" * 80)

    if results["resume_test"]["can_resume"] and results["resume_test"]["pending_ids"]:
        print("\n🚀 断点续跑可用！执行以下命令继续训练:")
        print("   python -m experiments.autotune.main --dataset melbourne_temp --horizon 12 --verbose --compare")
        print(f"   (将从窗口 {min(results['resume_test']['pending_ids'])} 继续)")
    elif results["refined_json"]["exists"]:
        print("\n✅ 第1轮已完成，可以执行演化轮次:")
        print("   python -m experiments.autotune.main --dataset melbourne_temp --horizon 12 --verbose --compare")
    elif not results["window_results"]["exists"]:
        print("\n❌ 没有窗口结果文件，需要从头开始:")
        print("   python -m experiments.autotune.main --dataset melbourne_temp --horizon 12 --verbose --compare")
    elif results["can_rebuild"] and not args.write:
        print("\n💡 执行修复后即可恢复断点:")
        print("   python -m experiments.autotune.diagnose_breakpoint --write")


if __name__ == "__main__":
    main()