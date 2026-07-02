# experiments/autotune/inducer.py
"""
Skill Policy Induction - 统一入口
继承核心逻辑，混入候选处理
★ 重写 _process_windows 方法，支持窗口级并行评估（ProcessPoolExecutor）
★ 收集困难窗口（MASE > 1）
★ 加载历史进度时，将已处理窗口的 MASE > 1 也加入困难池
★ 并行失败时自动回退到串行，并支持重试
★ ★ 修复子进程中 logger 缺失的问题
★ ★ 完整日志收集：子进程直接写入日志文件（不依赖主进程）
★ ★ 加载进度时打印历史摘要（平均MASE、最小最大、困难窗口数）
★ ★ ★ 子进程独立保存窗口结果文件，防止主进程中断导致进度丢失
★ ★ ★ ★ 子进程 LLM 日志直接写入主日志文件，终端输出带PID，格式清晰
★ ★ ★ ★ ★ 主进程显示待处理窗口范围
★ ★ ★ ★ ★ ★ 困难窗口计入池时打印确认日志，窗口完成后打印累计统计
★ ★ ★ ★ ★ ★ ★ 2026-06-24 新增 force_regenerate 参数，Re-Induction 时强制重新生成
★ ★ ★ ★ ★ ★ ★ ★ 2026-06-25 返回 PolicyGraph，支持补丁聚类直接加入全局池
★ ★ ★ ★ ★ ★ ★ ★ ★ 2026-06-25 增加失败窗口持久化记录
★ ★ ★ ★ ★ ★ ★ ★ ★ ★ 2026-06-26 增强异常捕获与打印，减少并行线程数到 4
★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ 2026-06-26 修复 horizon 传递问题（pred=7 修复为 pred=12）
★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ 2026-06-26 修复 _call_llm_with_logging 参数问题
★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ 2026-06-27 增加技能有效性诊断日志
★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ 2026-07-01 增加缓存命中统计，帮助用户了解重用情况
★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ 2026-07-XX 困难窗口池动态管理（连续三次验证）
★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ ★ 2026-08-XX 增加策略簇分配（由外部调用）
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import traceback
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm
from experiments.autotune.inducer_core import SkillPolicyInductorCore
from experiments.autotune.inducer_candidate import CandidateProcessor
from experiments.autotune.utils import load_window_data, compute_mase, extract_features, ProgressLogger
from experiments.autotune.policy_graph import PolicyGraph


# ★★★ 哑日志器（用于 CandidateProcessor，避免序列化问题） ★★★
class DummyLogger:
    """简单的哑日志器，只记录到 stderr 或忽略"""
    def log(self, msg: str, level: str = "INFO"):
        pass


# ★★★ 模块级函数：用于子进程窗口级并行 ★★★
def _process_single_window(
    window_info: Dict,
    horizon: int,
    trouble_threshold: float,
    llog_dir: str = None,
    total_windows: int = 0
) -> Dict:
    """
    在子进程中处理单个窗口的完整流程
    返回: {
        'window_id': int,
        'origin': int,
        'window_size': int,
        'window_data_path': str,
        'best_strategy': Dict,
        'best_mase': float,
        'features': Dict,
        'is_trouble': bool,
        'logs': List[str],
        'error': str or None
    }
    """
    import sys
    import os
    import json
    import numpy as np
    import traceback
    import time

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    logs = []

    def collect_error(msg: str, level: str = "ERROR"):
        logs.append(f"[{level}] {msg}")

    def _save_window_result_file(
        window_id: int,
        origin: int,
        window_size: int,
        window_data_path: str,
        best_strategy: Dict,
        best_mase: float,
        features: Dict,
        is_trouble: bool,
        llog_dir: str
    ):
        if llog_dir is None:
            return
        try:
            results_dir = os.path.join(llog_dir, 'window_results')
            os.makedirs(results_dir, exist_ok=True)

            temp_file = os.path.join(results_dir, f'window_{window_id}.tmp')
            final_file = os.path.join(results_dir, f'window_{window_id}.json')

            result_data = {
                'window_id': window_id,
                'origin': origin,
                'window_size': window_size,
                'window_data_path': window_data_path,
                'best_strategy': best_strategy,
                'best_mase': best_mase,
                'features': features,
                'is_trouble': is_trouble,
                'saved_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }

            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())

            os.replace(temp_file, final_file)
        except Exception as e:
            logs.append(f"   ⚠️ 保存窗口结果文件失败 (window_{window_id}): {e}")

    try:
        window_id = window_info.get('window_id')
        window_data_path = window_info.get('window_data_path')
        origin = window_info.get('origin', 0)
        window_size = window_info.get('window_size', 600)
        features = window_info.get('features', {})
        trajectory = window_info.get('trajectory', [])

        sub_logger = None
        if llog_dir is not None:
            try:
                sub_logger = ProgressLogger(log_dir=llog_dir, verbose=True)
                sub_logger.start_log("spls_autotune")
            except Exception as e:
                pass

        pid = os.getpid()
        if sub_logger is not None:
            sub_logger.log("")
            sub_logger.log(f"[PID:{pid}] 🔹 窗口 {window_id}/{total_windows} (origin={origin}, size={window_size})")
        else:
            logs.append(f"[PID:{pid}] 🔹 窗口 {window_id}/{total_windows} (origin={origin}, size={window_size})")

        if not window_data_path or not os.path.exists(window_data_path):
            return {
                'window_id': window_id,
                'origin': origin,
                'window_size': window_size,
                'window_data_path': window_data_path,
                'best_strategy': None,
                'best_mase': float('inf'),
                'features': features,
                'is_trouble': False,
                'logs': logs,
                'error': f"数据路径不存在: {window_data_path}"
            }

        wdata = load_window_data(window_data_path)
        train = wdata['train']
        test = wdata['test']
        period = wdata.get('period', 365)
        mase_scale = wdata.get('mase_scale', 1.0)

        if not features:
            features = extract_features(train)
            features['window_size'] = window_size

        tmp_config = {
            'skill_filter': window_info.get('skill_filter', {}),
            'llm': window_info.get('llm_config', {}),
            'parallel': {'enabled': False}
        }

        if sub_logger is not None:
            candidate_processor = CandidateProcessor(tmp_config, sub_logger)
            candidate_processor.set_log_collection(False)
        else:
            candidate_processor = CandidateProcessor(tmp_config, DummyLogger())
            candidate_processor.set_log_collection(True)

        def _call_llm_with_logging(prompt: str, window_id: int = None) -> Dict:
            from src.agents.llm_client import LLMClient
            llm_config = tmp_config.get('llm', {})
            max_tokens = llm_config.get('max_tokens', 4096)
            model_name = llm_config.get('model', 'glm-4')

            if sub_logger is not None:
                sub_logger.log(f"   ⏳ 正在向 LLM 发送请求... (窗口 {window_id})")
                sub_logger.log(f"  📌 模型: {model_name}")
                sub_logger.log(f"  💰 额度信息: 基础模型（可能已用完）")
                sub_logger.log(f"  📤 请求模型: {model_name} (API: {model_name})")
                sub_logger.log(f"  📏 Prompt长度: {len(prompt)} 字符")
                sub_logger.log(f"  📤 尝试 1/2...")

            client = LLMClient(
                model=model_name,
                max_tokens=max_tokens,
                verbose=True,
                logger=sub_logger
            )

            try:
                start_time = time.time()
                resp = client.call_with_retry(prompt, max_retries=1)
                elapsed = time.time() - start_time

                content = resp.choices[0].message.content

                if sub_logger is not None:
                    sub_logger.log(f"  📥 响应完成 (耗时 {elapsed:.1f}s)")
                    if hasattr(resp, 'usage') and resp.usage:
                        sub_logger.log(
                            f"  📊 Token: prompt={resp.usage.prompt_tokens}, "
                            f"completion={resp.usage.completion_tokens}, "
                            f"total={resp.usage.total_tokens}"
                        )
                else:
                    logs.append(f"  📥 响应完成 (耗时 {elapsed:.1f}s)")
                    if hasattr(resp, 'usage') and resp.usage:
                        logs.append(
                            f"  📊 Token: prompt={resp.usage.prompt_tokens}, "
                            f"completion={resp.usage.completion_tokens}, "
                            f"total={resp.usage.total_tokens}"
                        )

                if not content:
                    if sub_logger is not None:
                        sub_logger.log(f"   ❌ LLM 返回空内容")
                    else:
                        logs.append(f"   ❌ LLM 返回空内容")
                    return {}

                import re
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if not match:
                    if sub_logger is not None:
                        sub_logger.log(f"   ❌ 未找到 JSON 对象")
                    else:
                        logs.append(f"   ❌ 未找到 JSON 对象")
                    return {}

                json_str = match.group()
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)

                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    if sub_logger is not None:
                        sub_logger.log(f"   ❌ JSON 解析失败: {e}")
                    else:
                        logs.append(f"   ❌ JSON 解析失败: {e}")
                    return {}

            except Exception as e:
                error_msg = f"   ❌ LLM调用异常: {type(e).__name__}: {e}"
                error_trace = traceback.format_exc()
                if sub_logger is not None:
                    sub_logger.log(error_msg)
                    sub_logger.log(error_trace)
                else:
                    logs.append(error_msg)
                    logs.append(error_trace)
                return {}

        candidate_processor._call_llm = _call_llm_with_logging

        available_skills = candidate_processor._get_available_skills()
        if not available_skills:
            return {
                'window_id': window_id,
                'origin': origin,
                'window_size': window_size,
                'window_data_path': window_data_path,
                'best_strategy': None,
                'best_mase': float('inf'),
                'features': features,
                'is_trouble': False,
                'logs': logs,
                'error': "无可用技能"
            }

        if sub_logger is not None:
            sub_logger.log(f"   📋 可用技能: {len(available_skills)} 个")

        candidates = candidate_processor._generate_candidates(
            features, trajectory, window_id, horizon
        )

        if not candidates:
            return {
                'window_id': window_id,
                'origin': origin,
                'window_size': window_size,
                'window_data_path': window_data_path,
                'best_strategy': None,
                'best_mase': float('inf'),
                'features': features,
                'is_trouble': False,
                'logs': logs,
                'error': "无候选策略"
            }

        best = None
        best_mase = float('inf')

        from tqdm import tqdm
        pbar = tqdm(
            total=len(candidates),
            desc=f"   🔍 窗口 {window_id} 评估策略",
            unit="策略",
            ncols=110,
            position=0,
            leave=False
        )

        for idx, strategy in enumerate(candidates):
            try:
                if sub_logger is not None:
                    sub_logger.log(f"       开始评估策略 {idx + 1}...")
                start_eval = time.time()

                try:
                    pred = candidate_processor._predict_with_strategy(
                        train, horizon, period, strategy
                    )
                    eval_elapsed = time.time() - start_eval
                except Exception as pred_error:
                    eval_elapsed = time.time() - start_eval
                    error_msg = f"       策略 {idx + 1}: _predict_with_strategy 异常 - {type(pred_error).__name__}: {pred_error}"
                    error_trace = traceback.format_exc()
                    if sub_logger is not None:
                        sub_logger.log(error_msg)
                        sub_logger.log(error_trace)
                    else:
                        logs.append(error_msg)
                        logs.append(error_trace)
                    pbar.set_postfix({
                        '当前': f'{idx + 1}/{len(candidates)}',
                        '状态': f'❌ {type(pred_error).__name__}'
                    })
                    pbar.update(1)
                    continue

                if pred is not None and len(pred) == len(test):
                    mase = compute_mase(pred, test, mase_scale)
                    if sub_logger is not None:
                        sub_logger.log(f"       策略 {idx + 1} 预测耗时: {eval_elapsed:.2f}s")
                        sub_logger.log(f"       策略 {idx + 1}: MASE={mase:.6f}")
                    else:
                        logs.append(f"       策略 {idx + 1} 预测耗时: {eval_elapsed:.2f}s")
                        logs.append(f"       策略 {idx + 1}: MASE={mase:.6f}")
                    if mase < best_mase:
                        best_mase = mase
                        best = strategy
                    pbar.set_postfix({
                        '当前': f'{idx + 1}/{len(candidates)}',
                        'MASE': f'{mase:.4f}'
                    })
                else:
                    if pred is None:
                        err_detail = "预测返回 None"
                    else:
                        err_detail = f"长度不匹配 (pred={len(pred)}, test={len(test)})"
                    if sub_logger is not None:
                        sub_logger.log(f"       策略 {idx + 1}: {err_detail}")
                    else:
                        logs.append(f"       策略 {idx + 1}: {err_detail}")
                    pbar.set_postfix({
                        '当前': f'{idx + 1}/{len(candidates)}',
                        '状态': f'❌ {err_detail}'
                    })
            except Exception as e:
                error_msg = f"       策略 {idx + 1}: 评估异常 - {type(e).__name__}: {e}"
                error_trace = traceback.format_exc()
                if sub_logger is not None:
                    sub_logger.log(error_msg)
                    sub_logger.log(error_trace)
                else:
                    logs.append(error_msg)
                    logs.append(error_trace)
                pbar.set_postfix({
                    '当前': f'{idx + 1}/{len(candidates)}',
                    '状态': f'⚠️ {type(e).__name__}'
                })
            finally:
                pbar.update(1)

        pbar.close()

        if best is None:
            return {
                'window_id': window_id,
                'origin': origin,
                'window_size': window_size,
                'window_data_path': window_data_path,
                'best_strategy': None,
                'best_mase': float('inf'),
                'features': features,
                'is_trouble': False,
                'logs': logs,
                'error': "无有效策略（所有候选评估失败）"
            }

        is_trouble = best_mase > trouble_threshold

        if sub_logger is not None:
            sub_logger.log(f"   🏆 窗口 {window_id} 最优策略: {best.get('name', '未知')}, MASE={best_mase:.6f}")
            if is_trouble:
                sub_logger.log(f"   📌 困难窗口 {window_id} (MASE={best_mase:.4f} > {trouble_threshold})")
        else:
            logs.append(f"   🏆 窗口 {window_id} 最优策略: {best.get('name', '未知')}, MASE={best_mase:.6f}")
            if is_trouble:
                logs.append(f"   📌 困难窗口 {window_id} (MASE={best_mase:.4f} > {trouble_threshold})")

        _save_window_result_file(
            window_id=window_id,
            origin=origin,
            window_size=window_size,
            window_data_path=window_data_path,
            best_strategy=best,
            best_mase=best_mase,
            features=features,
            is_trouble=is_trouble,
            llog_dir=llog_dir
        )

        return {
            'window_id': window_id,
            'origin': origin,
            'window_size': window_size,
            'window_data_path': window_data_path,
            'best_strategy': best,
            'best_mase': best_mase,
            'features': features,
            'is_trouble': is_trouble,
            'logs': logs,
            'error': None
        }

    except Exception as e:
        error_msg = f"   ❌ 子进程异常: {type(e).__name__}: {e}"
        error_trace = traceback.format_exc()
        logs.append(error_msg)
        logs.append(error_trace)
        import sys as _sys
        print(f"\n{'='*60}", file=_sys.stderr)
        print(f"❌ 子进程 (PID:{os.getpid()}) 崩溃:", file=_sys.stderr)
        print(error_msg, file=_sys.stderr)
        print(error_trace, file=_sys.stderr)
        print(f"{'='*60}\n", file=_sys.stderr)
        return {
            'window_id': window_info.get('window_id', -1),
            'origin': window_info.get('origin', 0),
            'window_size': window_info.get('window_size', 600),
            'window_data_path': window_info.get('window_data_path', ''),
            'best_strategy': None,
            'best_mase': float('inf'),
            'features': {},
            'is_trouble': False,
            'logs': logs,
            'error': f"子进程异常: {type(e).__name__}: {e}"
        }


class SkillPolicyInductor(SkillPolicyInductorCore):
    """整合核心与候选处理，并支持窗口级并行评估"""

    def __init__(self, config: Dict, logger):
        super().__init__(config, logger)
        self.candidate = CandidateProcessor(config, logger)
        self._get_available_skills = self.candidate._get_available_skills
        self._generate_candidates = self.candidate._generate_candidates
        self._call_llm = self.candidate._call_llm
        self._predict_with_strategy = self.candidate._predict_with_strategy
        self._parallel_evaluate = self.candidate._parallel_evaluate

        trouble_cfg = config.get('trouble_patch', {})
        self.trouble_mase_threshold = trouble_cfg.get('mase_threshold', 1.0)
        self.trouble_pool = self._load_trouble_pool()

        parallel_cfg = config.get('parallel', {})
        self.window_parallel = parallel_cfg.get('window_parallel', True)

        configured_workers = parallel_cfg.get('window_workers', 4)
        self.window_workers = max(4, configured_workers)

        self.window_retry_delay = parallel_cfg.get('window_retry_delay', 5)
        self.window_max_retries = parallel_cfg.get('window_max_retries', 2)

        self.llog_dir = config.get('llog_dir', 'llog')
        self.window_results_dir = os.path.join(self.llog_dir, 'window_results')

        self._processed_count = 0
        self._trouble_count = 0

        self._failed_windows_path = os.path.join(self.llog_dir, 'failed_windows.json')

        self.logger.log(f"   📌 窗口级并行线程数: {self.window_workers} (配置值: {configured_workers})")

    def induce(self, collected_data: pd.DataFrame, force_regenerate: bool = False) -> Dict:
        """归纳策略（主入口）

        Args:
            collected_data: 窗口数据
            force_regenerate: 是否强制重新生成（不读缓存），Re-Induction 时传入 True

        Returns:
            Dict: {
                'policies': List[Dict],
                'policy_graph': Dict or None   # ★ 新增
            }
        """
        self.logger.log("\n" + "=" * 70)
        self.logger.log("🧠 Skill Policy Induction v5 (策略生命周期管理版)")
        self.logger.log("=" * 70)

        if collected_data.empty:
            return {'policies': [], 'policy_graph': None}

        self.logger.log(f"📊 数据量: {len(collected_data)} 个窗口")

        window_best = self._process_windows(collected_data, force_regenerate=force_regenerate)

        if not window_best:
            self.logger.log("⚠️ 没有有效的策略，返回空")
            return {'policies': [], 'policy_graph': None}

        self.logger.log("\n" + "=" * 70)
        self.logger.log("🔬 Policy Space 分割（动态聚类）")
        self.logger.log("=" * 70)

        n_strategies = len(window_best)
        n_clusters = min(4, max(3, n_strategies // 3))
        n_clusters = max(3, min(n_clusters, 6))

        try:
            from experiments.autotune.cluster import PolicySpacePartitioner
        except ImportError as e:
            self.logger.log(f"   ❌ PolicySpacePartitioner 导入失败: {e}")
            return {'policies': [], 'policy_graph': None}

        partitioner = PolicySpacePartitioner(self.logger)
        partitions = partitioner.partition(window_best, n_clusters=n_clusters)
        partitions = self._ensure_min_partition_size(partitions, min_size=2)

        policies = self._generate_policies_with_distinct_conditions(partitions, is_reinduction=force_regenerate)

        policies = self._ensure_diversity(policies)
        policies = self._ensure_policy_count(policies)

        for policy in policies:
            if len(policy.feature_groups) < 2:
                self._supplement_feature_groups(policy)

        self.logger.log(f"\n✅ 归纳完成: {len(policies)} 条策略")
        self.logger.log("\n" + "=" * 70)
        self.logger.log("📋 策略详情:")
        self.logger.log("=" * 70)

        for i, policy in enumerate(policies):
            self._print_policy_detail(i + 1, policy)

        # ★★★ 构建 PolicyGraph（复用第一轮的聚类逻辑） ★★★
        policy_graph = PolicyGraph.from_policies(policies, self.config)
        for cluster in policy_graph.clusters:
            for pid in cluster.policies:
                for p in policies:
                    if p.policy_id == pid:
                        p.cluster_id = cluster.id
                        break

        self.logger.log(f"\n📊 构建 PolicyGraph 完成: {len(policy_graph.clusters)} 个簇")

        if not force_regenerate:
            progress_file = os.path.join(self.config.get('llog_dir', 'llog'), 'induction_progress.json')
            if os.path.exists(progress_file):
                os.remove(progress_file)
                self.logger.log("   🧹 已清理进度缓存")

        return {
            'policies': [p.to_dict() for p in policies],
            'policy_graph': policy_graph.to_dict() if policy_graph else None
        }

    # ---- 以下方法保持不变 ----
    def _load_trouble_pool(self) -> List[Dict]:
        pool_file = os.path.join(self.config.get('llog_dir', 'llog'), 'trouble_windows.json')
        pool = []
        if os.path.exists(pool_file):
            try:
                with open(pool_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                for item in loaded:
                    mase = item.get('mase', 1.0)
                    if mase > 1.0:
                        if 'mase_history' not in item or not isinstance(item['mase_history'], list):
                            item['mase_history'] = [mase, mase]
                        pool.append(item)
                if len(loaded) != len(pool):
                    self.logger.log(f"   🧹 加载池时自动清理 {len(loaded) - len(pool)} 个已解决窗口")
            except Exception as e:
                self.logger.log(f"   ⚠️ 加载困难池失败: {e}")
                return []
        return pool

    def _save_trouble_pool(self):
        pool_file = os.path.join(self.config.get('llog_dir', 'llog'), 'trouble_windows.json')
        try:
            seen = set()
            unique_pool = []
            for item in self.trouble_pool:
                wid = item.get('window_id')
                if wid not in seen:
                    seen.add(wid)
                    cleaned_item = {}
                    for k, v in item.items():
                        if isinstance(v, np.integer):
                            cleaned_item[k] = int(v)
                        elif isinstance(v, np.floating):
                            cleaned_item[k] = float(v)
                        elif isinstance(v, np.ndarray):
                            cleaned_item[k] = v.tolist()
                        else:
                            cleaned_item[k] = v
                    unique_pool.append(cleaned_item)
            with open(pool_file, 'w', encoding='utf-8') as f:
                json.dump(unique_pool, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.log(f"   ⚠️ 保存困难窗口池失败: {e}")

    def _remove_from_pool(self, window_id: int) -> bool:
        initial_len = len(self.trouble_pool)
        self.trouble_pool = [w for w in self.trouble_pool if w.get('window_id') != window_id]
        if len(self.trouble_pool) < initial_len:
            self._save_trouble_pool()
            return True
        return False

    def _update_trouble_window_mase(self, window_id: int, mase: float) -> bool:
        for item in self.trouble_pool:
            if item.get('window_id') == window_id:
                item['mase'] = mase
                item['last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
                self._save_trouble_pool()
                return True
        return False

    def _collect_trouble_window(self, window_id: int, mase: float,
                                window_data_path: str, origin: int,
                                window_size: int, best_strategy: Dict):
        if mase > 1.0:
            existing = [w for w in self.trouble_pool if w.get('window_id') == window_id]

            if not existing:
                self.trouble_pool.append({
                    'window_id': window_id,
                    'origin': origin,
                    'window_size': window_size,
                    'mase': mase,
                    'mase_history': [mase],
                    'window_data_path': window_data_path,
                    'best_strategy_name': best_strategy.get('name', 'unknown'),
                    'collected_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
                })
                self._save_trouble_pool()
                self.logger.log(
                    f"      ✅ 新困难窗口 {window_id} 已加入困难池 (MASE={mase:.4f} > 1.0)"
                )
                self._trouble_count += 1
            else:
                item = existing[0]
                item['mase'] = mase
                item['mase_history'] = []
                item['last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
                self._save_trouble_pool()
                self.logger.log(
                    f"      ⚠️ 困难窗口 {window_id} 仍 > 1.0 (MASE={mase:.4f})，历史已重置"
                )
            return

        existing = [w for w in self.trouble_pool if w.get('window_id') == window_id]

        if not existing:
            self.logger.log(
                f"      ℹ️ 窗口 {window_id} 已解决 (MASE={mase:.4f} <= 1.0)，不在困难池中"
            )
            return

        item = existing[0]
        history = item.get('mase_history', [])
        history.append(mase)
        if len(history) > 3:
            history.pop(0)
        item['mase_history'] = history
        item['mase'] = mase
        item['last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
        self._save_trouble_pool()

        if len(history) == 2:
            if history[-1] < 1.0 and history[-1] == min(history):
                self._remove_from_pool(window_id)
                self.logger.log(
                    f"      ✅ 窗口 {window_id} 已确认解决 (连续2次历史 {history}，"
                    f"最终 MASE={history[-1]:.4f} 为最小值且 < 1.0)，从困难池移除"
                )
            else:
                item['mase_history'] = []
                self._save_trouble_pool()
                self.logger.log(
                    f"      ⚠️ 窗口 {window_id} 三次历史 {history} 不满足条件 "
                    f"(最新={history[-1]:.4f}，最小={min(history):.4f})，历史已重置"
                )
        else:
            self.logger.log(
                f"      📈 窗口 {window_id} 当前 MASE={mase:.4f}，历史长度 {len(history)}/3"
            )

    def get_trouble_pool(self) -> List[Dict]:
        return self.trouble_pool

    def clear_trouble_pool(self):
        self.trouble_pool = []
        pool_file = os.path.join(self.config.get('llog_dir', 'llog'), 'trouble_windows.json')
        if os.path.exists(pool_file):
            os.remove(pool_file)

    def _load_window_results_from_files(self, data: pd.DataFrame) -> Tuple[List[Dict], set]:
        recovered_best = []
        recovered_ids = set()

        if not os.path.exists(self.window_results_dir):
            return recovered_best, recovered_ids

        try:
            for filename in os.listdir(self.window_results_dir):
                if not filename.startswith('window_') or not filename.endswith('.json'):
                    continue

                filepath = os.path.join(self.window_results_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)

                    window_id = result_data.get('window_id')
                    if window_id is None:
                        continue

                    best_strategy = result_data.get('best_strategy')
                    best_mase = result_data.get('best_mase', float('inf'))
                    origin = result_data.get('origin', 0)
                    window_size = result_data.get('window_size', 600)
                    features = result_data.get('features', {})
                    is_trouble = result_data.get('is_trouble', False)

                    if best_strategy is None or best_mase == float('inf'):
                        continue

                    best_strategy['_window_id'] = window_id
                    best_strategy['_origin'] = origin
                    best_strategy['_mase'] = best_mase
                    best_strategy['_features'] = features

                    recovered_best.append(best_strategy)
                    recovered_ids.add(window_id)

                    if is_trouble:
                        row = data[data['window_id'] == window_id]
                        if not row.empty:
                            window_data_path = row.iloc[0].get('window_data_path', '')
                            if window_data_path:
                                self.logger.log(f"   📌 从独立文件恢复困难窗口 {window_id} (MASE={best_mase:.4f})")
                                self._collect_trouble_window(
                                    window_id, best_mase, window_data_path,
                                    origin, window_size, best_strategy
                                )

                except Exception as e:
                    self.logger.log(f"   ⚠️ 读取窗口结果文件失败 {filename}: {e}")
                    continue

        except Exception as e:
            self.logger.log(f"   ⚠️ 扫描窗口结果目录失败: {e}")

        return recovered_best, recovered_ids

    def _save_failed_windows(self, failed_windows: List[Dict]):
        if not failed_windows:
            return

        failed_records = []
        for fw in failed_windows:
            failed_records.append({
                'window_id': fw.get('window_id'),
                'origin': fw.get('origin', 0),
                'window_size': fw.get('window_size', 600),
                'error': fw.get('error', 'unknown'),
                'retry_count': self.window_max_retries,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'phase': 'inducer'
            })

        existing = []
        if os.path.exists(self._failed_windows_path):
            try:
                with open(self._failed_windows_path, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            except:
                pass

        existing_ids = {r['window_id'] for r in existing}
        for r in failed_records:
            if r['window_id'] not in existing_ids:
                existing.append(r)
                existing_ids.add(r['window_id'])

        with open(self._failed_windows_path, 'w', encoding='utf-8') as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

        self.logger.log(f"📁 失败窗口记录已保存: {self._failed_windows_path} ({len(existing)} 个)")

    def _process_windows(self, data: pd.DataFrame, force_regenerate: bool = False) -> List[Dict]:
        total_windows = len(data)
        progress_file = os.path.join(self.config.get('llog_dir', 'llog'), 'induction_progress.json')
        window_best = []
        processed_window_ids = set()
        failed_windows = []

        self._processed_count = 0
        self._trouble_count = len(self.trouble_pool)

        cache_hit_count = 0
        cache_miss_count = 0

        if force_regenerate:
            self.logger.log("   🔄 强制重新生成模式（忽略缓存，基于当前窗口数据生成新策略）")
            if os.path.exists(progress_file):
                os.remove(progress_file)
                self.logger.log("   🧹 已清理进度缓存")
            window_best = []
            processed_window_ids = set()
        else:
            if os.path.exists(progress_file):
                try:
                    with open(progress_file, 'r', encoding='utf-8') as f:
                        progress = json.load(f)
                    window_best = progress.get('window_best', [])
                    processed_window_ids = set(progress.get('processed_ids', []))
                    self.logger.log(f"📂 加载进度缓存: 已处理 {len(processed_window_ids)} 个窗口")
                    if window_best:
                        self.logger.log(f"   已有 {len(window_best)} 个有效策略")

                    if window_best:
                        trouble_count = 0
                        mases = []
                        for item in window_best:
                            m = item.get('_mase', float('inf'))
                            if m != float('inf'):
                                mases.append(m)
                            if m > self.trouble_mase_threshold:
                                trouble_count += 1
                        if mases:
                            self.logger.log(f"   📊 历史窗口摘要:")
                            self.logger.log(f"      平均MASE: {np.mean(mases):.4f}")
                            self.logger.log(f"      最小MASE: {np.min(mases):.4f}")
                            self.logger.log(f"      最大MASE: {np.max(mases):.4f}")
                            self.logger.log(f"      困难窗口数: {trouble_count} (MASE > {self.trouble_mase_threshold})")

                    for item in window_best:
                        mase = item.get('_mase', float('inf'))
                        if mase > self.trouble_mase_threshold:
                            wid = item.get('_window_id')
                            origin = item.get('_origin', 0)
                            window_size = item.get('_window_size', 600)
                            window_data_path = None
                            if wid is not None:
                                row = data[data['window_id'] == wid]
                                if not row.empty:
                                    window_data_path = row.iloc[0].get('window_data_path')
                            if window_data_path is None:
                                continue
                            strategy_name = item.get('name', 'unknown')
                            self._collect_trouble_window(
                                wid, mase, window_data_path, origin, window_size, {'name': strategy_name}
                            )
                except Exception as e:
                    self.logger.log(f"   ⚠️ 加载进度缓存失败: {e}，从头开始")
                    window_best = []
                    processed_window_ids = set()

        if not force_regenerate:
            recovered_best, recovered_ids = self._load_window_results_from_files(data)

            if recovered_best:
                cache_hit_count += len(recovered_best)
                self.logger.log(f"   📂 从独立结果文件恢复 {len(recovered_best)} 个窗口，将跳过这些窗口的 LLM 调用和评估。")

            newly_recovered = []
            for item in recovered_best:
                wid = item.get('_window_id')
                if wid is not None and wid not in processed_window_ids:
                    window_best.append(item)
                    processed_window_ids.add(wid)
                    newly_recovered.append(wid)

            if newly_recovered:
                self.logger.log(f"   ✅ 从独立结果文件恢复 {len(newly_recovered)} 个窗口: {sorted(newly_recovered)}")
                self.logger.log(f"   📊 总进度: 已处理 {len(processed_window_ids)}/{total_windows} 个窗口")
                self._save_progress(progress_file, window_best, processed_window_ids)

        self.logger.log(f"   📊 全局困难池: {len(self.trouble_pool)} 个窗口")

        self.logger.log("\n" + "=" * 70)
        self.logger.log("📋 逐窗口策略生成与评估 (窗口级并行版)")
        self.logger.log("=" * 70)

        horizon = data.iloc[0].get('horizon', 12) if len(data) > 0 else 12
        self.logger.log(f"   📌 预测步数 (horizon): {horizon}")

        pending_windows = []
        for idx, row in data.iterrows():
            window_id = row.get('window_id', idx + 1)
            if window_id in processed_window_ids:
                continue
            window_data_path = row.get('window_data_path', '')
            if not window_data_path or not os.path.exists(window_data_path):
                self.logger.log(f"⚠️ 窗口 {window_id} 数据路径不存在，跳过")
                processed_window_ids.add(window_id)
                continue
            pending_windows.append({
                'window_id': window_id,
                'origin': row.get('origin', 0),
                'window_size': row.get('window_size', 600),
                'window_data_path': window_data_path,
                'features': self._extract_features(row),
                'trajectory': self._get_trajectory(row),
                'skill_filter': self.config.get('skill_filter', {}),
                'llm_config': self.config.get('llm', {}),
                'horizon': horizon
            })

        if not pending_windows:
            self.logger.log("✅ 所有窗口已处理完成")
            self._save_failed_windows(failed_windows)
            self.logger.log(f"   📊 缓存统计: 命中 {cache_hit_count} 个窗口 (复用), 未命中 {cache_miss_count} 个窗口 (新处理)")
            return window_best

        pending_ids = sorted([w['window_id'] for w in pending_windows])
        self.logger.log(f"📊 待处理窗口: {pending_ids[0]}-{pending_ids[-1]} (共 {len(pending_windows)} 个)")
        self.logger.log(f"📊 待处理窗口数: {len(pending_windows)}")

        if self.window_parallel and len(pending_windows) > 1:
            results, parallel_failed = self._process_windows_parallel(
                pending_windows, horizon, data, total_windows
            )
            failed_windows.extend(parallel_failed)
        else:
            results = self._process_windows_serial(pending_windows, horizon, data, total_windows)
            for res in results:
                if res.get('error') is not None:
                    failed_windows.append(res)

        for result in results:
            if result.get('error') is not None:
                self.logger.log(f"   ⚠️ 窗口 {result.get('window_id')} 处理失败: {result.get('error')}")
                continue

            wid = result.get('window_id')
            best = result.get('best_strategy')
            best_mase = result.get('best_mase', float('inf'))
            origin = result.get('origin', 0)
            window_size = result.get('window_size', 600)
            window_data_path = result.get('window_data_path', '')
            features = result.get('features', {})
            is_trouble = result.get('is_trouble', False)

            if result.get('logs'):
                for log_msg in result['logs']:
                    pid = os.getpid()
                    self.logger.log(f"[PID:{pid}] {log_msg}")

            if best is None or best_mase == float('inf'):
                self.logger.log(f"   ⚠️ 窗口 {wid} 无有效策略，跳过")
                processed_window_ids.add(wid)
                self._save_progress(progress_file, window_best, processed_window_ids)
                cache_miss_count += 1
                continue

            cache_miss_count += 1

            if is_trouble:
                self._collect_trouble_window(
                    wid, best_mase, window_data_path,
                    origin, window_size, best
                )

            best['_window_id'] = wid
            best['_origin'] = origin
            best['_mase'] = best_mase
            best['_features'] = features
            window_best.append(best)
            processed_window_ids.add(wid)

            self._processed_count += 1

            self._save_progress(progress_file, window_best, processed_window_ids)

        if failed_windows:
            self.logger.log(f"\n   🔄 检测到 {len(failed_windows)} 个失败窗口，准备重试...")
            retry_failed = []
            for retry_attempt in range(self.window_max_retries):
                if not failed_windows:
                    break
                self.logger.log(f"   🔄 重试第 {retry_attempt + 1}/{self.window_max_retries} 次...")
                time.sleep(self.window_retry_delay)

                remaining_windows = []
                for fw in failed_windows:
                    orig = next((w for w in pending_windows if w.get('window_id') == fw.get('window_id')), None)
                    if orig:
                        remaining_windows.append(orig)

                if not remaining_windows:
                    break

                retry_results = self._process_windows_serial(remaining_windows, horizon, data, total_windows)

                still_failed = []
                for result in retry_results:
                    if result.get('error') is not None:
                        self.logger.log(f"      ⚠️ 重试窗口 {result.get('window_id')} 仍失败: {result.get('error')}")
                        still_failed.append(result)
                        continue

                    wid = result.get('window_id')
                    best = result.get('best_strategy')
                    best_mase = result.get('best_mase', float('inf'))
                    origin = result.get('origin', 0)
                    window_size = result.get('window_size', 600)
                    window_data_path = result.get('window_data_path', '')
                    features = result.get('features', {})
                    is_trouble = result.get('is_trouble', False)

                    if result.get('logs'):
                        for log_msg in result['logs']:
                            pid = os.getpid()
                            self.logger.log(f"[PID:{pid}] {log_msg}")

                    if best is None or best_mase == float('inf'):
                        self.logger.log(f"      ⚠️ 窗口 {wid} 仍无有效策略，跳过")
                        processed_window_ids.add(wid)
                        self._save_progress(progress_file, window_best, processed_window_ids)
                        continue

                    self.logger.log(f"      ✅ 窗口 {wid} 重试成功! MASE={best_mase:.6f}")
                    if is_trouble:
                        self._collect_trouble_window(
                            wid, best_mase, window_data_path,
                            origin, window_size, best
                        )
                    best['_window_id'] = wid
                    best['_origin'] = origin
                    best['_mase'] = best_mase
                    best['_features'] = features
                    window_best.append(best)
                    processed_window_ids.add(wid)
                    self._processed_count += 1
                    self._save_progress(progress_file, window_best, processed_window_ids)

                failed_windows = still_failed

            if failed_windows:
                self.logger.log(f"   ⚠️ {len(failed_windows)} 个窗口在 {self.window_max_retries} 次重试后仍失败，强制跳过")
                for fw in failed_windows:
                    wid = fw.get('window_id')
                    processed_window_ids.add(wid)
                    if 'error' not in fw:
                        fw['error'] = '重试失败'
                self._save_progress(progress_file, window_best, processed_window_ids)
                self._save_failed_windows(failed_windows)

        self.logger.log(f"\n📊 第1轮归纳完成统计:")
        self.logger.log(f"   已处理窗口: {len(processed_window_ids)}/{total_windows}")
        self.logger.log(f"   有效策略数: {len(window_best)}")
        self.logger.log(f"   困难窗口数: {len(self.trouble_pool)} (MASE > {self.trouble_mase_threshold})")
        self.logger.log(f"   📊 缓存统计: 命中 {cache_hit_count} 个窗口 (复用), 未命中 {cache_miss_count} 个窗口 (新处理)")

        self.logger.log(f"\n📊 共收集 {len(window_best)} 个窗口的best策略")
        return window_best

    def _process_windows_parallel(
            self,
            pending_windows: List[Dict],
            horizon: int,
            data: pd.DataFrame,
            total_windows: int
    ) -> Tuple[List[Dict], List[Dict]]:
        self.logger.log(f"   ⚡ 窗口级并行处理 {len(pending_windows)} 个窗口 (workers={self.window_workers})...")

        llog_dir = self.llog_dir

        results = []
        failed_windows = []

        try:
            with ProcessPoolExecutor(max_workers=self.window_workers) as executor:
                futures = {}
                for w in pending_windows:
                    future = executor.submit(
                        _process_single_window,
                        w,
                        horizon,
                        self.trouble_mase_threshold,
                        llog_dir,
                        total_windows
                    )
                    futures[future] = w.get('window_id')

                for future in as_completed(futures):
                    wid = futures[future]
                    try:
                        result = future.result(timeout=600)
                        results.append(result)
                        if result.get('error') is not None:
                            self.logger.log(f"      ⚠️ 窗口 {wid} 并行处理失败，加入重试队列")
                            if 'origin' not in result:
                                orig = next((w.get('origin', 0) for w in pending_windows if w.get('window_id') == wid), 0)
                                result['origin'] = orig
                            if 'window_size' not in result:
                                sz = next((w.get('window_size', 600) for w in pending_windows if w.get('window_id') == wid), 600)
                                result['window_size'] = sz
                            failed_windows.append(result)
                        else:
                            self.logger.log(f"      ✅ 窗口 {wid} 并行处理完成")
                    except Exception as e:
                        self.logger.log(f"      ❌ 窗口 {wid} 并行处理异常: {e}")
                        orig = next((w.get('origin', 0) for w in pending_windows if w.get('window_id') == wid), 0)
                        sz = next((w.get('window_size', 600) for w in pending_windows if w.get('window_id') == wid), 600)
                        failed_windows.append({
                            'window_id': wid,
                            'origin': orig,
                            'window_size': sz,
                            'error': f"并行超时或异常: {e}"
                        })

        except Exception as e:
            self.logger.log(f"   ❌ 窗口级并行整体失败: {e}，回退到串行模式...")
            serial_results = self._process_windows_serial(pending_windows, horizon, data, total_windows)
            for res in serial_results:
                if res.get('error') is not None:
                    failed_windows.append(res)
            return serial_results, failed_windows

        return results, failed_windows

    def _process_windows_serial(
            self,
            pending_windows: List[Dict],
            horizon: int,
            data: pd.DataFrame,
            total_windows: int
    ) -> List[Dict]:
        self.logger.log(f"   🔄 串行处理 {len(pending_windows)} 个窗口...")

        llog_dir = self.llog_dir
        results = []

        for w in pending_windows:
            result = _process_single_window(
                w,
                horizon,
                self.trouble_mase_threshold,
                llog_dir,
                total_windows
            )
            results.append(result)
            if result.get('error') is not None:
                self.logger.log(f"      ⚠️ 窗口 {w.get('window_id')} 处理失败")
            else:
                self.logger.log(f"      ✅ 窗口 {w.get('window_id')} 处理完成")

        return results