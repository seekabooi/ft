#!/usr/bin/env python
"""
策略生成器（阶段1）：对测试集窗口调用 LLM 生成策略，保存到文件
★ 新增三种直接执行模式（rl_top10_semantic_best, rl_top30_semantic_best, rl_top60_semantic_best）
  这些模式不调用 LLM，而是根据 θ 分位数和语义匹配直接选择已有策略。

功能：
1. 对指定测试集（前25或后25个窗口）的每种模式：
   - 若模式为 LLM 生成型（no_rule, semantic_top1, semantic_top30_theta_max, semantic_top60_theta_max）：
       调用 LLM 生成策略，保存到文件。
   - 若模式为直接执行型（rl_top10_semantic_best, rl_top30_semantic_best, rl_top60_semantic_best）：
       计算候选策略（按 θ 分位数筛选）并选语义匹配最高的策略，保存其 policy_id 到文件。
2. 支持断点续跑（检查已生成的窗口，跳过已完成）。

用法：
    python -m experiments.autotune.generate_strategies \
        --resume llog/cs2 \
        --round 57 \
        --half-mode first \
        --workers 8

输出：
    llog/cs2_half/generated_strategies/
        no_rule_strategies.json
        semantic_top1_strategies.json
        semantic_top30_theta_max_strategies.json
        semantic_top60_theta_max_strategies.json
        rl_top10_semantic_best_strategies.json      # 新模式
        rl_top30_semantic_best_strategies.json      # 新模式
        rl_top60_semantic_best_strategies.json      # 新模式
        generation_status.json          # 记录哪些窗口已生成
"""

import os
import sys
import json
import time
import threading
import concurrent.futures
from datetime import datetime
from typing import Dict, List, Optional, Any
import re

import pandas as pd
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.autotune.utils import (
    load_config, load_window_data, extract_features
)
from experiments.autotune.skill_policy import SkillPolicy
from experiments.autotune.state_encoder import StateEncoder
from experiments.autotune.prompts import build_strategy_generation_prompt
from experiments.autotune.inducer_candidate import _safe_extract_json, _extract_strategies_from_text
from src.agents.llm_client import LLMClient
from src.agents.llm_planner import LLMPlannerAgent
from run_benchmark import build_full_registry

# ★★★ 全局信号量：控制并发 LLM 请求数（仅用于 LLM 生成模式） ★★★
_LLM_SEMAPHORE = threading.Semaphore(6)


class StrategyGenerator:
    """策略生成器 - 阶段1，支持直接执行模式"""

    def __init__(self, run_dir: str, round_num: int, half_mode: str = 'first',
                 config_path: str = None, workers: int = 8):

        self.run_dir = run_dir
        self.round_num = round_num
        self.half_mode = half_mode
        self.config = load_config(config_path)
        self.output_dir = self.config.get('output_dir', 'storage/autotune_results')
        self.test_workers = workers
        self._timeout_counter = 0
        self._lock = threading.Lock()

        # 输出目录
        self.run_dir_out = run_dir + "_half"
        self.strategies_dir = os.path.join(self.run_dir_out, "generated_strategies")
        os.makedirs(self.strategies_dir, exist_ok=True)

        print("   🔧 构建技能注册表...")
        self.full_registry, self.all_skills = build_full_registry()
        self.skill_names = [s.name for s in self.all_skills]
        blacklist = set(self.config.get('skill_filter', {}).get('blacklist', []))
        self.skill_names = [s for s in self.skill_names if s not in blacklist]
        print(f"   ✅ 可用技能数: {len(self.skill_names)}")

        self.state_encoder = StateEncoder(self.config)
        self.model = self._detect_model()
        self.test_df = self._load_test_df()
        self.policies = self._load_round_policies(round_num)
        self._llm_client_cache = {}

        self._formatter_agent = LLMPlannerAgent(
            model=self.model if self.model else "glm-4",
            skill_registry=self.full_registry,
            verbose=False,
            use_skills=True
        )

        # ★★★ 所有模式定义 ★★★
        # LLM 生成型模式
        self.llm_modes = [
            'no_rule',
            'semantic_top1',
            'semantic_top30_theta_max',
            'semantic_top60_theta_max',
        ]
        # 直接执行型模式（不使用 LLM）
        self.direct_modes = [
            'rl_top10_semantic_best',
            'rl_top30_semantic_best',
            'rl_top60_semantic_best',
        ]
        self.modes = self.llm_modes + self.direct_modes

        # 加载已有进度
        self._status = self._load_status()

        # 加载已有策略
        self._strategies = self._load_existing_strategies()

        print(f"\n📊 策略生成器初始化完成")
        print(f"   📁 输出目录: {self.strategies_dir}")
        print(f"   📋 模式: {self.modes}")
        print(f"   📊 窗口数: {len(self.test_df)}")
        print(f"   ⚡ 并发数: {self.test_workers}")
        print(f"   🔄 断点续跑: 已生成 {self._count_generated()} 个策略")

    def _log(self, msg: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {msg}")

    def _load_test_df(self) -> pd.DataFrame:
        csv_path = os.path.join(self.output_dir, "collected_windows.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ 未找到采集数据: {csv_path}")

        df = pd.read_csv(csv_path)

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
            test_size = int(n_b * 0.5)
            test_df = b_df.iloc[:test_size].copy()

        test_df = test_df.sort_values('window_id').reset_index(drop=True)

        if self.half_mode == 'first':
            test_df = test_df.head(25).copy()
        elif self.half_mode == 'second':
            test_df = test_df.tail(25).copy()
        else:
            raise ValueError(f"half_mode 必须为 'first' 或 'second'，得到 {self.half_mode}")

        print(f"📊 使用测试集 {'前' if self.half_mode=='first' else '后'} 25 个窗口（共 {len(test_df)} 个）")
        return test_df

    def _load_round_policies(self, round_num: int) -> List[SkillPolicy]:
        path = os.path.join(self.run_dir, f"round_{round_num}", "refined_policies_optimized.json")
        if not os.path.exists(path):
            path = os.path.join(self.run_dir, f"round_{round_num}", "refined_policies_raw.json")
        if not os.path.exists(path):
            print(f"❌ 未找到第 {round_num} 轮策略文件")
            return []

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            policies = [SkillPolicy.from_dict(p) for p in data.get('policies', [])]
            print(f"📋 加载第 {round_num} 轮策略: {len(policies)} 条")
            return policies
        except Exception as e:
            print(f"⚠️ 加载失败: {e}")
            return []

    def _detect_model(self) -> Optional[str]:
        from src.agents.llm_client import LLMClient
        models = ["glm-4"]
        for model in models:
            try:
                client = LLMClient(model=model, verbose=False)
                resp = client.call_with_retry("请回复'OK'", max_retries=1)
                if resp and resp.choices and resp.choices[0].message.content:
                    print(f"   ✅ 使用模型: {model}")
                    return model
            except:
                continue
        models = ["glm-4.5-air", "glm-4.7"]
        for model in models:
            try:
                client = LLMClient(model=model, verbose=False)
                resp = client.call_with_retry("请回复'OK'", max_retries=1)
                if resp and resp.choices and resp.choices[0].message.content:
                    print(f"   ✅ 使用模型: {model}")
                    return model
            except:
                continue
        print("   ⚠️ 无可用模型，将使用均值回退（仅 LLM 生成模式受影响）")
        return None

    def _get_llm_client(self, thread_id: int) -> LLMClient:
        if thread_id not in self._llm_client_cache:
            self._llm_client_cache[thread_id] = LLMClient(
                model=self.model if self.model else "glm-4",
                verbose=False
            )
        return self._llm_client_cache[thread_id]

    def _format_reference_strategy(self, policy: SkillPolicy) -> str:
        if policy is None:
            return "无"
        return self._formatter_agent._format_strategy(policy.skill_strategy)

    def _load_status(self) -> Dict:
        status_file = os.path.join(self.strategies_dir, "generation_status.json")
        if os.path.exists(status_file):
            try:
                with open(status_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {"generated": {}}

    def _save_status(self):
        status_file = os.path.join(self.strategies_dir, "generation_status.json")
        with open(status_file, 'w', encoding='utf-8') as f:
            json.dump(self._status, f, ensure_ascii=False, indent=2)

    def _load_existing_strategies(self) -> Dict:
        strategies = {}
        for mode in self.modes:
            mode_file = os.path.join(self.strategies_dir, f"{mode}_strategies.json")
            if os.path.exists(mode_file):
                try:
                    with open(mode_file, 'r', encoding='utf-8') as f:
                        strategies[mode] = json.load(f)
                    self._log(f"   📂 加载已有策略: {mode} ({len(strategies[mode])} 个)")
                except:
                    strategies[mode] = {}
            else:
                strategies[mode] = {}
        return strategies

    def _save_mode_strategies(self, mode: str):
        mode_file = os.path.join(self.strategies_dir, f"{mode}_strategies.json")
        with open(mode_file, 'w', encoding='utf-8') as f:
            json.dump(self._strategies.get(mode, {}), f, ensure_ascii=False, indent=2)

    def _count_generated(self) -> int:
        count = 0
        for mode in self.modes:
            count += len(self._strategies.get(mode, {}))
        return count

    def _generate_strategy_for_window(self, mode: str, window_id: int,
                                      features: Dict, horizon: int,
                                      thread_id: int = 0) -> Optional[Dict]:
        """
        为单个窗口生成策略。
        对于 LLM 模式，调用 LLM 生成新策略。
        对于直接执行模式，根据 θ 分位数和语义匹配选择已有策略，并保存其 policy_id。
        """
        # ----- 直接执行模式：不调用 LLM -----
        if mode in self.direct_modes:
            # 确定 θ 分位数阈值
            if mode == 'rl_top10_semantic_best':
                percentile = 0.10
            elif mode == 'rl_top30_semantic_best':
                percentile = 0.30
            elif mode == 'rl_top60_semantic_best':
                percentile = 0.60
            else:
                return None

            # 过滤掉非 ACTIVE 和 TRIAL 策略（或只保留 ACTIVE？根据需求，一般只使用 ACTIVE，但也可包含 TRIAL）
            # 这里我们使用所有状态非 ARCHIVE/DELETE 的策略
            active_policies = [p for p in self.policies if p.status not in ['ARCHIVE', 'DELETE']]
            if not active_policies:
                return None

            # 按 θ 降序排序
            sorted_by_theta = sorted(active_policies, key=lambda p: p.logit_weight, reverse=True)
            k = max(1, int(len(sorted_by_theta) * percentile))
            theta_top = sorted_by_theta[:k]

            # 在这些策略中计算语义分数，取最高者
            best_policy = None
            best_score = -1
            for policy in theta_top:
                score = policy.compute_applicability_score(features)
                if score > best_score:
                    best_score = score
                    best_policy = policy

            if best_policy is None:
                return None

            # 返回直接执行型策略信息
            return {
                "direct_policy_id": best_policy.policy_id,
                "direct_policy_name": best_policy.name,
                "semantic_score": best_score,
                "theta": best_policy.logit_weight
            }

        # ----- LLM 生成模式 -----
        # 构建基础 Prompt
        base_prompt = build_strategy_generation_prompt(
            features=features,
            trajectory=[],
            window_id=window_id,
            horizon=horizon
        )

        prompt = base_prompt

        # 技能列表
        skill_list_str = ', '.join(self.skill_names)
        prompt += f"\n\n★★★★★ 可用技能列表（必须从以下名称中选择，不得使用列表外的任何名称）：\n{skill_list_str}\n"
        prompt += "\n⚠️ 请只生成一个候选策略（candidate_strategies 数组只包含一个对象）。\n"

        ref_policy = None
        ref_score = 0.0

        # 根据模式选择参考策略
        if mode != 'no_rule':
            scored = []
            for policy in self.policies:
                if policy.status in ['ARCHIVE', 'DELETE']:
                    continue
                score = policy.compute_applicability_score(features)
                scored.append((policy, score))

            if scored:
                scored.sort(key=lambda x: x[1], reverse=True)

                if mode == 'semantic_top1':
                    k = 1
                elif mode == 'semantic_top30_theta_max':
                    k = max(1, int(len(scored) * 0.30))
                elif mode == 'semantic_top60_theta_max':
                    k = max(1, int(len(scored) * 0.60))
                else:
                    k = 1

                candidate_pool = scored[:k]
                if candidate_pool:
                    ref_policy = max(candidate_pool, key=lambda x: x[0].logit_weight)[0]
                    ref_score = max(candidate_pool, key=lambda x: x[0].logit_weight)[1]

        # ★★★ 增强提示词：强烈要求以特征为首要依据，参考策略仅作背景了解 ★★★
        if ref_policy is not None:
            ref_desc = self._format_reference_strategy(ref_policy)
            # 加入醒目警告
            warning = (
                "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "🚨 【核心原则】请以下列【窗口特征】为首要决策依据！\n"
                "   参考策略仅供理解问题背景，【切勿照搬其结构】。\n"
                "   你必须基于当前窗口的特征值独立设计全新的策略组合。\n"
                "   若参考策略与特征不匹配，请果断抛弃，以特征为准。\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            )
            prompt += warning
            prompt += f"\n📌 参考策略（仅供参考，请勿复制）：\n{ref_desc}\n"
            prompt += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"

        # 调用 LLM（受信号量控制）
        with _LLM_SEMAPHORE:
            client = self._get_llm_client(thread_id)
            try:
                self._log(f"      ⏳ [窗口 {window_id}] 向 LLM 发送请求...")
                resp = client.call_with_retry(prompt, max_retries=2)
                content = resp.choices[0].message.content
                self._log(f"      ✅ [窗口 {window_id}] LLM 响应完成 (长度: {len(content)})")

                data = _safe_extract_json(content)
                strategies = data.get('candidate_strategies', [])
                if strategies and len(strategies) > 0:
                    strategy = strategies[0]
                    if strategy.get('stages'):
                        self._log(f"      📋 [窗口 {window_id}] 解析到策略: {strategy.get('name', '未命名')} (阶段数: {len(strategy['stages'])})")
                        return strategy

                strategies = _extract_strategies_from_text(content)
                if strategies and len(strategies) > 0:
                    self._log(f"      📋 [窗口 {window_id}] 正则解析到策略: {strategies[0].get('name', '未命名')}")
                    return strategies[0]

                self._log(f"      ⚠️ [窗口 {window_id}] 解析策略失败")
                return None
            except Exception as e:
                self._log(f"      ❌ [窗口 {window_id}] LLM 策略生成失败: {e}")
                return None

    def generate_all(self):
        """生成所有窗口的策略"""
        self._log("\n" + "=" * 80)
        self._log("🚀 阶段1：策略生成（调用 LLM 或直接选择策略）")
        self._log(f"📁 输出目录: {self.strategies_dir}")
        self._log("=" * 80)

        # 收集所有任务
        tasks = []
        for _, row in self.test_df.iterrows():
            window_id = row.get('window_id')
            window_data_path = row.get('window_data_path')
            if not window_data_path or not os.path.exists(window_data_path):
                continue
            try:
                wdata = load_window_data(window_data_path)
                train = wdata['train']
                horizon = wdata.get('horizon', 7)
                features = extract_features(train)

                for mode in self.modes:
                    # 检查是否已生成
                    if mode in self._strategies and str(window_id) in self._strategies[mode]:
                        continue
                    tasks.append({
                        'mode': mode,
                        'window_id': window_id,
                        'features': features,
                        'horizon': horizon,
                        'window_data_path': window_data_path
                    })
            except Exception as e:
                self._log(f"   ⚠️ 窗口 {window_id} 加载失败: {e}")

        if not tasks:
            self._log("✅ 所有策略已生成，无需额外工作")
            return

        self._log(f"\n📊 待生成策略: {len(tasks)} 个")

        # 并行处理（注意：直接执行模式不调用 LLM，但我们也使用线程池）
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.test_workers) as executor:
            futures = {}
            for task in tasks:
                future = executor.submit(
                    self._generate_strategy_for_window,
                    task['mode'],
                    task['window_id'],
                    task['features'],
                    task['horizon'],
                    task['window_id'] % self.test_workers
                )
                futures[future] = task

            pbar = tqdm(total=len(futures), desc="生成策略", unit="个", ncols=100)

            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                mode = task['mode']
                window_id = task['window_id']

                try:
                    strategy = future.result(timeout=120)
                    if strategy is not None:
                        # 保存策略
                        if mode not in self._strategies:
                            self._strategies[mode] = {}
                        self._strategies[mode][str(window_id)] = strategy
                        # 立即保存到文件
                        self._save_mode_strategies(mode)
                        # 更新状态
                        if 'generated' not in self._status:
                            self._status['generated'] = {}
                        self._status['generated'][f"{mode}_{window_id}"] = time.time()
                        self._save_status()
                    else:
                        pbar.set_postfix({'失败': f'窗口{window_id}'})
                except concurrent.futures.TimeoutError:
                    pbar.set_postfix({'超时': f'窗口{window_id}'})
                except Exception as e:
                    pbar.set_postfix({'异常': f'{type(e).__name__}'})

                pbar.update(1)
                total = self._count_generated()
                pbar.set_postfix({'已生成': total})

            pbar.close()

        self._log(f"\n✅ 策略生成完成！共生成 {self._count_generated()} 个策略")
        self._log(f"📁 策略保存位置: {self.strategies_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="策略生成器（阶段1）")
    parser.add_argument('--resume', type=str, required=True,
                        help='原始运行目录（如 llog/cs2）')
    parser.add_argument('--round', type=int, required=True,
                        help='指定轮次（如 57）')
    parser.add_argument('--half-mode', type=str, default='first', choices=['first', 'second'],
                        help='选择测试集前25个窗口(first)还是后25个窗口(second)')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径')
    parser.add_argument('--workers', type=int, default=8,
                        help='并行线程数（默认 8）')

    args = parser.parse_args()

    if os.path.exists(args.resume):
        run_dir = args.resume
    else:
        run_dir = os.path.join("llog", args.resume)

    if not os.path.exists(run_dir):
        print(f"❌ 目录不存在: {run_dir}")
        return

    generator = StrategyGenerator(
        run_dir=run_dir,
        round_num=args.round,
        half_mode=args.half_mode,
        config_path=args.config,
        workers=args.workers
    )
    generator.generate_all()


if __name__ == '__main__':
    main()