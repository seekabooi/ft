# experiments/autotune/inducer_candidate.py
"""
候选生成与 LLM 调用模块 - 最终正则提取版
不再依赖 JSON 解析，直接通过正则从 LLM 响应中提取策略结构
★ ★ 2026-06-27 增加技能有效性诊断 + 自动修正（naive 替代无效技能）
"""

import os
import sys
import json
import re
import time
import traceback
import hashlib
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.autotune.prompts import build_strategy_generation_prompt
from experiments.autotune.utils import compute_mase

DEBUG_FILE = os.path.join("llog", "debug_llm_responses.txt")
os.makedirs("llog", exist_ok=True)


def _format_weight(w: float) -> str:
    s = f"{w:.6f}".rstrip('0').rstrip('.')
    return s if s else "0.0"


def _extract_strategies_from_text(text: str) -> List[Dict]:
    """
    核心提取函数：使用正则从 LLM 响应中提取所有策略
    完全不依赖 JSON 解析，鲁棒性极高
    """
    strategies = []

    # 1. 尝试提取整个 candidate_strategies 数组
    array_pattern = r'"candidate_strategies"\s*:\s*\[(.*?)\]\s*\}'
    array_match = re.search(array_pattern, text, re.DOTALL)

    if array_match:
        array_content = array_match.group(1)
    else:
        # 如果没有找到完整数组，直接从文本中搜索所有策略对象
        array_content = text

    # 2. 提取每个策略对象（从 { "name": ... } 到匹配的 }
    # 使用栈匹配找到每个完整的策略对象
    i = 0
    length = len(array_content)
    while i < length:
        if array_content[i] == '{':
            start = i
            brace_count = 0
            in_string = False
            escape = False
            while i < length:
                char = array_content[i]
                if escape:
                    escape = False
                    i += 1
                    continue
                if char == '\\':
                    escape = True
                    i += 1
                    continue
                if char == '"' and not escape:
                    in_string = not in_string
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end = i
                            obj_str = array_content[start:end + 1]
                            # 解析这个策略对象
                            strategy = _parse_strategy_object(obj_str)
                            if strategy:
                                strategies.append(strategy)
                            break
                i += 1
        i += 1

    return strategies


def _parse_strategy_object(obj_str: str) -> Optional[Dict]:
    """解析单个策略对象字符串，提取 name 和 stages"""
    # 提取 name
    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', obj_str)
    if not name_match:
        return None
    name = name_match.group(1)

    # 提取 stages 数组部分
    stages_match = re.search(r'"stages"\s*:\s*\[(.*?)\]', obj_str, re.DOTALL)
    if not stages_match:
        return None
    stages_str = stages_match.group(1)

    # 提取每个 stage
    stages = []
    stage_pattern = r'\{\s*"steps"\s*:\s*(\d+)\s*,\s*"weights"\s*:\s*\{([^}]*)\}\s*\}'
    for steps_str, weights_str in re.findall(stage_pattern, stages_str, re.DOTALL):
        weights = {}
        # 提取权重键值对
        weight_pattern = r'"([^"]+)"\s*:\s*([0-9.]+)'
        for skill, val in re.findall(weight_pattern, weights_str):
            try:
                weights[skill] = float(val)
            except:
                pass
        if weights:
            stages.append({"steps": int(steps_str), "weights": weights})

    if not stages:
        return None

    # 尝试提取 description（可选）
    desc_match = re.search(r'"description"\s*:\s*"([^"]*)"', obj_str)
    description = desc_match.group(1) if desc_match else ""

    return {
        "name": name,
        "stages": stages,
        "description": description
    }


def _safe_extract_json(text: str) -> Dict:
    """
    安全提取策略，先尝试 JSON，失败则用正则直接提取
    """
    # 先尝试标准 JSON 解析（快速路径）
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_candidate = text[start:end + 1]
            data = json.loads(json_candidate, strict=False)
            if 'candidate_strategies' in data:
                return data
    except:
        pass

    # 如果 JSON 解析失败，使用正则提取
    strategies = _extract_strategies_from_text(text)
    if strategies:
        return {"candidate_strategies": strategies}

    return {}


# ★★★ 模块级函数：用于子进程并行评估 ★★★
def _predict_and_evaluate_single(train: np.ndarray, test: np.ndarray,
                                 horizon: int, period: int,
                                 mase_scale: float,
                                 strategy: Dict,
                                 skill_blacklist: set = None) -> Tuple[int, float, str]:
    try:
        from src.skills.registry import SkillRegistry
        from run_benchmark import build_full_registry
        full_registry, _ = build_full_registry()
        stages = strategy.get('stages', [])
        if not stages:
            return -1, float('inf'), "策略无 stages"
        predictions = []
        current_hist = train.copy()
        max_iter = horizon + 10
        for stage in stages:
            steps = stage.get('steps', 0)
            weights = stage.get('weights', {})
            if steps <= 0:
                continue
            for _ in range(steps):
                if len(predictions) >= max_iter:
                    break
                pred_val = 0.0
                total_w = 0.0
                for skill_name, weight in weights.items():
                    if weight <= 0:
                        continue
                    skill = full_registry.get(skill_name)
                    if skill is None:
                        continue
                    try:
                        forecast = skill.execute(current_hist, 1, period=period)
                        if forecast is not None and len(forecast) > 0:
                            pred_val += forecast[0] * weight
                            total_w += weight
                    except Exception:
                        pass
                if total_w > 0:
                    pred_val /= total_w
                else:
                    pred_val = np.mean(current_hist[-5:]) if len(current_hist) >= 5 else np.mean(current_hist)
                predictions.append(pred_val)
                current_hist = np.append(current_hist, pred_val)
        if len(predictions) < horizon:
            last_weights = stages[-1].get('weights', {})
            while len(predictions) < horizon and len(predictions) < max_iter:
                pred_val = 0.0
                total_w = 0.0
                for skill_name, weight in last_weights.items():
                    if weight <= 0:
                        continue
                    skill = full_registry.get(skill_name)
                    if skill is None:
                        continue
                    try:
                        forecast = skill.execute(current_hist, 1, period=period)
                        if forecast is not None and len(forecast) > 0:
                            pred_val += forecast[0] * weight
                            total_w += weight
                    except Exception:
                        pass
                if total_w > 0:
                    pred_val /= total_w
                else:
                    pred_val = np.mean(current_hist[-5:]) if len(current_hist) >= 5 else np.mean(current_hist)
                predictions.append(pred_val)
                current_hist = np.append(current_hist, pred_val)
        pred_array = np.array(predictions[:horizon])
        if len(pred_array) != len(test):
            return -1, float('inf'), "预测长度不匹配"
        mase = compute_mase(pred_array, test, mase_scale)
        return 0, mase, ""
    except Exception as e:
        return -1, float('inf'), str(e)


class CandidateProcessor:
    def __init__(self, config: Dict, logger):
        self.config = config
        self.logger = logger
        self._available_skills = None
        self._logged_skill_errors = set()
        self._skill_cache = {}
        filter_cfg = config.get('skill_filter', {})
        self.enable_blacklist = filter_cfg.get('enable_blacklist', True)
        self.blacklist = set(filter_cfg.get('blacklist', []))
        self.blacklist.add('progressive_adaptive_combiner')
        if self.enable_blacklist and self.blacklist:
            self.logger.log(f"   📋 黑名单已启用，过滤技能: {', '.join(self.blacklist)}")
        self.max_candidates_per_window = 2

        parallel_cfg = config.get('parallel', {})
        self.parallel_enabled = parallel_cfg.get('enabled', True)
        self.parallel_workers = parallel_cfg.get('workers', max(2, multiprocessing.cpu_count() // 2))
        self.prefetch_llm = parallel_cfg.get('prefetch_llm', True)

        self._prefetch_future = None
        self._prefetch_window_id = None
        self._collect_logs_mode = False
        self._collected_logs = []

    def _get_available_skills(self) -> List[str]:
        if self._available_skills is not None:
            return self._available_skills
        try:
            from run_benchmark import build_full_registry
            full_registry, all_skills = build_full_registry()
            self._available_skills = [skill.name for skill in all_skills if hasattr(skill, 'name')]
            self.logger.log(f"   📋 可用技能数: {len(self._available_skills)}")
            if self._available_skills:
                self.logger.log(f"      示例: {self._available_skills[:5]}")
        except Exception as e:
            self.logger.log(f"   ⚠️ 获取技能列表失败: {e}，使用默认列表")
            default_skills = ['chunk_ensemble', 'multi_resolution', 'residual_correction_advanced']
            self.logger.log(f"   ⚠️ 使用默认技能列表: {default_skills}")
            self._available_skills = default_skills
        return self._available_skills

    def _log(self, msg: str, level: str = "INFO"):
        if self._collect_logs_mode:
            self._collected_logs.append(f"[{level}] {msg}")
        else:
            self.logger.log(msg, level)

    def set_log_collection(self, enabled: bool = True):
        self._collect_logs_mode = enabled
        if enabled:
            self._collected_logs = []

    def get_collected_logs(self) -> List[str]:
        return self._collected_logs

    def clear_collected_logs(self):
        self._collected_logs = []

    # ★★★ 核心修改：增加技能有效性诊断 + 自动修正 ★★★
    def _generate_candidates(self, features: Dict, trajectory: List,
                             window_id: int, horizon: int) -> List[Dict]:
        available_skills = self._get_available_skills()
        if not available_skills:
            self._log(f"   ⚠️ 无可用技能，跳过候选生成")
            return []
        base_prompt = build_strategy_generation_prompt(features, trajectory, window_id, horizon)
        skill_list_str = ', '.join(available_skills)
        prompt = base_prompt + f"\n\n★★★★★ 可用技能列表（必须从以下名称中选择，不得使用列表外的任何名称）：\n{skill_list_str}"
        result = self._call_llm(prompt, window_id)
        candidates = result.get('candidate_strategies', [])
        if not candidates:
            self._log(f"   ⚠️ LLM未返回候选策略，生成默认策略")
            default = self._create_default_strategy(horizon, available_skills)
            return [default] if default else []

        valid_candidates = []
        for i, s in enumerate(candidates):
            try:
                if not isinstance(s, dict):
                    self._log(f"   ⚠️ 候选策略 {i + 1} 格式错误，跳过")
                    continue
                if not s.get('name'):
                    s['name'] = f"策略{chr(65 + i)}"
                stages = s.get('stages', [])
                if not stages:
                    self._log(f"   ⚠️ 候选策略 {i + 1} 无 stages，跳过")
                    continue
                filtered_stages = []
                stage_skills_map = {}  # ★ 记录每个阶段的技能映射，用于诊断

                for stage_idx, stage in enumerate(stages):
                    weights = stage.get('weights', {})

                    # ★★★ 诊断：检查哪些技能无效 ★★★
                    invalid_skills = []
                    for skill_name in weights.keys():
                        if skill_name not in available_skills:
                            invalid_skills.append(skill_name)
                        elif skill_name in self.blacklist:
                            invalid_skills.append(f"{skill_name}(黑名单)")

                    if invalid_skills:
                        self._log(f"   🔍 策略 {i + 1} 阶段 {stage_idx + 1} 发现无效技能: {invalid_skills}")

                    # ★★★ 过滤可用技能 ★★★
                    if self.enable_blacklist:
                        filtered = {k: v for k, v in weights.items()
                                    if k in available_skills and k not in self.blacklist}
                    else:
                        filtered = {k: v for k, v in weights.items() if k in available_skills}

                    # ★★★ 如果阶段中无可用技能，尝试自动修正 ★★★
                    if not filtered:
                        # 尝试将无效技能的权重转移给 naive（作为兜底）
                        if invalid_skills and 'naive' in available_skills and 'naive' not in self.blacklist:
                            naive_weight = weights.get('naive', 0.0)
                            for invalid in invalid_skills:
                                # 清理无效技能名（去掉可能的(黑名单)后缀）
                                clean_name = invalid.split('(')[0] if '(' in invalid else invalid
                                if clean_name in weights:
                                    naive_weight += weights.pop(clean_name)
                            weights['naive'] = naive_weight

                            # 重新过滤
                            if self.enable_blacklist:
                                filtered = {k: v for k, v in weights.items()
                                            if k in available_skills and k not in self.blacklist}
                            else:
                                filtered = {k: v for k, v in weights.items() if k in available_skills}

                            if filtered:
                                self._log(f"   🔧 自动修正: 策略 {i + 1} 阶段 {stage_idx + 1} 无效技能 → naive")
                            else:
                                self._log(f"   ⚠️ 候选策略 {i + 1} 阶段 {stage_idx + 1} 修正后仍无可用技能，跳过该阶段")
                                continue
                        else:
                            self._log(f"   ⚠️ 候选策略 {i + 1} 阶段 {stage_idx + 1} 中无可用技能，跳过该阶段")
                            self._log(
                                f"      ❌ 无效技能: {invalid_skills[:5]}{'...' if len(invalid_skills) > 5 else ''}")
                            continue

                    total = sum(filtered.values())
                    if total == 0:
                        uniform = 1.0 / len(filtered)
                        filtered = {k: uniform for k in filtered}
                    else:
                        filtered = {k: v / total for k, v in filtered.items()}
                    for k in filtered:
                        filtered[k] = float(f"{filtered[k]:.6f}")
                    stage['weights'] = filtered
                    filtered_stages.append(stage)

                if not filtered_stages:
                    self._log(f"   ⚠️ 候选策略 {i + 1} 所有阶段均无效，跳过")
                    continue
                s['stages'] = filtered_stages
                total_steps = sum(stage.get('steps', 0) for stage in filtered_stages)
                if total_steps != horizon:
                    self._log(f"   ⚠️ 候选策略 {i + 1} 步数总和 {total_steps} != {horizon}，将在预测时补齐")
                valid_candidates.append(s)
            except Exception as e:
                self._log(f"   ⚠️ 候选策略 {i + 1} 校验异常: {e}")
                self._log(traceback.format_exc())
                continue

        if len(valid_candidates) > self.max_candidates_per_window:
            self._log(
                f"   📌 候选数 {len(valid_candidates)} > {self.max_candidates_per_window}，只保留前{self.max_candidates_per_window}个")
            valid_candidates = valid_candidates[:self.max_candidates_per_window]
        if not valid_candidates:
            self._log(f"   ⚠️ 所有候选策略无效，生成默认策略")
            default = self._create_default_strategy(horizon, available_skills)
            if default:
                valid_candidates.append(default)

        self._print_candidates(valid_candidates)
        return valid_candidates

    def _print_candidates(self, candidates: List[Dict]):
        if not candidates:
            self._log(f"   📋 无候选策略")
            return
        self._log(f"   📋 候选策略 ({len(candidates)} 个):")
        for i, strategy in enumerate(candidates):
            try:
                name = strategy.get('name', f'候选{i + 1}')
                stages = strategy.get('stages', [])
                if not stages:
                    self._log(f"      策略 {i + 1}: {name} → (无阶段)")
                    continue
                stage_desc = []
                for j, stage in enumerate(stages):
                    steps = stage.get('steps', 0)
                    weights = stage.get('weights', {})
                    w_str = ', '.join([f"{k}:{_format_weight(v)}" for k, v in weights.items()])
                    stage_desc.append(f"{steps}步{{{w_str}}}")
                self._log(f"      策略 {i + 1}: {name} → {' → '.join(stage_desc)}")
            except Exception as e:
                self._log(f"      策略 {i + 1}: 打印异常 - {e}")

    def _parallel_evaluate(self, candidates: List[Dict], train: np.ndarray,
                           test: np.ndarray, horizon: int, period: int,
                           mase_scale: float, window_id: int) -> List[Tuple[int, float, Dict]]:
        if not candidates:
            return []
        self._log(f"   ⚡ 并行评估 {len(candidates)} 个候选策略...")
        results = []
        with ProcessPoolExecutor(max_workers=self.parallel_workers) as executor:
            future_to_idx = {}
            for idx, strategy in enumerate(candidates):
                future = executor.submit(
                    _predict_and_evaluate_single,
                    train, test, horizon, period, mase_scale,
                    strategy, self.blacklist
                )
                future_to_idx[future] = idx
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    ret_idx, mase, error = future.result(timeout=300)
                    if ret_idx == 0:
                        results.append((idx, mase, candidates[idx]))
                        self._log(f"      策略 {idx + 1}: MASE={mase:.6f}")
                    else:
                        self._log(f"      策略 {idx + 1}: 评估失败 - {error}")
                except Exception as e:
                    self._log(f"      策略 {idx + 1}: 并行评估异常 - {e}")
        results.sort(key=lambda x: x[0])
        return results

    def _create_default_strategy(self, horizon: int, available_skills: List[str]) -> Dict:
        if not available_skills:
            return None
        skills = available_skills[:3]
        if len(skills) < 2:
            skills = available_skills[:2] if len(available_skills) >= 2 else available_skills
        if not skills:
            return None
        weight = 1.0 / len(skills)
        weights = {s: float(f"{weight:.6f}") for s in skills}
        import math
        num_stages = min(4, max(2, horizon // 3))
        base = horizon // num_stages
        stages = []
        remaining = horizon
        for i in range(num_stages):
            steps = base + 1 if i < horizon % num_stages else base
            if steps <= 0:
                steps = 1
            stages.append({"steps": steps, "weights": weights.copy()})
            remaining -= steps
        if remaining > 0:
            stages[-1]["steps"] += remaining
        return {
            "name": "默认策略",
            "stages": stages,
            "description": "所有候选无效时的默认策略"
        }

    def _call_llm(self, prompt: str, window_id: int = None) -> Dict:
        self._log(f"   ⏳ 正在向 LLM 发送请求...")
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                from src.agents.llm_client import LLMClient
                llm_config = self.config.get('llm', {})
                max_tokens = llm_config.get('max_tokens', 4096)
                model_name = llm_config.get('model', 'glm-4')
                client = LLMClient(
                    model=model_name,
                    max_tokens=max_tokens,
                    verbose=True,
                    logger=self.logger
                )
                self._log(f"  📌 模型: {model_name}")
                self._log(f"  💰 额度信息: {llm_config.get('quota_info', '基础模型（可能已用完）')}")
                self._log(f"  📤 请求模型: {model_name} (API: {model_name})")
                self._log(f"  📏 Prompt长度: {len(prompt)} 字符")
                self._log(f"  📤 尝试 {attempt + 1}/2...")

                start_time = time.time()
                resp = client.call_with_retry(prompt, max_retries=1)
                elapsed = time.time() - start_time

                content = resp.choices[0].message.content
                reasoning = getattr(resp.choices[0].message, 'reasoning_content', None)
                text = content if content else reasoning

                self._log(f"  📥 响应完成 (耗时 {elapsed:.1f}s)")

                if hasattr(resp, 'usage') and resp.usage:
                    self._log(
                        f"  📊 Token: prompt={resp.usage.prompt_tokens}, completion={resp.usage.completion_tokens}, total={resp.usage.total_tokens}")

                if not text:
                    self._log(f"   ❌ LLM 返回空内容")
                    continue

                # 保存到调试文件
                try:
                    with open(DEBUG_FILE, 'a', encoding='utf-8') as f:
                        f.write("=" * 80 + "\n")
                        f.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"模型: {model_name}\n")
                        if window_id is not None:
                            f.write(f"窗口ID: {window_id}\n")
                        f.write(f"Prompt长度: {len(prompt)} 字符\n")
                        f.write(f"响应长度: {len(text)} 字符\n")
                        f.write("--- 原始响应 ---\n")
                        f.write(text + "\n")
                        f.write("=" * 80 + "\n\n")
                        f.flush()
                except Exception as e:
                    self._log(f"   ⚠️ 写入调试文件失败: {e}")

                # 使用安全提取函数（纯正则兜底）
                data = _safe_extract_json(text)
                if data and data.get('candidate_strategies'):
                    self._log(f"   ✅ 提取到 {len(data['candidate_strategies'])} 个策略")
                    return data

                # 如果仍然失败，保存原始响应
                if window_id is not None:
                    try:
                        failed_dir = os.path.join("llog", "failed_raw")
                        os.makedirs(failed_dir, exist_ok=True)
                        failed_file = os.path.join(failed_dir, f"failed_raw_w{window_id}.txt")
                        with open(failed_file, 'w', encoding='utf-8') as f:
                            f.write(text)
                        self._log(f"   📁 原始响应已保存到: {failed_file}")
                    except Exception as e:
                        self._log(f"   ⚠️ 保存失败响应失败: {e}")

                if attempt < max_retries:
                    self._log(f"   🔄 解析失败，重试 {attempt + 1}/{max_retries}...")
                    continue
                else:
                    self._log(f"   ❌ 所有解析方式均失败，返回空（将使用默认策略）")
                    return {}

            except Exception as e:
                self._log(f"   ❌ LLM调用异常: {type(e).__name__}: {e}")
                self._log(traceback.format_exc())
                if attempt < max_retries:
                    self._log(f"   🔄 重试 {attempt + 1}/{max_retries}...")
                    time.sleep(2)
                    continue
                return {}
        return {}

    def _hash_history(self, hist: np.ndarray) -> str:
        data = hist[-50:] if len(hist) >= 50 else hist
        return hashlib.md5(data.tobytes()).hexdigest()

    def _predict_with_strategy(self, train: np.ndarray, horizon: int,
                               period: int, strategy: Dict) -> Optional[np.ndarray]:
        try:
            from src.skills.registry import SkillRegistry
            from run_benchmark import build_full_registry
            full_registry, _ = build_full_registry()
            stages = strategy.get('stages', [])
            if not stages:
                self._log(f"      策略无 stages")
                return None
            for idx, stage in enumerate(stages):
                steps = stage.get('steps', 0)
                if steps <= 0:
                    self._log(f"      阶段 {idx + 1} steps={steps} 无效，跳过")
                    return None
            predictions = []
            current_hist = train.copy()
            max_iter = horizon + 10
            self._skill_cache.clear()
            for stage in stages:
                steps = stage.get('steps', 0)
                weights = stage.get('weights', {})
                for _ in range(steps):
                    if len(predictions) >= max_iter:
                        self._log(f"      预测步数达到上限 {max_iter}，强制退出")
                        break
                    pred_val = 0.0
                    total_w = 0.0
                    for skill_name, weight in weights.items():
                        if weight <= 0:
                            continue
                        skill = full_registry.get(skill_name)
                        if skill is None:
                            continue
                        hist_hash = self._hash_history(current_hist)
                        cache_key = (skill_name, hist_hash)
                        if cache_key in self._skill_cache:
                            forecast = self._skill_cache[cache_key]
                        else:
                            try:
                                forecast = skill.execute(current_hist, 1, period=period)
                                if forecast is not None and len(forecast) > 0:
                                    self._skill_cache[cache_key] = forecast
                                else:
                                    forecast = None
                            except Exception as e:
                                if skill_name not in self._logged_skill_errors:
                                    self._log(f"      技能 {skill_name} 执行异常: {e}")
                                    self._logged_skill_errors.add(skill_name)
                                forecast = None
                        if forecast is not None and len(forecast) > 0:
                            pred_val += forecast[0] * weight
                            total_w += weight
                    if total_w > 0:
                        pred_val /= total_w
                    else:
                        pred_val = np.mean(current_hist[-5:]) if len(current_hist) >= 5 else np.mean(current_hist)
                    predictions.append(pred_val)
                    current_hist = np.append(current_hist, pred_val)
            if len(predictions) < horizon:
                last_weights = stages[-1].get('weights', {})
                while len(predictions) < horizon and len(predictions) < max_iter:
                    pred_val = 0.0
                    total_w = 0.0
                    for skill_name, weight in last_weights.items():
                        if weight <= 0:
                            continue
                        skill = full_registry.get(skill_name)
                        if skill is None:
                            continue
                        hist_hash = self._hash_history(current_hist)
                        cache_key = (skill_name, hist_hash)
                        if cache_key in self._skill_cache:
                            forecast = self._skill_cache[cache_key]
                        else:
                            try:
                                forecast = skill.execute(current_hist, 1, period=period)
                                if forecast is not None and len(forecast) > 0:
                                    self._skill_cache[cache_key] = forecast
                                else:
                                    forecast = None
                            except Exception as e:
                                if skill_name not in self._logged_skill_errors:
                                    self._log(f"      技能 {skill_name} 执行异常: {e}")  # 修复：使用 _log 而不是 self._log
                                    self._logged_skill_errors.add(skill_name)
                                forecast = None
                        if forecast is not None and len(forecast) > 0:
                            pred_val += forecast[0] * weight
                            total_w += weight
                    if total_w > 0:
                        pred_val /= total_w
                    else:
                        pred_val = np.mean(current_hist[-5:]) if len(current_hist) >= 5 else np.mean(current_hist)
                    predictions.append(pred_val)
                    current_hist = np.append(current_hist, pred_val)
            return np.array(predictions[:horizon])
        except Exception as e:
            self._log(f"      _predict_with_strategy 内部异常: {type(e).__name__}: {e}")
            self._log(traceback.format_exc())
            return None