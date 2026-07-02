#!/usr/bin/env python
"""
独立测试单个窗口的 LLM 响应并验证解析（与实际运行相同的逻辑）
用法: python test_single_window.py --window_id 113
"""

import os
import sys
import argparse
import json
import re
import pandas as pd
import numpy as np
from datetime import datetime

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.autotune.utils import load_window_data, extract_features
from experiments.autotune.prompts import build_strategy_generation_prompt
from experiments.autotune.inducer_candidate import _safe_extract_json, _extract_strategies_from_text
from src.agents.llm_client import LLMClient
from run_benchmark import build_full_registry


def get_window_data(window_id: int):
    csv_path = "storage/autotune_results/collected_windows.csv"
    if not os.path.exists(csv_path):
        print(f"❌ 未找到采集数据: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    row = df[df['window_id'] == window_id]
    if row.empty:
        print(f"❌ 未找到窗口 ID: {window_id}")
        return None
    return row.iloc[0].to_dict()


def main():
    parser = argparse.ArgumentParser(description="测试单个窗口的 LLM 响应并验证解析（与实际运行相同）")
    parser.add_argument('--window_id', type=int, required=True, help='窗口 ID (如 113)')
    parser.add_argument('--model', type=str, default='glm-4', help='LLM 模型名称')
    args = parser.parse_args()

    window_id = args.window_id
    model_name = args.model

    print("=" * 80)
    print(f"🔍 测试窗口 {window_id} 的 LLM 响应并验证解析（实际运行逻辑）")
    print(f"模型: {model_name}")
    print("=" * 80)

    # 获取窗口数据
    row_dict = get_window_data(window_id)
    if row_dict is None:
        return

    window_data_path = row_dict.get('window_data_path')
    if not window_data_path or not os.path.exists(window_data_path):
        print(f"❌ 窗口数据文件不存在: {window_data_path}")
        return

    wdata = load_window_data(window_data_path)
    train = wdata['train']
    test = wdata['test']
    period = wdata.get('period', 365)
    horizon = wdata.get('horizon', 12)
    mase_scale = wdata.get('mase_scale', 1.0)

    print(f"📊 窗口 {window_id}: train={len(train)}, test={len(test)}, horizon={horizon}, period={period}")

    features = extract_features(train)
    features['window_size'] = row_dict.get('window_size', 600)

    # 获取可用技能列表
    full_registry, all_skills = build_full_registry()
    available_skills = [skill.name for skill in all_skills if hasattr(skill, 'name')]
    skill_list_str = ', '.join(available_skills)

    trajectory = []
    base_prompt = build_strategy_generation_prompt(features, trajectory, window_id, horizon)
    prompt = base_prompt + f"\n\n★★★★★ 可用技能列表（必须从以下名称中选择，不得使用列表外的任何名称）：\n{skill_list_str}"

    print(f"📏 Prompt 长度: {len(prompt)} 字符")
    print("-" * 80)

    # 调用 LLM
    print(f"⏳ 正在调用 LLM ({model_name})...")
    client = LLMClient(model=model_name, verbose=True)

    try:
        resp = client.call_with_retry(prompt, max_retries=1)
        content = resp.choices[0].message.content
        reasoning = getattr(resp.choices[0].message, 'reasoning_content', None)
        raw_text = content if content else reasoning

        print("\n" + "=" * 80)
        print("📥 原始响应内容（未解析）")
        print("=" * 80)
        print(raw_text[:2000] + ("..." if len(raw_text) > 2000 else ""))
        print("=" * 80)

        # 保存原始响应
        output_dir = "llog"
        os.makedirs(output_dir, exist_ok=True)
        raw_file = os.path.join(output_dir, f"single_window_{window_id}_raw.txt")
        with open(raw_file, 'w', encoding='utf-8') as f:
            f.write(raw_text)
        print(f"\n📁 原始响应已保存到: {raw_file}")

        if hasattr(resp, 'usage') and resp.usage:
            print(f"\n📊 Token 统计:")
            print(f"  Prompt Tokens: {resp.usage.prompt_tokens}")
            print(f"  Completion Tokens: {resp.usage.completion_tokens}")
            print(f"  Total Tokens: {resp.usage.total_tokens}")

        # ★★★ 使用与 _call_llm 完全相同的解析逻辑 ★★★
        print("\n" + "=" * 80)
        print("🔧 使用 _safe_extract_json 解析（与实际运行相同）")
        print("=" * 80)

        parsed = _safe_extract_json(raw_text)

        if parsed and parsed.get('candidate_strategies'):
            strategies = parsed['candidate_strategies']
            print(f"✅ 解析成功！提取到 {len(strategies)} 个候选策略")
            # 打印第一个策略概要
            if strategies and isinstance(strategies[0], dict):
                print(f"第一个策略名称: {strategies[0].get('name', 'N/A')}")
                stages = strategies[0].get('stages', [])
                print(f"阶段数: {len(stages)}")
                if stages:
                    print(f"第一阶段 steps: {stages[0].get('steps', 0)}, weights: {stages[0].get('weights', {})}")
        else:
            print("❌ 解析失败！未提取到候选策略")
            # 如果解析失败，尝试用正则提取（兜底）
            print("\n尝试正则提取（兜底）...")
            regex_strategies = _extract_strategies_from_text(raw_text)
            if regex_strategies:
                print(f"✅ 正则提取成功，提取到 {len(regex_strategies)} 个策略")
                # 打印第一个策略
                if regex_strategies and isinstance(regex_strategies[0], dict):
                    print(f"第一个策略名称: {regex_strategies[0].get('name', 'N/A')}")
            else:
                print("❌ 正则提取也失败")

                # ★★★ 打印详细的诊断信息 ★★★
                print("\n" + "=" * 80)
                print("🔍 诊断信息（用于定位问题）")
                print("=" * 80)

                # 提取 JSON 片段（从第一个 { 到最后一个 }）
                start = raw_text.find('{')
                end = raw_text.rfind('}')
                if start != -1 and end != -1 and end > start:
                    json_fragment = raw_text[start:end+1]
                    print(f"提取的 JSON 片段长度: {len(json_fragment)}")
                    print("片段开头 500 字符:")
                    print(json_fragment[:500])
                    print("...")
                    print("片段结尾 500 字符:")
                    print(json_fragment[-500:])
                else:
                    print("未找到有效的 JSON 边界字符 { 和 }")

    except Exception as e:
        print(f"❌ LLM 调用失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()