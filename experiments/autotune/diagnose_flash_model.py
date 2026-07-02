#!/usr/bin/env python
"""
测试 GLM-4.7-Flash 模型是否可用于本项目
★ 轻量测试，不消耗大量 token
★ 测试基础连通性、核心预测响应、JSON解析
★ 给出明确结论
"""

import sys
import os
import json
import re
import time
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("=" * 80)
print("🔍 测试模型: GLM-4.7-Flash")
print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# ============================================================
# 测试模型列表
# ============================================================
MODELS_TO_TEST = [
    ("glm-4.7-flash", "FLASH 轻量版，Agentic Coding 强化"),
    ("glm-4.7", "赠送额度 4,196,175 tokens（对比基准）"),
    ("glm-4.5-air", "付费额度 9.6M + 赠送 12M（对比基准）"),
]

# ============================================================
# 测试函数
# ============================================================

def test_model(model_name: str, description: str) -> dict:
    """测试单个模型"""
    print(f"\n{'=' * 80}")
    print(f"📌 测试模型: {model_name}")
    print(f"   {description}")
    print(f"{'=' * 80}")

    result = {
        "model": model_name,
        "description": description,
        "available": False,
        "response_time": None,
        "error": None,
        "token_usage": {},
        "json_valid": False,
        "keys_found": [],
        "keys_missing": [],
        "overall": "❌ 不可用"
    }

    try:
        from src.agents.llm_client import LLMClient
        from src.agents.llm_prompts import build_prompt

        # ============================================================
        # 测试1：基础连通性
        # ============================================================
        print(f"\n   🔗 测试1: 基础连通性...")
        client = LLMClient(model=model_name, verbose=False)

        try:
            start_time = time.time()
            simple_resp = client.call_with_retry("请回复'OK'，只输出OK两个字，不要其他内容", max_retries=2)
            elapsed = time.time() - start_time
            result["response_time"] = elapsed
            result["status_code"] = 200

            simple_content = simple_resp.choices[0].message.content.strip()
            print(f"      ✅ 响应成功 (耗时: {elapsed:.2f}s)")
            print(f"      📝 响应内容: {simple_content[:50]}...")

            if simple_content == "OK":
                print(f"      ✅ 基础连通性正常")
            else:
                print(f"      ⚠️ 响应内容不是预期的'OK'，但模型可用")

            # Token统计
            if hasattr(simple_resp, 'usage') and simple_resp.usage:
                print(f"      📊 Token: prompt={simple_resp.usage.prompt_tokens}, completion={simple_resp.usage.completion_tokens}")

        except Exception as e:
            result["error"] = str(e)
            print(f"      ❌ 基础连通性失败: {e}")
            return result

        # ============================================================
        # 测试2：核心预测响应模拟
        # ============================================================
        print(f"\n   🔗 测试2: 核心预测响应模拟...")

        # 模拟数据
        mock_features = {
            "trend_strength": 0.0806,
            "seasonal_strength": 0.0793,
            "adf_pvalue": 0.2166,
            "data_length": 600.0,
            "skewness": 0.2862,
            "cv": 0.4100,
        }
        mock_history = [11.5, 11.8, 12.1, 11.9, 12.3, 12.6, 12.4, 12.8, 13.1, 12.9]

        from src.skills.data_profiler import DataProfiler

        # 构建真实prompt
        profile = {
            'seasonal_strength': 0.0793,
            'trend_strength': 0.0806,
            'period': 7,
            'adf_pvalue': 0.2166,
            'data_length': 600,
            'has_dates': False,
            'missing_rate': 0.0,
            'recent_volatility': 1.0,
            'local_slope': -0.1192,
            'change_point_detected': False,
            'acf_peak_lag': 0,
            'diff_adf_pvalue': 0.5,
            'sample_entropy': 0.0,
            'spectral_entropy': 0.0,
            'fft_peak_freq': 0.0,
            'acf_365': 0.0,
            '_local_errors': {
                "chunk_ensemble": 0.1523,
                "multi_resolution": 0.1687,
                "residual_correction_advanced": 0.1456,
            }
        }

        # 模拟候选技能
        class Skill:
            def __init__(self, name, min_data=30, full_hist=True, tags=[], hint=""):
                self.name = name
                self.min_data_points = min_data
                self.requires_full_history = full_hist
                self.strength_tags = tags
                self.decision_hint = hint

        candidates = [
            Skill("chunk_ensemble", 50, True, ["long_sequence", "ensemble"], "长序列多步预测首选"),
            Skill("multi_resolution", 30, False, ["multiscale", "long_sequence"], "多分辨率预测"),
            Skill("residual_correction_advanced", 50, True, ["corrector", "advanced"], "高级残差修正"),
        ]

        long_skills = ['chunk_ensemble', 'multi_resolution', 'residual_correction_advanced']

        prompt = build_prompt(
            profile=profile,
            history=mock_history,
            candidates=candidates,
            local_errors=profile['_local_errors'],
            LONG_SKILLS=long_skills,
            step_counter=1
        )

        print(f"      📏 Prompt长度: {len(prompt)} 字符")

        try:
            start_time = time.time()
            resp = client.call_with_retry(prompt, max_retries=2)
            elapsed = time.time() - start_time
            result["response_time"] = elapsed

            content = resp.choices[0].message.content

            # Token统计
            if hasattr(resp, 'usage') and resp.usage:
                result["token_usage"] = {
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                    "total_tokens": resp.usage.total_tokens
                }
                print(f"      📊 Token: prompt={resp.usage.prompt_tokens}, completion={resp.usage.completion_tokens}, total={resp.usage.total_tokens}")

            result["content"] = content
            print(f"      ✅ 响应成功 (耗时: {elapsed:.2f}s)")
            print(f"      📝 响应长度: {len(content)} 字符")

        except Exception as e:
            print(f"      ❌ 核心预测响应失败: {e}")
            result["error"] = str(e)
            return result

        # ============================================================
        # 测试3：JSON解析验证
        # ============================================================
        print(f"\n   🔗 测试3: JSON解析验证...")

        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            print(f"      ✅ 找到JSON (长度: {len(json_str)})")

            try:
                data = json.loads(json_str)
                result["json_valid"] = True
                print(f"      ✅ JSON解析成功")
                print(f"      📋 字段: {list(data.keys())}")

                # 检查必要字段
                required_keys = ["skill_weights", "replan_interval"]
                for key in required_keys:
                    if key in data:
                        result["keys_found"].append(key)
                        print(f"         ✅ 找到 '{key}'")
                    else:
                        result["keys_missing"].append(key)
                        print(f"         ❌ 缺少 '{key}'")

                # 检查 skill_weights
                if "skill_weights" in data:
                    weights = data["skill_weights"]
                    print(f"         📊 权重: {weights}")
                    weight_sum = sum(weights.values())
                    print(f"         📊 权重之和: {weight_sum:.6f}")

                    if abs(weight_sum - 1.0) < 0.001:
                        print(f"         ✅ 权重之和为1 (合理)")
                    else:
                        print(f"         ⚠️ 权重之和不为1 ({weight_sum:.6f})")

                # 检查 relation_to_reference（如果有）
                if "relation_to_reference" in data:
                    relation = data["relation_to_reference"]
                    print(f"         📌 relation_to_reference: {relation}")

            except json.JSONDecodeError as e:
                print(f"      ❌ JSON解析失败: {e}")
                result["json_valid"] = False
        else:
            print(f"      ❌ 未找到JSON")
            result["json_valid"] = False

        # ============================================================
        # 综合判断
        # ============================================================
        print(f"\n   🔗 综合判断:")

        score = 0
        max_score = 10

        if result["status_code"] == 200 and result["response_time"]:
            score += 3
            print(f"      ✅ 基础连通性: 通过 (+3分)")

        if result["response_time"] and result["response_time"] < 30:
            score += 2
            print(f"      ✅ 响应速度: {result['response_time']:.1f}s < 30s (+2分)")
        elif result["response_time"] and result["response_time"] < 60:
            score += 1
            print(f"      ⚠️ 响应速度: {result['response_time']:.1f}s < 60s (+1分)")
        else:
            print(f"      ❌ 响应速度: {result['response_time']:.1f}s >= 60s (0分)")

        if result["json_valid"]:
            score += 3
            print(f"      ✅ JSON解析: 成功 (+3分)")
        else:
            print(f"      ❌ JSON解析: 失败 (0分)")

        if "skill_weights" in result["keys_found"] and "replan_interval" in result["keys_found"]:
            score += 2
            print(f"      ✅ 必要字段: 完整 (+2分)")
        else:
            missing = ", ".join(result["keys_missing"])
            print(f"      ❌ 缺少字段: {missing} (0分)")

        result["score"] = score
        result["max_score"] = max_score

        if score >= 7:
            result["overall"] = "✅ 可用 (推荐)"
            print(f"\n      🎯 总评分: {score}/{max_score} - ✅ 模型可用，推荐使用")
        elif score >= 4:
            result["overall"] = "⚠️ 部分可用 (谨慎使用)"
            print(f"\n      🎯 总评分: {score}/{max_score} - ⚠️ 部分可用，建议进一步测试")
        else:
            result["overall"] = "❌ 不可用"
            print(f"\n      🎯 总评分: {score}/{max_score} - ❌ 模型不可用")

        # 额度建议
        print(f"\n      💰 额度建议:")
        if model_name == "glm-4.7-flash":
            print(f"         - 如果 GLM-4.7-Flash 消耗 glm-4.7 额度: 可用 4,196,175 tokens")
            print(f"         - 如果有独立 Flash 额度包: 请查看控制台")
            print(f"         - 推荐作为主要模型使用（速度快）")

        return result

    except Exception as e:
        result["error"] = str(e)
        result["overall"] = "❌ 不可用 (异常)"
        print(f"\n   ❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return result


# ============================================================
# 主测试流程
# ============================================================

def main():
    print("\n" + "=" * 80)
    print("🚀 开始模型测试 (GLM-4.7-Flash 专项)")
    print("=" * 80)

    results = []

    for model_name, description in MODELS_TO_TEST:
        result = test_model(model_name, description)
        results.append(result)

    # ============================================================
    # 汇总报告
    # ============================================================
    print("\n" + "=" * 80)
    print("📊 测试汇总报告")
    print("=" * 80)

    print("\n┌─────────────────┬──────────┬────────────┬────────────┬─────────────┐")
    print("│ 模型名称        │ 可用状态 │ 响应时间(s)│ Token消耗  │ 评分        │")
    print("├─────────────────┼──────────┼────────────┼────────────┼─────────────┤")

    for r in results:
        status = "✅ 可用" if r.get("overall", "").startswith("✅") else "❌ 不可用"
        if r.get("overall", "").startswith("⚠️"):
            status = "⚠️ 部分可用"
        response_time = f"{r['response_time']:.1f}" if r.get('response_time') else "N/A"
        tokens = r.get("token_usage", {}).get("total_tokens", "N/A")
        score = f"{r.get('score', 0)}/{r.get('max_score', 10)}"
        print(f"│ {r['model']:<15} │ {status:<8} │ {response_time:<12} │ {str(tokens):<12} │ {score:<13} │")

    print("└─────────────────┴──────────┴────────────┴────────────┴─────────────┘")

    # ============================================================
    # 结论与建议
    # ============================================================
    print("\n" + "=" * 80)
    print("📋 结论与建议")
    print("=" * 80)

    # 找到 GLM-4.7-Flash 的结果
    flash_result = None
    for r in results:
        if r['model'] == 'glm-4.7-flash':
            flash_result = r
            break

    if flash_result:
        print(f"\n   📌 GLM-4.7-Flash 测试结果:")
        print(f"      {'✅' if flash_result.get('overall', '').startswith('✅') else '❌'} {flash_result.get('overall', '未知')}")
        print(f"      评分: {flash_result.get('score', 0)}/{flash_result.get('max_score', 10)}")
        print(f"      响应时间: {flash_result.get('response_time', 'N/A'):.1f}s")

        if flash_result.get('overall', '').startswith('✅'):
            print(f"\n   ✅ 推荐使用 GLM-4.7-Flash 作为主要模型")
            print(f"\n   📌 配置建议:")
            print(f"      在 config.yaml 中设置:")
            print(f"      llm:")
            print(f"        model: \"glm-4.7-flash\"")
            print(f"        temperature: 0.0")
            print(f"        max_tokens: 4096")
        else:
            print(f"\n   ⚠️ GLM-4.7-Flash 不可用，建议:")
            print(f"      1. 检查模型名称是否正确（智谱API实际名称）")
            print(f"      2. 检查是否有对应额度")
            print(f"      3. 使用 glm-4.7 作为替代")
    else:
        print("\n   ⚠️ GLM-4.7-Flash 未测试到结果")

    print("\n" + "=" * 80)
    print("✅ 模型测试完成")
    print("=" * 80)


if __name__ == '__main__':
    main()