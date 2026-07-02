#!/usr/bin/env python
"""
模型诊断测试程序（猫眼程序）
测试各模型是否可用、响应质量、性能
★ 截取正式运行的真实响应环节
★ 详细打印各方位信息
★ 判断分析输出是否满足要求
★ 给出明确结论
"""

import sys
import os
import json
import time
import re
from datetime import datetime
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("=" * 80)
print("🔍 模型诊断测试程序 (猫眼程序)")
print("=" * 80)
print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# ============================================================
# 测试模型列表（按优先级排序）
# ============================================================
MODELS_TO_TEST = [
    ("glm-4.7", "赠送额度 4,196,175 tokens（优先使用）"),
    ("glm-4.5-air", "付费+赠送 21,691,666 tokens（备用）"),
    ("glm-4", "基础模型（可能已用完）"),
]

# ============================================================
# 测试数据：模拟真实请求
# ============================================================

# 1. 模拟窗口特征（来自真实数据）
MOCK_FEATURES = {
    "trend_strength": 0.0806,
    "seasonal_strength": 0.0793,
    "adf_pvalue": 0.2166,
    "data_length": 600.0,
    "skewness": 0.2862,
    "cv": 0.4100,
    "local_slope_30": -0.1192,
    "local_std_ratio_30": 0.5369,
    "local_change_rate_30": 0.0741,
    "local_mean_ratio_30": 1.2975,
    "local_slope_120": 0.0504,
    "local_std_ratio_120": 0.7162,
}

# 2. 模拟候选技能
MOCK_CANDIDATES = [
    {
        "skill": type('Skill', (), {
            "name": "chunk_ensemble",
            "min_data_points": 50,
            "requires_full_history": True,
            "strength_tags": ["long_sequence", "ensemble"],
            "decision_hint": "长序列多步预测首选"
        })(),
        "prototype_similarity": 0.85,
    },
    {
        "skill": type('Skill', (), {
            "name": "multi_resolution",
            "min_data_points": 30,
            "requires_full_history": False,
            "strength_tags": ["multiscale", "long_sequence"],
            "decision_hint": "多分辨率预测，适合长序列"
        })(),
        "prototype_similarity": 0.80,
    },
    {
        "skill": type('Skill', (), {
            "name": "residual_correction_advanced",
            "min_data_points": 50,
            "requires_full_history": True,
            "strength_tags": ["corrector", "advanced"],
            "decision_hint": "高级残差修正"
        })(),
        "prototype_similarity": 0.70,
    },
]

# 3. 模拟 local_errors
MOCK_LOCAL_ERRORS = {
    "chunk_ensemble": 0.1523,
    "multi_resolution": 0.1687,
    "residual_correction_advanced": 0.1456,
}

# 4. 模拟历史数据（最近10点）
MOCK_HISTORY = [11.5, 11.8, 12.1, 11.9, 12.3, 12.6, 12.4, 12.8, 13.1, 12.9]

# 5. LONG_SKILLS
LONG_SKILLS = ['chunk_ensemble', 'multi_resolution', 'residual_correction_advanced']

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
        "status_code": None,
        "error": None,
        "token_usage": {},
        "content": None,
        "json_valid": False,
        "weight_sum": None,
        "keys_found": [],
        "keys_missing": [],
        "overall": "❌ 不可用"
    }

    try:
        from src.agents.llm_client import LLMClient
        from src.agents.llm_prompts import build_prompt

        # ============================================================
        # 测试1：基础连通性（简单对话）
        # ============================================================
        print(f"\n   🔗 测试1: 基础连通性...")
        client = LLMClient(model=model_name, verbose=False)

        try:
            start_time = time.time()
            simple_resp = client.call_with_retry("请回复'OK'，只输出OK两个字，不要其他内容", max_retries=1)
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

        except Exception as e:
            result["error"] = str(e)
            print(f"      ❌ 基础连通性失败: {e}")
            return result

        # ============================================================
        # 测试2：正式响应模拟（核心预测决策）
        # ============================================================
        print(f"\n   🔗 测试2: 正式响应模拟（核心预测决策）...")

        # 构建真实prompt
        from src.skills.data_profiler import DataProfiler
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
            '_local_errors': MOCK_LOCAL_ERRORS,
        }

        prompt = build_prompt(
            profile=profile,
            history=np.array(MOCK_HISTORY),
            candidates=MOCK_CANDIDATES,
            local_errors=MOCK_LOCAL_ERRORS,
            LONG_SKILLS=LONG_SKILLS,
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

            # ============================================================
            # 测试3：JSON解析验证
            # ============================================================
            print(f"\n   🔗 测试3: JSON解析验证...")

            # 尝试提取JSON
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
                        result["weight_sum"] = weight_sum
                        print(f"         📊 权重之和: {weight_sum:.6f}")

                        # 检查权重是否接近1
                        if abs(weight_sum - 1.0) < 0.001:
                            print(f"         ✅ 权重之和为1 (合理)")
                        else:
                            print(f"         ⚠️ 权重之和不为1 ({weight_sum:.6f})")

                        # 检查权重精度
                        weight_strs = []
                        for k, v in weights.items():
                            s = f"{v:.10f}"
                            weight_strs.append(s)
                            # 检查是否有过多尾随零
                            if s.endswith("0000000000") or s.endswith("00000000"):
                                print(f"         ⚠️ 权重 {k}={s} 可能有精度问题 (尾随零过多)")

                    # 检查 relation_to_reference
                    if "relation_to_reference" in data:
                        relation = data["relation_to_reference"]
                        print(f"         📌 relation_to_reference: {relation}")
                        if relation in ["completely_different", "partially_referenced", "adopted_with_modifications"]:
                            print(f"         ✅ relation_to_reference 有效")
                        else:
                            print(f"         ⚠️ relation_to_reference 值异常: {relation}")

                except json.JSONDecodeError as e:
                    print(f"      ❌ JSON解析失败: {e}")
                    result["json_valid"] = False
            else:
                print(f"      ❌ 未找到JSON")
                result["json_valid"] = False

        except Exception as e:
            print(f"      ❌ 正式响应失败: {e}")
            result["error"] = str(e)
            return result

        # ============================================================
        # 综合判断
        # ============================================================
        print(f"\n   🔗 综合判断:")

        # 计分规则
        score = 0
        max_score = 10
        reasons = []

        # 基础连通性
        if result["status_code"] == 200 and result["response_time"]:
            score += 3
            reasons.append("✅ 基础连通性正常")
            print(f"      ✅ 基础连通性: 通过 (+3分)")

        # 响应时间
        if result["response_time"] and result["response_time"] < 30:
            score += 2
            reasons.append(f"✅ 响应快速 ({result['response_time']:.1f}s)")
            print(f"      ✅ 响应时间: {result['response_time']:.1f}s < 30s (+2分)")
        elif result["response_time"] and result["response_time"] < 60:
            score += 1
            reasons.append(f"⚠️ 响应较慢 ({result['response_time']:.1f}s)")
            print(f"      ⚠️ 响应时间: {result['response_time']:.1f}s < 60s (+1分)")
        else:
            reasons.append(f"❌ 响应过慢 ({result['response_time']:.1f}s)")
            print(f"      ❌ 响应时间: {result['response_time']:.1f}s >= 60s (0分)")

        # JSON有效性
        if result["json_valid"]:
            score += 3
            reasons.append("✅ JSON解析成功")
            print(f"      ✅ JSON解析: 成功 (+3分)")
        else:
            reasons.append("❌ JSON解析失败")
            print(f"      ❌ JSON解析: 失败 (0分)")

        # 权重合理性
        if result["weight_sum"] is not None and abs(result["weight_sum"] - 1.0) < 0.001:
            score += 2
            reasons.append(f"✅ 权重之和为1 ({result['weight_sum']:.6f})")
            print(f"      ✅ 权重之和: {result['weight_sum']:.6f} ≈ 1 (+2分)")
        elif result["weight_sum"] is not None:
            reasons.append(f"⚠️ 权重之和不为1 ({result['weight_sum']:.6f})")
            print(f"      ⚠️ 权重之和: {result['weight_sum']:.6f} ≠ 1 (+1分)")
            score += 1

        # 必要字段
        if "skill_weights" in result["keys_found"] and "replan_interval" in result["keys_found"]:
            reasons.append("✅ 必要字段完整")
            print(f"      ✅ 必要字段: 完整 (保留)")
        else:
            missing = ", ".join(result["keys_missing"])
            reasons.append(f"❌ 缺少字段: {missing}")
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
            print(f"\n      🎯 总评分: {score}/{max_score} - ❌ 模型不可用，请使用其他模型")

        # 详细原因
        print(f"\n      📋 详细原因:")
        for reason in reasons:
            print(f"         {reason}")

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
    print("🚀 开始模型诊断测试")
    print("=" * 80)

    results = []
    usable_models = []

    for model_name, description in MODELS_TO_TEST:
        result = test_model(model_name, description)
        results.append(result)

        if result.get("overall", "").startswith("✅"):
            usable_models.append(model_name)

    # ============================================================
    # 汇总报告
    # ============================================================
    print("\n" + "=" * 80)
    print("📊 测试汇总报告")
    print("=" * 80)

    print("\n┌─────────────┬──────────┬────────────┬────────────┬─────────────┐")
    print("│ 模型名称     │ 可用状态 │ 响应时间(s)│ Token消耗  │ 评分        │")
    print("├─────────────┼──────────┼────────────┼────────────┼─────────────┤")

    for r in results:
        status = "✅ 可用" if r.get("overall", "").startswith("✅") else "❌ 不可用"
        if r.get("overall", "").startswith("⚠️"):
            status = "⚠️ 部分可用"
        response_time = f"{r['response_time']:.1f}" if r.get('response_time') else "N/A"
        tokens = r.get("token_usage", {}).get("total_tokens", "N/A")
        score = f"{r.get('score', 0)}/{r.get('max_score', 10)}"
        print(f"│ {r['model']:<13} │ {status:<8} │ {response_time:<12} │ {str(tokens):<12} │ {score:<13} │")

    print("└─────────────┴──────────┴────────────┴────────────┴─────────────┘")

    # ============================================================
    # 结论与建议
    # ============================================================
    print("\n" + "=" * 80)
    print("📋 结论与建议")
    print("=" * 80)

    if usable_models:
        print(f"\n   ✅ 可用模型: {', '.join(usable_models)}")

        # 推荐最优模型
        best = None
        best_score = -1
        for r in results:
            if r.get("score", 0) > best_score:
                best_score = r.get("score", 0)
                best = r

        if best:
            print(f"\n   🏆 推荐使用: {best['model']}")
            print(f"      评分: {best.get('score', 0)}/{best.get('max_score', 10)}")
            print(f"      响应时间: {best.get('response_time', 'N/A'):.1f}s")
            if best.get("token_usage"):
                print(f"      Token消耗: {best['token_usage'].get('total_tokens', 'N/A')} tokens")

            # 配置建议
            print(f"\n   📌 配置建议:")
            print(f"      在 config.yaml 中设置:")
            print(f"      llm:")
            print(f"        model: \"{best['model']}\"")
            print(f"        temperature: 0.0")
            print(f"        max_tokens: 4096")
    else:
        print("\n   ❌ 没有可用模型")
        print("\n   建议:")
        print("   1. 检查 API 密钥是否有效")
        print("   2. 检查网络连接")
        print("   3. 检查模型名称是否正确")
        print("   4. 检查是否有足够额度")

    print("\n" + "=" * 80)
    print("✅ 模型诊断测试完成")
    print("=" * 80)

    return results


if __name__ == '__main__':
    main()