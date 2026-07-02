# test_zhipu_api.py
"""
智谱大模型 API 全面测试脚本
测试内容：
1. 环境信息
2. API 基础连通性
3. 简单对话
4. 带推理的对话 (reasoning_content)
5. 超时测试
6. Token 统计测试
"""

import sys
import os
import time
import json
from datetime import datetime

print("=" * 70)
print("🔍 智谱 API 全面测试")
print(f"📅 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# ==================== 1. 环境信息 ====================
print("\n📋 1. 环境信息")
print("-" * 40)

print(f"   Python 版本: {sys.version}")
print(f"   工作目录: {os.getcwd()}")

# 检查 openai 版本
try:
    import openai

    print(f"   OpenAI 版本: {openai.__version__}")
except Exception as e:
    print(f"   ❌ OpenAI 导入失败: {e}")

# 检查 requests 版本
try:
    import requests

    print(f"   Requests 版本: {requests.__version__}")
except Exception as e:
    print(f"   ❌ Requests 导入失败: {e}")

# ==================== 2. 配置信息 ====================
print("\n📋 2. API 配置")
print("-" * 40)

# 从 src.config 导入
try:
    from src.config import ZHIPU_API_KEY, OPENAI_API_BASE

    API_KEY = ZHIPU_API_KEY
    API_BASE = OPENAI_API_BASE
    print(f"   API Base URL: {API_BASE}")
    print(f"   API Key: {API_KEY[:8]}...{API_KEY[-4:]}")
except ImportError:
    print("   ⚠️ 无法从 src.config 导入，使用手动配置")
    # 使用用户提供的 API Key
    API_KEY = "6a4a1ccfac924e95a8d7ab903325a5c1.teAGpy4lhpWvofKF"
    API_BASE = "https://open.bigmodel.cn/api/paas/v4/"
    print(f"   API Base URL: {API_BASE}")
    print(f"   API Key: {API_KEY[:8]}...{API_KEY[-4:]}")

# ==================== 3. 基础连通性测试 ====================
print("\n📋 3. 基础连通性测试（使用 requests）")
print("-" * 40)

import requests

url = f"{API_BASE}/chat/completions"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

data = {
    "model": "glm-4.5-air",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 20
}

print(f"   📤 请求 URL: {url}")
print(f"   📤 请求数据: {json.dumps(data, ensure_ascii=False)}")

try:
    start_time = time.time()
    resp = requests.post(url, headers=headers, json=data, timeout=15)
    elapsed = time.time() - start_time

    print(f"   📥 响应状态码: {resp.status_code}")
    print(f"   📥 响应耗时: {elapsed:.2f}s")
    print(f"   📥 响应头: {dict(resp.headers)}")

    if resp.status_code == 200:
        result = resp.json()
        print(f"   ✅ 请求成功!")
        print(f"   📝 响应内容: {result.get('choices', [{}])[0].get('message', {}).get('content', '')}")
        if 'usage' in result:
            print(f"   📊 Token 消耗: {result['usage']}")
    else:
        print(f"   ❌ HTTP 错误: {resp.text}")
except requests.exceptions.Timeout:
    print("   ❌ 请求超时 (15s)")
except requests.exceptions.ConnectionError as e:
    print(f"   ❌ 连接错误: {e}")
except Exception as e:
    print(f"   ❌ 未知错误: {type(e).__name__}: {e}")

# ==================== 4. OpenAI SDK 测试 ====================
print("\n📋 4. OpenAI SDK 测试")
print("-" * 40)

try:
    from openai import OpenAI

    print(f"   📤 创建客户端...")
    client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE,
        timeout=15
    )
    print(f"   ✅ 客户端创建成功")

    print(f"   📤 发送请求 (model=glm-4.5-air)...")
    start_time = time.time()
    resp = client.chat.completions.create(
        model="glm-4.5-air",
        messages=[{"role": "user", "content": "你好，请用一句话介绍你自己"}],
        max_tokens=50,
        temperature=0.0
    )
    elapsed = time.time() - start_time

    print(f"   📥 响应耗时: {elapsed:.2f}s")
    print(f"   📥 finish_reason: {resp.choices[0].finish_reason}")
    print(f"   📝 响应内容: {resp.choices[0].message.content}")

    if hasattr(resp, 'usage') and resp.usage:
        print(
            f"   📊 Token: prompt={resp.usage.prompt_tokens}, comp={resp.usage.completion_tokens}, total={resp.usage.total_tokens}")

    print(f"   ✅ OpenAI SDK 测试通过!")

except ImportError as e:
    print(f"   ❌ OpenAI 导入失败: {e}")
except Exception as e:
    print(f"   ❌ OpenAI SDK 测试失败: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()

# ==================== 5. 带推理的测试 (reasoning_content) ====================
print("\n📋 5. 带推理的测试 (reasoning_content)")
print("-" * 40)

try:
    from openai import OpenAI

    client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE,
        timeout=30
    )

    print(f"   📤 发送需要推理的请求...")
    start_time = time.time()
    resp = client.chat.completions.create(
        model="glm-4.5-air",
        messages=[{"role": "user", "content": "1+1=？请先思考再回答"}],
        max_tokens=100,
        temperature=0.0
    )
    elapsed = time.time() - start_time

    print(f"   📥 响应耗时: {elapsed:.2f}s")
    print(f"   📥 finish_reason: {resp.choices[0].finish_reason}")

    msg = resp.choices[0].message
    content = getattr(msg, 'content', '')
    reasoning = getattr(msg, 'reasoning_content', None)

    if reasoning:
        print(f"   📝 reasoning_content 存在，长度: {len(reasoning)}")
        print(f"   📝 reasoning_content 预览: {reasoning[:200]}...")
    else:
        print(f"   ℹ️ 无 reasoning_content（可能模型不支持或未返回）")

    print(f"   📝 最终回答: {content}")

    if hasattr(resp, 'usage') and resp.usage:
        print(
            f"   📊 Token: prompt={resp.usage.prompt_tokens}, comp={resp.usage.completion_tokens}, total={resp.usage.total_tokens}")

    print(f"   ✅ 带推理测试通过!")

except Exception as e:
    print(f"   ❌ 带推理测试失败: {type(e).__name__}: {e}")

# ==================== 6. 不同模型测试 ====================
print("\n📋 6. 不同模型测试")
print("-" * 40)

models_to_test = [
    "glm-4.5-air",
    "glm-4-plus",
    "glm-4-air",
    "glm-4-flash",
]

try:
    from openai import OpenAI

    client = OpenAI(api_key=API_KEY, base_url=API_BASE, timeout=15)

    for model in models_to_test:
        print(f"   📤 测试模型: {model}")
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "1+1="}],
                max_tokens=10
            )
            content = resp.choices[0].message.content
            print(f"   ✅ {model}: {content}")
        except Exception as e:
            print(f"   ❌ {model}: {type(e).__name__}")

except Exception as e:
    print(f"   ❌ 模型测试失败: {e}")

# ==================== 7. 总结 ====================
print("\n" + "=" * 70)
print("📊 测试总结")
print("=" * 70)

print("""
✅ 如果所有测试通过，说明 API 连接正常
❌ 如果某测试失败，请根据错误信息排查：
   - 网络连接: 检查防火墙/代理
   - API Key: 检查是否有效/过期
   - 模型名称: 检查模型是否存在
   - 超时: 检查网络延迟

📌 当前使用的模型建议: glm-4.5-air（资源包充足）
""")

print("=" * 70)