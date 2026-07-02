"""
LLM Client with Token Statistics - 修复版（模型自动切换）
★ 模型自动切换（额度用完后切换到下一个）
★ 优先级：glm-4.7 > glm-4.5-air
★ ★ 自带日志文件写入：所有 print 信息同时写入 llog/llm_detail.log
★ ★ 兼容外部 logger（如果传入则使用，否则直接写文件）
"""

import re
import json
import time
import sys
import io
import traceback
import os
from openai import OpenAI
from src.config import ZHIPU_API_KEY, OPENAI_API_BASE


class LLMClient:
    _total_prompt_tokens = 0
    _total_completion_tokens = 0
    _total_tokens = 0
    _call_count = 0
    _error_count = 0
    _token_log = []
    _last_error = None

    DEFAULT_MAX_TOKENS = 4096
    DEFAULT_TIMEOUT = 180

    # ★ 模型优先级列表（按速度和额度综合排序）
    MODEL_PRIORITY = [
        ("glm-4.7", "赠送额度 4,196,175 tokens，速度快 (1.2s)"),
        ("glm-4.5-air", "付费+赠送 21,607,713 tokens，速度快"),
        ("glm-4", "基础模型（可能已用完）"),
    ]

    # ★ 当前使用的模型索引
    _current_model_index = 0

    def __init__(self, model="glm-4.7", log_file=None, max_tokens=None, verbose=False, logger=None):
        self.display_model = model
        self._set_model(model)
        self.max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS
        self.log_file = log_file
        self.verbose = verbose
        self.logger = logger

        # ★ 日志文件路径（独立于外部 logger）
        self.llm_detail_log = os.path.join('llog', 'llm_detail.log')
        try:
            os.makedirs('llog', exist_ok=True)
        except:
            pass

        if self.verbose:
            self._log_info(f"  📌 模型: {self.display_model}")
            self._log_info(f"  💰 额度信息: {self._get_quota_info(self.display_model)}")

        import httpx
        timeout = httpx.Timeout(
            connect=10.0,
            read=self.DEFAULT_TIMEOUT,
            write=10.0,
            pool=10.0
        )

        try:
            self.client = OpenAI(
                api_key=ZHIPU_API_KEY,
                base_url=OPENAI_API_BASE,
                timeout=timeout
            )
        except Exception as e:
            print(f"❌ OpenAI客户端初始化失败: {e}")
            raise

    # ★★★ 核心日志方法：同时输出到终端和日志文件 ★★★
    def _log_info(self, msg: str):
        """输出到终端和日志文件（INFO级别）"""
        # 1. 输出到终端
        print(msg)
        # 2. 如果传入了 logger，使用 logger
        if self.logger is not None:
            self.logger.log(msg)
        # 3. 无论是否传入 logger，都写入独立日志文件（保证子进程也能记录）
        try:
            with open(self.llm_detail_log, 'a', encoding='utf-8') as f:
                f.write(msg + '\n')
        except:
            pass

    def _log_warning(self, msg: str):
        """输出警告到终端和日志文件（WARNING级别）"""
        print(msg)
        if self.logger is not None:
            self.logger.log(msg, level="WARNING")
        try:
            with open(self.llm_detail_log, 'a', encoding='utf-8') as f:
                f.write(msg + '\n')
        except:
            pass

    def _set_model(self, model_name: str):
        self.display_model = model_name
        self.model = model_name

    def _get_quota_info(self, model_name: str) -> str:
        quota_info = {
            "glm-4.7": "赠送额度 4,196,175 tokens",
            "glm-4.5-air": "付费+赠送 21,607,713 tokens",
            "glm-4": "基础模型（可能已用完）",
        }
        return quota_info.get(model_name, "未知额度")

    def switch_to_next_model(self):
        current_index = self._current_model_index
        if current_index + 1 < len(self.MODEL_PRIORITY):
            next_model, desc = self.MODEL_PRIORITY[current_index + 1]
            self._current_model_index = current_index + 1
            self._set_model(next_model)
            if self.verbose:
                self._log_info(f"  🔄 切换到模型: {next_model} ({desc})")
            return True
        else:
            if self.verbose:
                self._log_warning(f"  ⚠️ 没有更多可用模型")
            return False

    def _log(self, data: dict):
        if self.log_file:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")
            except Exception:
                pass

    def test_model_available(self) -> tuple:
        try:
            old_out = sys.stdout
            sys.stdout = io.StringIO()
            resp = self.call_with_retry("请回复OK", max_retries=1)
            sys.stdout = old_out
            if resp and resp.choices and resp.choices[0].message.content:
                return True, None
            return False, "模型返回空响应"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    def call_with_retry(self, prompt, max_retries=2):
        last_exception = None

        if self.verbose:
            self._log_info(f"  📤 请求模型: {self.display_model} (API: {self.model})")
            self._log_info(f"  📏 Prompt长度: {len(prompt)} 字符")

        for attempt in range(max_retries + 1):
            try:
                if self.verbose:
                    self._log_info(f"  📤 尝试 {attempt + 1}/{max_retries + 1}...")

                start_time = time.time()

                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=self.max_tokens
                )

                elapsed = time.time() - start_time
                if self.verbose:
                    self._log_info(f"  📥 响应完成 (耗时 {elapsed:.1f}s)")

                if hasattr(resp, 'usage') and resp.usage is not None:
                    usage = resp.usage
                    LLMClient._total_prompt_tokens += usage.prompt_tokens
                    LLMClient._total_completion_tokens += usage.completion_tokens
                    LLMClient._total_tokens += usage.total_tokens
                    LLMClient._call_count += 1
                    LLMClient._token_log.append({
                        'model': self.display_model,
                        'api_model': self.model,
                        'prompt_tokens': usage.prompt_tokens,
                        'completion_tokens': usage.completion_tokens,
                        'total_tokens': usage.total_tokens,
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    })
                    if self.verbose:
                        self._log_info(f"  📊 Token: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}")

                    if LLMClient._total_tokens > 1000000:
                        if self.verbose:
                            self._log_warning(f"  ⚠️ 已消耗 {LLMClient._total_tokens} tokens，注意额度")

                else:
                    LLMClient._call_count += 1

                return resp

            except Exception as e:
                last_exception = e
                LLMClient._error_count += 1
                LLMClient._last_error = str(e)

                if "quota" in str(e).lower() or "insufficient" in str(e).lower():
                    if self.verbose:
                        self._log_warning(f"  ⚠️ 模型 {self.display_model} 额度可能已用完，尝试切换...")
                    if self.switch_to_next_model():
                        import httpx
                        timeout = httpx.Timeout(
                            connect=10.0,
                            read=self.DEFAULT_TIMEOUT,
                            write=10.0,
                            pool=10.0
                        )
                        self.client = OpenAI(
                            api_key=ZHIPU_API_KEY,
                            base_url=OPENAI_API_BASE,
                            timeout=timeout
                        )
                        continue

                if self.verbose:
                    self._log_info(f"  ❌ 请求失败 (尝试 {attempt + 1}/{max_retries + 1})")
                    self._log_info(f"     错误类型: {type(e).__name__}")
                    self._log_info(f"     错误信息: {e}")
                    self._log_info(traceback.format_exc())

                self._log({
                    "event": "llm_error",
                    "attempt": attempt + 1,
                    "model": self.display_model,
                    "api_model": self.model,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": traceback.format_exc()
                })

                if attempt < max_retries:
                    if self.verbose:
                        self._log_info(f"  ⏳ 等待 2 秒后重试...")
                    time.sleep(2)
                else:
                    raise last_exception

        raise last_exception

    def parse_weights_and_interval(self, content):
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                weights = data.get("skill_weights")
                interval = data.get("replan_interval", 2)
                reasoning = data.get("reasoning", "")
                relation = data.get("relation_to_reference", "")
                if weights and isinstance(weights, dict):
                    interval = max(1, min(5, int(interval)))
                    return weights, interval, reasoning, relation, data
            except (json.JSONDecodeError, ValueError) as e:
                self._log({"event": "parse_error", "error": str(e), "content": content})
        return None, 2, "", "", {}

    def parse_weights(self, content):
        weights, _, _, _, _ = self.parse_weights_and_interval(content)
        return weights

    @classmethod
    def get_token_stats(cls) -> dict:
        stats = {
            'calls': cls._call_count,
            'prompt_tokens': cls._total_prompt_tokens,
            'completion_tokens': cls._total_completion_tokens,
            'total_tokens': cls._total_tokens,
            'errors': cls._error_count,
            'last_error': cls._last_error,
            'token_log': cls._token_log
        }
        if cls._call_count > 0:
            stats['avg_prompt_tokens'] = cls._total_prompt_tokens / cls._call_count
            stats['avg_completion_tokens'] = cls._total_completion_tokens / cls._call_count
            stats['avg_total_tokens'] = cls._total_tokens / cls._call_count
        return stats

    @classmethod
    def reset_token_stats(cls):
        cls._total_prompt_tokens = 0
        cls._total_completion_tokens = 0
        cls._total_tokens = 0
        cls._call_count = 0
        cls._error_count = 0
        cls._token_log = []
        cls._last_error = None

    @classmethod
    def print_token_stats(cls, description: str = "LLM Token 统计") -> str:
        stats = cls.get_token_stats()
        lines = []
        lines.append("=" * 60)
        lines.append(f"📊 {description}")
        lines.append("=" * 60)
        lines.append(f"   LLM 调用次数: {stats['calls']}")
        lines.append(f"   Prompt Tokens: {stats['prompt_tokens']:,}")
        lines.append(f"   Completion Tokens: {stats['completion_tokens']:,}")
        lines.append(f"   总 Tokens: {stats['total_tokens']:,}")
        if stats['calls'] > 0:
            lines.append(f"   平均 Prompt: {stats['avg_prompt_tokens']:.0f}")
            lines.append(f"   平均 Completion: {stats['avg_completion_tokens']:.0f}")
            lines.append(f"   平均 Total: {stats['avg_total_tokens']:.0f}")
        lines.append(f"   错误次数: {stats['errors']}")
        if stats['last_error']:
            lines.append(f"   最后错误: {stats['last_error']}")
        lines.append("=" * 60)
        return '\n'.join(lines)