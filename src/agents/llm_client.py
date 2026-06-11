import re
import json
import time
from openai import OpenAI
from tqdm import tqdm
from src.config import ZHIPU_API_KEY, OPENAI_API_BASE

class LLMClient:
    def __init__(self, model="glm-4", log_file=None):
        self.model = model
        self.client = OpenAI(api_key=ZHIPU_API_KEY, base_url=OPENAI_API_BASE, timeout=30)
        self.log_file = log_file

    def _log(self, data: dict):
        if self.log_file:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")
            except Exception:
                pass

    def call_with_retry(self, prompt, max_retries=2):
        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=400,
                    timeout=20
                )
                self._log({"event": "llm_raw_response", "content": resp.choices[0].message.content})
                return resp
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    tqdm.write(f"  ⚠️ LLM调用失败 (尝试 {attempt+1}/{max_retries+1}), 重试...")
                    time.sleep(2)
                else:
                    raise last_exception

    def parse_weights_and_interval(self, content):
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                weights = data.get("skill_weights")
                interval = data.get("replan_interval", 2)
                if weights and isinstance(weights, dict):
                    interval = max(1, min(5, int(interval)))
                    return weights, interval
            except (json.JSONDecodeError, ValueError) as e:
                self._log({"event": "parse_error", "error": str(e), "content": content})
        return None, 2

    def parse_weights(self, content):
        weights, _ = self.parse_weights_and_interval(content)
        return weights