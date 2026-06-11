import numpy as np
from .base import BaseSkill
from .naive import NaiveSkill

class FFTFilterSkill(BaseSkill):
    """频谱分解+低频趋势保留：FFT去噪后预测平滑序列"""
    def __init__(self, base_skill=None, n_components=5):
        super().__init__()
        self.name = "fft_filter"
        self.description = "FFT滤波预测：保留主频分量重构序列后再预测，自动去噪"
        self.base_skill = base_skill or NaiveSkill()
        self.n_components = n_components
        self.min_data_points = 50
        self.requires_full_history = True
        self.strength_tags = ["spectral", "denoise"]
        self.model_family = "lightweight"
        self.required_features = ["data_length"]
        self.decision_hint = "适合包含明显周期成分且噪声较多的序列。"
        self.state_card = {
            "when_to_use": {"conditions": [{"field": "data_length", "op": ">", "value": 100}], "logic": "AND"},
            "when_not_to_use": {"conditions": [], "logic": "OR"},
            "visible_cues": ["序列有周期波动但局部噪声大"],
            "verification_cue": "重构序列更平滑",
            "fallback_skill": "naive"
        }

    def execute(self, history, horizon, **kwargs):
        n = len(history)
        if n < self.min_data_points:
            return self.base_skill.execute(history, horizon, **kwargs)
        fft_vals = np.fft.fft(history)
        magnitude = np.abs(fft_vals)
        # 保留最大的 n_components 个频率成分（不含直流）
        top_idx = np.argsort(magnitude[1:])[-self.n_components:] + 1
        filtered = np.zeros_like(fft_vals, dtype=complex)
        filtered[0] = fft_vals[0]  # 保留直流
        filtered[top_idx] = fft_vals[top_idx]
        smoothed = np.fft.ifft(filtered).real
        return self.base_skill.execute(smoothed, horizon, **kwargs)