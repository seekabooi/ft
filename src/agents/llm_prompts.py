# src/agents/llm_prompts.py
"""
LLM Prompts - 桥接文件
所有提示词已迁移到 experiments/autotune/prompts.py
此文件保留向后兼容
"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 从新位置导入
from experiments.autotune.prompts import (
    build_prompt,
    build_preprocess_prompt,
    build_post_enhance_prompt,
    build_strategy_generation_prompt
)

# 保持向后兼容
__all__ = [
    'build_prompt',
    'build_preprocess_prompt',
    'build_post_enhance_prompt',
    'build_strategy_generation_prompt'
]