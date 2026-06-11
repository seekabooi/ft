from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class TaskInstance(BaseModel):
    id: str
    dataset_id: str
    template_id: str
    question: str
    question_type: str
    history: List[float]
    horizon: int
    frequency: str = "daily"          # 必须存在
    prediction_target: dict
    resolution_date: datetime
    difficulty_level: int
    ground_truth: Optional[float] = None
    ground_truth_extractor: str = ""
    dates: Optional[List[str]] = None  # 新增：对应历史数据的日期列表（字符串或时间戳）
    target_date: Optional[str] = None