## 项目总结

### 一、目录结构及文件作用

```
futureTime/
├── run_benchmark.py                 # 主入口：解析命令行参数，构建技能注册表，启动固定起源多步评估，输出指标并保存CSV
│
├── data/
│   └── dataset_registry.yaml        # 数据集配置（路径、频率、目标列等）
│
├── src/
│   ├── config.py                    # 全局配置（API Key、存储路径等）
│   │
│   ├── dataset/
│   │   ├── registry.py              # 读取数据集注册表
│   │   └── loader.py                # 加载Parquet/CSV为DataFrame（日期索引）
│   │
│   ├── tasks/
│   │   └── instance.py              # TaskInstance数据模型（history, dates, frequency, horizon等）
│   │
│   ├── skills/                      # 技能库（共28个技能）
│   │   ├── base.py                  # 技能基类：状态卡、验证、元数据（min_data_points, decision_hint等）
│   │   ├── registry.py              # 技能注册表
│   │   ├── data_profiler.py         # 特征提取器：统计特征、周期检测、多视角快照、日期特征、样本熵、频谱熵等
│   │   ├── skill_matcher.py         # 技能匹配器：硬过滤 + 原型DTW相似度 + 探索加分 + 路由加成（长序列推荐技能+0.25）
│   │   │
│   │   ├── 基础统计技能
│   │   │   ├── naive.py              # 简单移动平均（最后5点均值）
│   │   │   ├── naive_drift.py        # 带漂移朴素
│   │   │   ├── seasonal_naive.py     # 季节性朴素（支持动态周期检测）
│   │   │   ├── local_drift.py        # 局部斜率外推
│   │   │   ├── ets.py                # 指数平滑（自动选择趋势/季节）
│   │   │   ├── theta.py              # Theta方法
│   │   │   ├── holt_winters.py       # Holt-Winters三指数平滑
│   │   │   ├── croston.py            # Croston间歇需求预测
│   │   │   ├── tbats.py              # TBATS复杂季节模型
│   │   │   ├── arima.py (已禁用)     # 固定阶ARIMA(1,1,1)
│   │   │   └── auto_arima.py         # 自动ARIMA（支持季节性）
│   │   │
│   │   ├── 机器学习技能
│   │   │   ├── prophet_skill.py      # Facebook Prophet
│   │   │   ├── feature_gbm.py        # LightGBM（滞后特征）
│   │   │   └── incremental_gbm.py    # 增量LightGBM（可选）
│   │   │
│   │   ├── 长序列专用技能
│   │   │   ├── chunk_ensemble.py     # 分块集成预测
│   │   │   ├── multi_resolution.py   # 多分辨率下采样+重构
│   │   │   ├── residual_correction_advanced.py  # 递归残差修正
│   │   │   ├── fft_filter.py         # FFT滤波去噪
│   │   │   └── adaptive_weighted_ensemble.py  # 自适应加权组合
│   │   │
│   │   ├── 日历/分解技能
│   │   │   ├── calendar_skill.py     # 日历同期预测（支持日度/月度）
│   │   │   ├── fourier_skill.py      # 傅里叶级数拟合季节
│   │   │   ├── multi_seasonal_naive.py  # 乘法季节性朴素
│   │   │   ├── stl_decompose_skill.py   # STL分解后预测
│   │   │   ├── detrender.py          # 线性趋势分离
│   │   │   ├── seasonal_extractor.py # 季节成分提取
│   │   │   ├── trend_forecaster.py   # 趋势预测器
│   │   │   ├── seasonal_forecaster.py# 季节预测器
│   │   │   └── bias_corrector.py     # 偏差修正
│   │   │
│   │   └── 组合器（仅DAG模式，已从注册表中移除）
│   │       ├── additive_combiner.py
│   │       ├── multiplicative_combiner.py
│   │       ├── progressive_adaptive_combiner.py
│   │       └── gated_ensemble.py
│   │
│   ├── agents/
│   │   ├── base.py                   # Agent抽象基类
│   │   ├── llm_planner.py            # 核心决策器：特征提取→候选匹配→局部误差→LLM决策→加权预测（支持递归与一次性模式）
│   │   ├── llm_client.py             # LLM调用封装（含重试、解析权重与重决策间隔）
│   │   └── llm_prompts.py            # 构建Prompt（包含序列特征、候选技能、重决策间隔要求）
│   │
│   ├── evaluation/
│   │   └── fixed_origin_evaluator.py # 固定起源多步评估器：切分训练/测试集，计算MASE/sMAPE/RMSSE/OWA等指标
│   │
│   └── analysis/
│       └── diagnostic.py             # 诊断日志（预留）
│
├── storage/                          # 运行时生成
│   ├── logs/                         # 详细决策日志（JSON格式）
│   ├── eval_*.csv                    # 预测明细（预测值、真实值）
│   └── plots/                        # 可视化图表输出
│
├── visualization.py                  # 静态可视化
├── visualization_live.py             # 实时仪表板（Dash）
└── experiments/                      # 实验脚本（预留）
```

---

### 二、处理逻辑流程图

```
命令行参数解析 (run_benchmark.py)
        │
        ▼
构建技能注册表 (28个技能，移除DAG组合器)
        │
        ▼
创建 LLMPlannerAgent (llm_planner.py)
        │
        ▼
创建 FixedOriginEvaluator (fixed_origin_evaluator.py)
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ 取前 min_train_size 个点作为训练集                         │
│ 后 horizon 个点作为测试集                                  │
│ 计算 MASE/RMSSE 缩放因子（基于季节性朴素或随机游走）       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
          LLMPlannerAgent.predict(task)
                     │
    ┌────────────────┼────────────────────────┐
    ▼                ▼                        ▼
DataProfiler     SkillMatcher            局部误差计算
(提取特征:       (硬过滤+原型DTW+         (多步滚动MAE)
季节/趋势/周期   路由加成+探索加分)        对推荐技能折扣0.8
ADF/熵/FFT等)         │                        │
    │                │                        │
    └──────┬─────────┴──────┬─────────────────┘
           │                │
           ▼                ▼
     profile 字典    candidates 列表 + local_errors
           │                │
           └──────┬─────────┘
                  ▼
        _decide_weights / _decide_weights_and_interval
        ┌───────────────────────────────────────────────┐
        │ 构造Prompt（特征、候选技能局部误差、决策提示）│
        │ 调用GLM-4 (temperature=0.35, 可调整)         │
        │ 解析JSON → skill_weights + replan_interval   │
        │ 归一化 + 微小扰动                            │
        │ 后处理：长序列推荐技能总权重<0.8 → 强制单技能│
        └───────────────┬───────────────────────────────┘
                        │
                        ▼
              plan (权重 + 重决策间隔)
                        │
                        ▼
         ┌──────────────┴──────────────┐
         │ 若 data_len < 200：         │
         │   一次性加权预测（不递归）   │
         │ 否则：                      │
         │   递归预测（每步按LLM决定   │
         │   的间隔重新决策，并支持     │
         │   不确定性触发重决策）       │
         └──────────────┬──────────────┘
                        │
                        ▼
                  加权预测数组
                        │
                        ▼
               返回预测值 → 指标计算 → 输出报告
```

---

### 三、主要创新点

1. **LLM 驱动的动态技能组合与精细化权重**  
   LLM 根据序列特征（长度、季节强度、趋势、周期、ADF、熵、FFT等）和局部误差，自主选择 1~3 个技能并分配十位小数精度权重，实现高度个性化的动态集成。

2. **递归预测 + LLM 自主决策重决策间隔**  
   - 对于中等长度序列（≥200点），采用递归多步预测，每预测一步可将预测值加入历史再决策。  
   - LLM 输出 `replan_interval`（1~5），控制下次强制重决策的步数，同时支持预测值异常（z-score > 阈值）时立即重决策。  
   - 短序列（<200点）自动切换为一次性预测，避免误差累积。

3. **多视角原型匹配与状态条件化技能包**  
   引入趋势快照和季节快照，与技能原型进行DTW相似度计算；技能携带状态卡（when_to_use/when_not_to_use）和决策提示（decision_hint），形成有明确适用边界的知识单元。

4. **长序列专用技能与路由加成**  
   设计了 `chunk_ensemble`、`multi_resolution`、`residual_correction_advanced` 等长序列优化技能，并在 `SkillMatcher` 中对长度>400且季节强度>0.4的序列给予 +0.25 相似度加成，提升其被选中的概率。

5. **按需特征计算 + 丰富特征集**  
   技能通过 `required_features` 声明所需特征，`DataProfiler` 仅计算必要统计量。特征集包括：季节强度、趋势强度、ADF p-value、缺失率、自相关峰值、一阶差分平稳性、样本熵、频谱熵、FFT主频、年自相关等。

6. **多层兜底与技能休眠**  
   - LLM 决策失败 → 基于局部误差倒数加权（兜底）。  
   - 加权预测失败 → 回退到 `naive` 或均值。  
   - 技能长期未选中自动休眠，减少无效干扰。

7. **日历技能深度利用日期信息**  
   `CalendarSkill` 根据频率自动选择日度（按月-日匹配）或月度模式，利用历史同期数据加权平均或趋势外推，在强季节数据上提供稳健基准。

8. **严格的固定起源多步评估**  
   完全遵循 M4 竞赛标准，支持一次性预测和递归预测两种模式，指标包括 MASE、RMSSE、sMAPE、OWA 等。

---

### 四、运行指令

#### 基本一次性预测（默认，短序列自动一次性，长序列可选递归）
```bash
# 使用默认参数（训练窗口132，预测12步）
python run_benchmark.py --dataset airline_passengers

# 指定训练窗口和预测步数
python run_benchmark.py --dataset melbourne_temp --min_train_size 600 --horizon 7

# 指定LLM模型（如 glm-4）
python run_benchmark.py --dataset melbourne_temp --model glm-4
```

#### 使用递归预测（强制对所有长度序列使用递归，需修改代码或通过参数）
当前代码中，`data_len < 200` 自动使用一次性预测；若希望所有序列都递归，可将 `predict` 方法中的短序列分支删除或调整阈值。默认行为已区分长短序列。

#### 单技能模式（测试特定技能）
```bash
python run_benchmark.py --dataset melbourne_temp --skill_mode single --skill_name chunk_ensemble --min_train_size 600 --horizon 7
```

#### 禁用技能（仅使用基准模型）
```bash
python run_benchmark.py --dataset melbourne_temp --no_skills
```

#### 生成可视化图表（需先运行评估）
```bash
python visualization.py --dataset melbourne_temp
```

#### 其他参数
- `--data_ratio`：使用数据集的比例（0~1）  
- `--llm_call_interval`：LLM调用间隔步数（仅影响一次性预测中的重复调用，目前未使用）  
- `--skill_mode ensemble`：使用固定的集成技能（简单平均）

---

### 五、当前性能示例

| 数据集 | 训练窗口 | 预测步数 | MASE | 模式 |
|--------|----------|----------|------|------|
| airline_passengers | 132 | 12 | 0.605 | 一次性加权组合 |
| melbourne_temp | 600 | 7 | 1.243 | 一次性强制规则（或递归最佳） |

递归预测在墨尔本温度上 MASE 约 1.22~1.33，与强制规则相当。