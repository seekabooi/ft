# futureTime 自动调优+Generator 完整项目总结

## 一、目录结构及文件作用

```
futureTime/
├── run_benchmark.py                      # ★ 主入口：解析命令行参数，构建技能注册表，启动评估
│                                         # 支持 --use_rules 参数（使用规则策略预测，跳过LLM）
│                                         # 支持 --no_residual 参数（禁用残差修正技能）
│
├── experiments/
│   └── autotune/                         # ★ 自动调优+Generator 核心模块
│       ├── __init__.py                   # 模块初始化，导出主要类
│       ├── main.py                       # ★ 主控制器：编排采集→生成→验证→对比全流程
│       │                                 # 支持 --verbose 和 --compare 参数
│       ├── config.yaml                   # ★ 配置文件：数据集、固定参数、LLM设置
│       │                                 # 固定参数: long_skill_force_threshold=0.70
│       │                                 #          route_bonus_long_skill=0.15
│       │                                 #          residual_acf_threshold=0.20
│       ├── collector.py                  # ★ 数据采集器：滑动窗口采集窗口数据
│       │                                 # 每个窗口运行预测，记录轨迹（技能权重序列）
│       │                                 # 保存窗口数据到 CSV 和 pickle 文件
│       ├── inducer.py                    # ★ 策略生成器（Generator核心）
│       │                                 # 对每个窗口调用LLM生成3-5个候选策略
│       │                                 # 在测试集上回测，选出每个窗口的best策略
│       │                                 # 支持缓存（避免重复LLM调用）
│       ├── cluster.py                    # ★ 策略聚类器
│       │                                 # 将各窗口的best策略聚类为3个簇
│       │                                 # 从每个簇提炼代表性策略作为规则
│       ├── cache_manager.py              # ★ 缓存管理器
│       │                                 # 缓存每个窗口的策略生成结果
│       │                                 # 基于配置哈希生成缓存键
│       ├── rule_engine.py                # ★ 规则执行引擎
│       │                                 # 根据特征匹配规则，返回参数和策略
│       │                                 # 支持分段权重查询
│       ├── validator.py                  # 规则验证器：在验证集上测试规则效果
│       ├── visualizer.py                 # 可视化器：生成对比图表和报告
│       └── utils.py                      # 工具函数：特征提取、序列化、数据加载
│
├── src/                                  # ★ 核心预测框架
│   ├── agents/
│   │   ├── llm_planner.py                # ★ LLM规划器：三阶段预测（预处理→递归预测→后处理）
│   │   │                                 # 支持规则模式（--use_rules）：使用固定策略预测，跳过LLM
│   │   │                                 # 提供 predict_with_trajectory 方法记录决策轨迹
│   │   ├── llm_client.py                 # LLM客户端：调用GLM-4，支持重试和JSON解析
│   │   └── llm_prompts.py                # Prompt构建：预处理、核心预测、后处理增强
│   ├── skills/                           # ★ 28个预测技能
│   │   ├── base.py                       # 技能基类：状态卡、验证、元数据
│   │   ├── registry.py                   # 技能注册表
│   │   ├── data_profiler.py              # 特征提取器：统计特征、周期检测、偏度、熵等
│   │   ├── skill_matcher.py              # 技能匹配器：硬过滤+DTW相似度+路由加成
│   │   ├── preprocess_skills.py          # 预处理技能：缺失填充、异常截断、标准化等
│   │   ├── postprocess_skills.py         # 后处理技能：逆变换、残差修正、分位数校准
│   │   └── [naive, seasonal_naive, prophet, ...] # 28个具体技能实现
│   ├── evaluation/
│   │   └── fixed_origin_evaluator.py     # 固定起源多步评估：MASE/sMAPE/RMSSE/OWA
│   └── tasks/
│       └── instance.py                   # TaskInstance数据模型
│
├── storage/                               # ★ 运行时生成目录
│   ├── autotune_results/                  # 调优结果
│   │   ├── logs/                         # 详细日志（带时间戳）
│   │   ├── cache/                        # 预测结果缓存（避免重复LLM调用）
│   │   ├── strategy_cache/               # ★ 策略缓存（避免重复策略生成）
│   │   ├── window_data/                  # 窗口数据（pickle格式）
│   │   ├── collected_windows.csv         # 采集结果汇总
│   │   ├── generated_rules.json          # ★ 最终生成的规则文件
│   │   ├── validation_compare_*.png      # 可视化图表
│   │   └── report_*.txt                  # 文本报告
│   ├── logs/                             # Agent运行日志
│   └── eval_*.csv                        # 预测明细
│
└── visualization.py                       # 静态可视化（误差分析图表）
```


## 二、处理逻辑流程图

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        1. 启动调优 (run_benchmark.py 或 autotune.main)      │
│         python -m experiments.autotune.main --dataset melbourne_temp         │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    2. 读取配置 (config.yaml)                                 │
│    固定参数: threshold=0.70, bonus=0.15, acf=0.20  → 不再调优               │
│    数据集: window_sizes=[600], step_size=150, horizon=7                     │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│              3. 数据采集 (collector.py)                                      │
│    滑动窗口: origin从0开始，步长150，直到数据末尾                             │
│    共21个窗口，每个窗口大小600，预测7步                                      │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│              4. 每个窗口执行预测 (LLMPlannerAgent)                            │
│    ┌──────────────────────────────────────────────────────────────────────┐  │
│    │ 阶段1: 预处理 (LLM选择: zscore_normalize 等)                        │  │
│    │ 阶段2: 递归核心预测 (LLM每步决策技能权重 + 重决策间隔)               │  │
│    │ 阶段3: 逆变换 (系统自动绑定)                                        │  │
│    │ 阶段4: 后处理增强 (LLM选择: residual_ar 等)                         │  │
│    └──────────────────────────────────────────────────────────────────────┘  │
│    记录: 预测值 + 轨迹 (每一步的技能权重组合 + 间隔)                        │
│    保存: collected_windows.csv + window_data/*.pkl                         │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│              5. 策略生成 (inducer.py)                                        │
│    ┌──────────────────────────────────────────────────────────────────────┐  │
│    │ 检查缓存 → 命中则直接加载，跳过LLM                                   │  │
│    │ 对每个窗口:                                                         │  │
│    │   1. 提取特征 + 轨迹                                                │  │
│    │   2. 调用LLM生成3-5个候选策略 (分段加权求和策略)                     │  │
│    │   3. 在测试集上回测每个候选策略，计算MASE                           │  │
│    │   4. 选出该窗口的best策略 (MASE最低)                                │  │
│    │ 保存策略缓存                                                        │  │
│    └──────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│              6. 策略聚类 (cluster.py)                                        │
│    收集21个窗口的best策略 → 提取特征向量 → K-Means聚类 → 3个簇              │
│    每个簇代表一种典型的预测模式                                              │
│    从簇中提炼代表性策略作为最终规则                                          │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│              7. 生成规则 (generated_rules.json)                              │
│    {                                                                        │
│      "rules": [                                                             │
│        {                                                                    │
│          "condition": "period == 365 and trend_strength > 0.5",             │
│          "params": {...},                                                   │
│          "skill_strategy": {           # ★ 多阶段技能权重组合               │
│            "stages": [                                                      │
│              {"steps": 3, "weights": {"chunk_ensemble": 0.70, ...}},        │
│              {"steps": 4, "weights": {"chunk_ensemble": 0.60, ...}}         │
│            ]                                                                │
│          }                                                                  │
│        }                                                                    │
│      ]                                                                      │
│    }                                                                        │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│              8. 对比测试 (--compare)                                          │
│    使用规则 vs 不使用规则 → 对比MASE → 打印改善百分比                        │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│              9. 验证与可视化 (validator.py + visualizer.py)                   │
│    生成 validation_compare_*.png 和 report_*.txt                            │
└──────────────────────────────────────────────────────────────────────────────┘
```


## 三、主要创新点

### 1. 策略聚类（核心创新）
- **每个窗口生成3-5个候选策略**（分段加权求和策略），在测试集上回测选出best
- **收集所有窗口的best策略**，通过K-Means聚类归纳为3种典型预测模式
- 最终规则包含**多阶段技能权重组合**，而非单一技能选择

### 2. 固定参数 + 策略归纳（替代参数调优）
- 不再优化三个阈值参数，固定为最优值 `(0.70, 0.15, 0.20)`
- 将优化重点从“参数调优”转向“策略归纳”
- 输出的规则是**可执行的预测策略**，而非单纯的参数值

### 3. 多阶段加权策略
- 每个策略由多个阶段(stages)组成
- 每个阶段指定：预测步数 + 技能权重字典
- 例如：前3步用`chunk_ensemble:0.7 + calendar:0.3`，后4步用`multi_resolution:0.6 + naive:0.4`

### 4. 策略缓存机制
- 避免每个窗口重复调用LLM生成策略
- 基于配置哈希生成缓存键（数据集名、窗口大小、步长、固定参数）
- 第二次运行直接从缓存加载，速度提升**80%以上**

### 5. 规则驱动预测（跳过LLM）
- 匹配规则后直接使用固定策略预测，**完全跳过LLM调用**
- 推理速度从**10-30秒降至毫秒级**
- 支持 `--use_rules` 参数切换模式

### 6. 轨迹记录与回放
- `predict_with_trajectory` 记录每一步决策（技能权重 + 重决策间隔）
- 策略生成时参考原始轨迹，确保策略可解释且与原始行为一致

### 7. 技能状态卡 + 硬过滤
- 每个技能自带状态卡(when_to_use/when_not_to_use)
- `SkillMatcher` 基于特征硬过滤，减少LLM候选池
- 保证技能选择的可靠性和可解释性


## 四、运行指令

### 1. 完整调优 + Generator（生成规则）
```bash
python -m experiments.autotune.main --dataset melbourne_temp --horizon 7 --verbose --compare
```

### 2. 仅调优（不对比）
```bash
python -m experiments.autotune.main --dataset melbourne_temp --horizon 7 --verbose
```

### 3. 使用规则进行预测（推荐，最快）
```bash
python run_benchmark.py --dataset melbourne_temp --min_train_size 600 --horizon 7 --use_rules storage/autotune_results/generated_rules.json
```

### 4. 不使用规则（默认，调用LLM）
```bash
python run_benchmark.py --dataset melbourne_temp --min_train_size 600 --horizon 7
```

### 5. 查看缓存状态
```bash
# 查看策略缓存
dir storage\autotune_results\strategy_cache\

# 查看预测缓存
dir storage\autotune_results\cache\
```

### 6. 清除缓存（重新生成）
```bash
rmdir /s /q storage\autotune_results\strategy_cache
rmdir /s /q storage\autotune_results\cache
rmdir /s /q storage\autotune_results\window_data
```

### 7. 查看结果
```bash
# 查看规则文件
cat storage\autotune_results\generated_rules.json

# 查看采集数据
cat storage\autotune_results\collected_windows.csv

# 查看日志
dir storage\autotune_results\logs\
```


## 五、配置说明

### config.yaml 核心配置
```yaml
datasets:
  - name: "melbourne_temp"
    window_sizes: [600]          # 固定窗口大小
    step_size: 150               # 滑动步长
    horizon: 7

fixed_params:                    # ★ 固定参数，不再调优
  long_skill_force_threshold: 0.70
  route_bonus_long_skill: 0.15
  residual_acf_threshold: 0.20

llm:
  model: "glm-4"
  max_tokens: 4000
```


## 六、性能对比

| 模式 | LLM调用 | 耗时 | MASE |
| :--- | :--- | :--- | :--- |
| 原始LLM预测（无规则） | 3-4次/步 | 10-30秒 | 0.65~0.70 |
| 规则驱动预测（有规则） | 0次 | <100ms | 0.65~0.70 |
| 调优采集（21窗口） | 21次 | 3-5分钟 | — |
| 缓存加载（二次运行） | 0次 | <1分钟 | — |

---

## 七、总结

| 维度 | 描述 |
| :--- | :--- |
| **输入** | 历史时间序列数据 + 28个预测技能 |
| **输出** | 规则文件(generated_rules.json)：包含条件化策略和多阶段技能权重 |
| **核心价值** | 自动化发现多阶段加权预测策略，替代LLM决策，加速推理100倍 |
| **适用场景** | 需要快速推理的生产环境、成本敏感场景、离线批处理 |