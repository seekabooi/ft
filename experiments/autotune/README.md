# 📁 SPLS++ 项目完整总结（稳定性修复版）

---

## 一、目录结构（标明文件作用）

```
futureTime_autoQian_055_middle/
│
├── llog/                                    # ★ 策略JSON和运行日志存放目录
│   ├── logs/
│   │   ├── spls_autotune_*.log              # 主流程运行日志
│   │   ├── policy_evolution_*.log           # 策略演化日志
│   │   └── token_stats_*.txt                # Token 统计
│   ├── refined_policies.json                # ★ 最终策略 JSON
│   ├── policies_round_*.json                # 演化快照
│   └── diagnostic_*.json                    # ★ 诊断日志（策略匹配详情）
│
├── experiments/
│   └── autotune/
│       │
│       ├── 【主入口】
│       ├── main.py                          # ★ SPLS 主入口（稳定性修复版）
│       │                                   #    关闭 validation-driven filtering
│       │                                   #    验证集只做报告
│       ├── iterative_refiner.py             # ★ Policy Evolution Engine（冻结版）
│       │                                   #    merge/split/patch/retire 全部冻结
│       ├── spls_loop.py                     # ★ SPLS 主循环核心（稳定性修复版）
│       │                                   #    Soft Mixture 熵约束 ≥0.7
│       │                                   #    EMA 更新 only
│       │                                   #    禁用 LLM 梯度注入
│       │
│       ├── 【核心策略层】★ 全部 Policy 化
│       ├── skill_policy.py                  # ★ 唯一核心对象：SkillPolicy
│       │                                   #    含 Soft Mixture + Error Memory + Internal State
│       ├── skill_card.py                    # ★ Legacy Adapter（降级为适配器）
│       ├── rule_engine.py                   # ★ Policy Execution Engine（优先匹配有条件策略）
│       │
│       ├── 【策略归纳层】★ 稳定性修复
│       ├── inducer.py                       # ★ Skill Policy Induction Module
│       │                                   #    策略数量 5-12 条（保持多样性）
│       │                                   #    不使用 merge 压缩
│       │                                   #    关闭 validation-driven filtering
│       │                                   #    动态聚类，保留所有策略
│       ├── cluster.py                       # ★ Latent Policy Space Partition（增强特征）
│       ├── meta_cluster.py                  # ★ Policy Compression Layer（暂未使用）
│       │
│       ├── 【验证与诊断层】
│       ├── validator.py                     # ★ Policy Evaluation Oracle
│       ├── performance_auditor.py           # ★ Policy Diagnostic Module
│       ├── hard_refiner.py                  # ★ Local Policy Repair Engine（暂未使用）
│       ├── diagnostic_logger.py             # ★ 诊断日志模块
│       │
│       ├── 【MMSkills 对齐层】
│       ├── state_encoder.py                 # ★ Regime-aware Encoder（简化版）
│       │                                   #    只保留 learned_embedding
│       │                                   #    移除手工特征（减少过拟合）
│       ├── causal_skill_graph.py            # ★ Policy Influence Graph
│       ├── skill_lifecycle.py               # ★ 全部 Policy 化
│       ├── reflection.py                    # ★ Diagnostic Feedback Generator
│       │
│       ├── 【元技能】（降级为工程保障）
│       ├── meta_skills.py                   # ★ System Robustness Modules
│       │
│       ├── 【工具层】
│       ├── utils.py                         # 通用工具函数
│       ├── cache_manager.py                 # 缓存管理
│       ├── visualizer.py                    # 可视化
│       ├── config.yaml                      # ★ 系统级配置（稳定性修复版）
│       │                                   #    LLM gradient injection: disabled
│       │                                   #    online_evolution: disabled
│       │                                   #    min_policies: 5, max_policies: 12
│       │                                   #    entropy_min: 0.7
│       │                                   #    model: glm-4.5-air
│       │
│       └── __init__.py                      # 包初始化
│
├── src/                                     # 底层技能与 Agent
│   ├── agents/
│   │   ├── llm_planner.py                   # LLM Agent（预测 + 轨迹）
│   │   ├── llm_client.py                    # ★ LLM客户端（含Token统计）
│   │   └── llm_prompts.py                   # Prompt模板
│   ├── skills/
│   │   ├── registry.py                      # 技能注册表
│   │   ├── data_profiler.py                 # 数据特征提取
│   │   ├── skill_matcher.py                 # 技能匹配
│   │   └── [28个具体技能实现...]
│   ├── dataset/
│   │   ├── registry.py                      # 数据集注册
│   │   └── loader.py                        # 数据集加载
│   └── tasks/
│       └── instance.py                      # 任务实例定义
│
├── storage/                                 # 运行时数据（采集数据、可视化等）
│   └── autotune_results/
│       ├── collected_windows.csv            # 采集结果（含split标签）
│       ├── window_data/                     # 窗口数据（.pkl）
│       ├── cache/                           # 预测缓存
│       ├── strategy_cache/                  # 策略缓存
│       ├── visualizations/                  # 可视化图表
│       └── reports/                         # 文本报告
│
└── requirements.txt                         # 依赖列表
```


## 二、处理逻辑流程图（文字描述）

### 总体流程（6个阶段）

**阶段一：状态窗口生成（数据采集）**
1. 加载原始时序数据（melbourne_temp，3650个数据点）
2. 计算数据周期（365天）和 MASE 缩放因子
3. 滑动窗口采集：窗口大小600，步长150，共21个窗口
4. 对每个窗口提取特征，调用 LLMPlannerAgent 预测，计算 MASE
5. 数据划分：train=14, val=3, test=4（70%:15%:15%）
6. 输出：`collected_windows.csv` + 窗口数据(.pkl)

**阶段二：策略归纳（Skill Policy Induction）**
1. 使用 train+val 窗口（共17个）
2. 对每个窗口：LLM 生成 2-3 个候选策略 → 回测评分 → 选出最优策略
3. 动态聚类分割 Policy Space：根据窗口数自动调整簇数（3-6个）
4. 对每个分区生成差异化条件（使用多个特征，不同方向）
5. ★ 策略数量确保在 5-12 条之间（保持多样性）
6. ★ 不使用 merge 压缩
7. ★ 关闭 validation-driven filtering（验证集只做报告）
8. 输出：`refined_policies.json`

**阶段三：Soft Mixture 预测（稳定性修复版）**
1. 状态编码：使用简化的 learned_embedding（移除手工特征）
2. 计算策略得分：π(k|s) = softmax(score_k / T)
3. ★ 熵约束：H(π) ≥ 0.7（如熵过低，自动提高温度）
4. Top-K 截断（K=4）
5. 加权混合预测：y_t = Σ_k π(k|s) · f_k(x_t)
6. 输出：最终预测

**阶段四：策略评估（只做报告）**
1. 加载 collected_windows.csv
2. 过滤 split='val' 的窗口（共3个）
3. 对每个验证窗口：检索策略 → 执行预测 → 计算 MASE
4. ★ 只做评估，不做筛选或过滤
5. 输出：评估报告 + 可视化

**阶段五：Ablation Study**
1. 过滤 split='test' 的窗口（共4个）
2. 对比无策略 vs 有策略的 MASE
3. 输出：改善率

**阶段六：诊断日志**
1. 记录策略匹配详情（每个窗口匹配到的策略）
2. ★ 所有演化机制已冻结（merge/split/patch/retire 全部禁用）
3. ★ EMA 更新 only（平滑稳定）
4. ★ LLM 梯度注入已禁用


## 三、主要创新点

### 核心贡献（6个）

**贡献1：Skill Policy Representation**
- 唯一核心对象，统一所有旧概念
- 公式：π_k(s_t) → a_t

**贡献2：State-conditioned Soft Policy Mixture**
- 从硬切换（argmax）→ 软加权混合
- 公式：y_t = Σ_k π(k|s_t) · f_k(x_t)

**贡献3：Regime-aware Encoding**
- 多维度状态编码 + Learned Regime Embedding

**贡献4：Continuous Evolution（EMA only）**
- π ← (1 - α)π + α·∇L
- 平滑稳定更新

**贡献5：Confidence-calibrated Selection**
- π̃ = π × confidence_k(s)

**贡献6：Entropy-constrained Mixture**
- ★ H(π) ≥ 0.7（防止策略塌缩）


## 四、运行指令

### 1. 完整 SPLS 主循环

```bash
# 清空缓存
rmdir /s /q storage\autotune_results\cache
rmdir /s /q storage\autotune_results\strategy_cache

# SPLS 主循环
python -m experiments.autotune.main --dataset melbourne_temp --horizon 7 --verbose --compare
```

### 2. 策略评估（仅报告，不演化）

```bash
python -m experiments.autotune.iterative_refiner --dataset melbourne_temp --horizon 7 --rounds 3 --verbose
```

### 3. 其他数据集

```bash
python -m experiments.autotune.main --dataset sunspots --horizon 12 --verbose --compare
```

### 4. 查看诊断日志

```bash
type llog\diagnostic_*.json
```


## 五、遇到的主要问题与解决

### 问题1：策略条件全部为 `"True"`（已修复）
**现象**：所有规则条件退化为 `"True"`，RuleEngine 无法区分  
**原因**：`_generate_condition` 中 `std` 阈值过高（0.05）  
**解决**：降低阈值到 0.01，添加单样本簇条件生成逻辑

### 问题2：聚类不平衡（已修复）
**现象**：15 个策略挤在一个簇，单样本簇被赋为 `"True"`  
**原因**：聚类特征维度少，缺乏区分度  
**解决**：增加 10+ 个区分度特征

### 问题3：效果从 +19.92% 暴跌到 -52%（本次修复）
**现象**：测试集改善 -52.36%，所有验证窗口使用同一条策略  
**原因**：Policy Space Collapse（策略被压到 3-4 条，过粗）+ False Soft Mixture（实际是 hard routing）+ Unstable Evolution（LLM + gradient + merge 三重扰动）  
**解决**：
1. 策略数量从 3-4 条增加到 5-12 条
2. 关闭 LLM gradient injection
3. 冻结 merge/split/patch/retire
4. Soft mixture 加入熵约束 ≥0.7
5. 简化 state encoder（移除手工特征）
6. 关闭 validation-driven filtering
7. EMA only update
8. 模型切换为 glm-4.5-air

### 问题4：语法错误（已修复）
**现象**：`SyntaxError: closing parenthesis ')' does not match opening parenthesis '{'`  
**原因**：f-string 内嵌套花括号导致括号不匹配  
**解决**：使用 `.format()` 替代 f-string


## 六、当前状态

| 模块 | 状态 |
|------|------|
| Policy 数量 | 5-12 条（动态） |
| Soft Mixture | 熵约束 ≥0.7 |
| Evolution | 冻结（仅 EMA） |
| LLM 梯度注入 | 已禁用 |
| Merge/Split/Patch | 已冻结 |
| Validation | 只做报告 |
| State Encoder | 简化版 |
| 模型 | glm-4.5-air |


## 七、一句话总结

> **SPLS 稳定性修复版：从"多机制叠加系统"收敛回"单学习目标系统"，通过熵约束防止策略塌缩，通过冻结演化机制消除扰动，通过增加策略数量保持表达力。**