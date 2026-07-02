# SPLS-RL v6 完整项目总结

---

## 一、目录结构与文件作用

```
futureTime_autoQian_055_middle/
│
├── experiments/autotune/                    # ★ 核心算法目录
│   │
│   ├── 【主入口与核心控制】
│   ├── main.py                              # ★ 程序入口，解析命令行，启动 SPLSAutoTuner
│   ├── tuner_core.py                        # ★ 核心类 SPLSAutoTuner，编排采集/归纳/RL训练/评估
│   ├── tuner_train.py                       # ★ 训练循环（第1轮归纳 + 后续强制RL演化）
│   ├── tuner_eval.py                        # Ablation 并行评估（含 Top‑2 集成投票）
│   ├── tuner_patch.py                       # 事后补丁（困难窗口聚类后直接加入全局池）
│   ├── tuner_utils.py                       # 工具（配置打印、策略保存、模型检测）
│   │
│   ├── 【核心策略与RL层】
│   ├── skill_policy.py                      # ★ 唯一核心对象 SkillPolicy（含 θ 字段、稳定性评分）
│   ├── policy_graph.py                      # ★ 策略图谱（统计/记忆组织层，负责动态簇发现）
│   ├── spls_loop.py                         # ★ SPLS 主循环（RL预测 + θ Bias注入 + 策略采样）
│   ├── policy_distribution.py               # ★ 策略分布模型（管理 θ，执行自适应衰减更新）
│   ├── state_encoder.py                     # ★ 状态编码器（连续特征 + Regime 标签提取）
│   ├── rl_components.py                     # Reward / Advantage / Baseline 计算
│   ├── replay_memory.py                     # 经验回放缓存
│   ├── rule_engine.py                       # 策略执行引擎（硬匹配升级为软路由）
│   │
│   ├── 【策略归纳层】
│   ├── inducer.py                           # ★ Skill Policy Induction（窗口级并行）
│   ├── inducer_core.py                      # 归纳核心逻辑（聚类、多样性、语义描述生成）
│   ├── inducer_candidate.py                 # 候选生成 + LLM调用 + 技能有效性校验
│   ├── cluster.py                           # Policy Space 分割（动态聚类）
│   ├── prompts.py / prompts.yaml            # 提示词模板
│   │
│   ├── 【演化层】（已简化，退休禁用）
│   ├── iterative_refiner.py                 # 演化引擎入口
│   ├── iterative_refiner_core.py            # ★ run_round核心（退休已禁用，只保留Re-Induction）
│   ├── iterative_refiner_utils.py           # 困难窗口识别 + 缓存读取/写入
│   ├── evolution_controller.py              # 演化触发器
│   ├── merge_simulator.py                   # 合并模拟器
│   ├── retirement_mechanism.py              # ★ 退休机制（已禁用，返回空）
│   │
│   ├── 【验证与诊断】
│   ├── validator.py                         # 策略评估Oracle
│   ├── checkpointer.py                      # ★ 检查点管理器（断点续训）
│   ├── coverage_gap_analyzer.py             # 覆盖缺口分析
│   ├── confidence_calculator.py             # Wilson Score 置信度计算
│   │
│   ├── 【数据与缓存】
│   ├── collector.py                         # 状态窗口生成器
│   ├── build_cache.py                       # ★ 独立缓存构建脚本
│   ├── build_missing_cache.py               # 缓存补全脚本
│   │
│   ├── 【测试工具】
│   ├── test_semantic_vs_rl.py               # ★ 语义匹配 vs RL参数消融实验
│   ├── test_semantic_vs_rl_half.py          # ★ 半窗口快速测试版（跳过no_rule，含技能描述）
│   ├── test_theta_ablation.py               # ★ θ 分位数 Ablation 测试
│   ├── test_ablation_compare.py             # Ablation 对比测试（三模式）
│   ├── round_status_trend.py                # 轮次状态趋势分析
│   ├── plot_window_comparison.py            # 窗口MASE折线图绘制
│   └── diagnose_breakpoint.py               # 断点续跑诊断工具
│
├── src/                                    # 底层技能与Agent
│   ├── agents/
│   │   ├── llm_planner.py                   # LLM Agent（核心预测入口）
│   │   └── llm_client.py                    # ★ LLM客户端（Token统计、模型切换）
│   └── skills/                              # 30+ 技能实现（含 state_card）
│
├── storage/autotune_results/               # 运行时数据
│   ├── collected_windows.csv               # 所有窗口元数据
│   ├── window_data/                        # 窗口 .pkl 文件
│   └── cache/                              # 预测缓存
│
└── llog/run_*/                             # 运行日志
    ├── checkpoint.json                     # ★ 检查点
    ├── full_output.log                     # 完整终端输出
    ├── refined_policies.json               # 最终策略
    ├── round_*/                            # 各轮策略快照
    ├── window_results/                     # 窗口独立结果
    ├── rl_cache_b1.pkl / b2.pkl            # ★ B1/B2子集预测缓存
    └── semantic_vs_rl_results/             # θ消融实验结果
```


## 二、文件调用关系（核心链路）

```text
main.py
  └── SPLSAutoTuner (tuner_core.py)
        ├── collector.py (采集窗口)
        ├── tuner_train.py (训练循环)
        │     ├── inducer.py (第1轮归纳 / Re-Induction)
        │     │     └── inducer_candidate.py (LLM调用+候选生成)
        │     ├── iterative_refiner_core.py (演化)
        │     ├── build_rl_cache() (缓存构建)
        │     ├── spls_loop.py (RL训练)
        │     │     ├── policy_distribution.py (θ管理)
        │     │     ├── skill_policy.py (策略对象)
        │     │     ├── state_encoder.py (状态编码)
        │     │     └── rl_components.py (Reward/Baseline)
        │     └── checkpointer.py (保存检查点)
        └── tuner_eval.py (Ablation评估)

★ 测试工具:
test_semantic_vs_rl_half.py
  └── SemanticVsRLTesterHalf
        ├── 硬编码 no_rule 前25窗口 MASE（跳过LLM）
        ├── 按语义分数排序 → 取 Top K 中 θ 最大
        ├── 调用 LLM 生成策略（含技能描述+特征注解）
        ├── 本地执行策略（复用 SkillPolicy.execute）
        └── 生成对比报告 + 柱状图 + 折线图
```


## 三、处理逻辑流程图

### 总体流程（退休已禁用）

```text
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    SPLS v6 总体流程（无退休版）                                  │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                  │
│  ╔══════════════════════════════════════════════════════════════════════════════════════════════╗ │
│  ║                              阶段 0：数据采集（一次性）                                      ║ │
│  ╚══════════════════════════════════════════════════════════════════════════════════════════════╝ │
│                                    │                                                              │
│                                    ▼                                                              │
│   ┌──────────────────────────────────────────────────────────────────────────────────────────┐  │
│   │  原始数据 → 滑动窗口(step=10, size=600) → 304个窗口                                      │  │
│   │  每个窗口：调用LLM预测 → 计算MASE → 保存为.pkl + CSV                                     │  │
│   └──────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                                              │
│                                    ▼                                                              │
│  ╔══════════════════════════════════════════════════════════════════════════════════════════════╗ │
│  ║                              阶段 1：策略归纳（第1轮）                                       ║ │
│  ╚══════════════════════════════════════════════════════════════════════════════════════════════╝ │
│                                    │                                                              │
│                                    ▼                                                              │
│   ┌──────────────────────────────────────────────────────────────────────────────────────────┐  │
│   │  A部分(50%窗口) → 逐窗口LLM生成候选(2个) → 聚类 → 初始策略池                            │  │
│   │  初始化 θ=0，PolicyDistributionModel，PolicyGraph（含动态簇发现）                       │  │
│   └──────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                                              │
│                                    ▼                                                              │
│  ╔══════════════════════════════════════════════════════════════════════════════════════════════╗ │
│  ║                         阶段 2：RL 演化优化（第2-24轮）                                     ║ │
│  ╚══════════════════════════════════════════════════════════════════════════════════════════════╝ │
│                                    │                                                              │
│         ┌──────────────────────────┼──────────────────────────────────────────────────┐          │
│         │                          │                                                  │          │
│         ▼                          ▼                                                  ▼          │
│   ┌─────────┐               ┌─────────┐                                        ┌─────────┐      │
│   │ 第2轮   │               │ 第3轮   │                      ...                │ 第24轮  │      │
│   │ B1(51窗)│               │ B2(51窗)│                                        │ B2(51窗)│      │
│   └────┬────┘               └────┬────┘                                        └────┬────┘      │
│        │                         │                                                  │          │
│        └─────────────────────────┼──────────────────────────────────────────────────┘          │
│                                  │                                                              │
│                                  ▼                                                              │
│   ┌──────────────────────────────────────────────────────────────────────────────────────────┐  │
│   │                    单轮详细逻辑：Re-Induction → 增量缓存 → RL训练                         │  │
│   │   ★ 退休机制已完全禁用                                                                    │  │
│   │   ★ TRIAL策略冻结期2轮                                                                    │  │
│   │   ★ 分策略学习率：TRIAL×2.0, ACTIVE×0.5                                                  │  │
│   └──────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                                              │
│                                    ▼                                                              │
│  ╔══════════════════════════════════════════════════════════════════════════════════════════════╗ │
│  ║                              阶段 3：评估与 Ablation                                        ║ │
│  ╚══════════════════════════════════════════════════════════════════════════════════════════════╝ │
│                                    │                                                              │
│                                    ▼                                                              │
│   ┌──────────────────────────────────────────────────────────────────────────────────────────┐  │
│   │  Test部分 → θ分位数Ablation测试 / 语义匹配vsRL消融实验                                  │  │
│   └──────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```


## 四、主要创新点

| 序号 | 创新点 | 说明 |
|------|--------|------|
| 1 | **唯一核心对象 `SkillPolicy`** | 统一策略结构、状态条件、健康指标、置信度、错误记忆、**RL权重(θ)** 及稳定性评分 |
| 2 | **策略分布学习（Policy Gradient）** | `π(p\|s) = softmax(semantic + θ_bias)`，θ 通过 Policy Gradient 在线更新 |
| 3 | **三大稳定性 Patch** | Patch1(Evolution Soft Gate)、Patch2(Adaptive Decay)、Patch3(θ Bias Injection) |
| 4 | **预测缓存复用** | 按子集分离缓存（`rl_cache_b1.pkl`/`b2.pkl`），断点续传，旧缓存自动迁移 |
| 5 | **新策略试用期冻结** | `TRIAL` 策略前2轮仅能被探索选中且不更新 θ，积累足够数据后再参与竞争 |
| 6 | **分策略学习率** | `TRIAL` 策略使用高学习率（快速适应），`ACTIVE` 策略使用低学习率（保护已训练参数） |
| 7 | **质量门控** | Re-Induction 候选策略若 MASE 不低于池中最差策略则被过滤 |
| 8 | **概率补偿（抗稀释）** | 新增策略时，对旧策略 θ 加上 `log((N+K)/N)`，保持旧策略采样概率不变 |
| 9 | **动态簇发现** | 新策略离所有簇较远时自动创建新簇，支持新问题空间的发现 |
| 10 | **测试时弃用 θ** | 测试时完全弃用 θ，仅使用语义匹配，保护专精策略 |


## 五、每一轮的执行步骤及作用

### 第 1 轮：策略归纳（A部分）

| 步骤 | 操作 | 作用 |
|------|------|------|
| 1.1 | 加载 A 部分窗口（152个） | 数据准备 |
| 1.2 | 逐窗口 LLM 生成 2 个候选策略 | 利用 LLM 创造力生成多样化策略 |
| 1.3 | 技能有效性校验（过滤无效技能） | 确保策略可执行 |
| 1.4 | 策略聚类（Policy Space Partitioning） | 合并同类策略，控制池子规模 |
| 1.5 | 动态簇发现 | 新策略离所有簇较远时自动创建新簇 |
| 1.6 | 生成初始 `SkillPolicy` 对象 | 统一数据结构 |
| 1.7 | 初始化 θ=0，加入分布模型 | 所有策略地位平等 |
| 1.8 | 保存策略和检查点 | 持久化，支持断点续训 |

**目标**：生成覆盖不同场景的初始策略池，所有策略 θ=0。

---

### 第 2~24 轮：RL 演化（交替 B1/B2）

| 阶段 | 步骤 | 操作 | 作用 |
|------|------|------|------|
| **A** | 1.1 | **加载/构建子集缓存** | 检查 `rl_cache_{subset}.pkl`，若无则自动构建 |
| | 1.2 | **Re-Induction**（困难窗口比例>20%） | 发现短板并生成补丁 |
| | 1.2.1 | 识别困难窗口（MASE > avg_mase * 1.0） | 发现策略池短板 |
| | 1.2.2 | 调用 LLM 生成候选策略（最多4个） | 针对短板生成补丁 |
| | 1.2.3 | ★ **质量门控**：过滤候选策略 | 只引入有益策略 |
| | 1.2.4 | ★ **概率补偿**：对旧策略 θ 加补偿值 | 保持旧策略采样概率不变 |
| | 1.2.5 | 新策略加入池，状态=`TRIAL`，θ=0 | 新策略进入试用期 |
| | 1.2.6 | ★ **立即保存检查点** | 防止中断后新增策略丢失 |
| **B** | 1.3 | **增量缓存构建**（仅新策略） | 为新策略构建缓存 |
| **C** | 1.4 | **强制 RL 在线训练**（每个窗口） | 更新 θ 和策略统计 |
| | 1.4.1 | 状态编码 | 提取特征 |
| | 1.4.2 | 计算策略分布（含 θ、regime_bonus） | 获得采样概率 |
| | 1.4.3 | ★ 策略采样（UCB 探索 + 利用，含冻结） | 平衡探索与利用 |
| | 1.4.4 | 检查缓存，获取预测结果 | 避免重复计算 |
| | 1.4.5 | ★ Reward 裁剪（至 [-10,0]） | 消除极端值干扰 |
| | 1.4.6 | ★ Advantage 裁剪（至 [-10,10]） | 防止 θ 更新步长过大 |
| | 1.4.7 | ★ 分策略学习率更新 θ | 快慢结合，稳定训练 |
| | 1.4.8 | 更新 Baseline（EMA） | 平滑基线 |
| | 1.4.9 | 存储经验 | 用于后续分析 |
| | 1.4.10 | 更新策略统计 | 记录战绩 |
| **D** | 1.5 | **保存检查点** | 支持断点续训 |


## 六、运行指令

### 基础运行

```cmd
# 首次运行（采集 + 训练）
python -m experiments.autotune.main --dataset melbourne_temp --horizon 12 --verbose --compare

# 断点续训（从指定运行目录恢复）
python -m experiments.autotune.main --dataset melbourne_temp --horizon 12 --verbose --compare --resume llog/cs2

# 纯训练（无 Ablation，可断网）
python -m experiments.autotune.main --dataset melbourne_temp --horizon 12 --verbose --resume llog/cs2
```

### 缓存管理

```cmd
# 手动构建 B2 缓存
python -m experiments.autotune.build_cache --dataset melbourne_temp --horizon 12 --subset B2 --resume llog/cs2 --workers 12

# 清理缓存（完全重新采集）
rd /s /q storage\autotune_results\window_data
del /f storage\autotune_results\collected_windows.csv
```

### 测试与诊断

```cmd
# ★ 语义匹配 vs RL 参数消融实验（半窗口快速版，含技能描述）
python -m experiments.autotune.test_semantic_vs_rl_half --resume llog/cs2 --round 57 --workers 12

# θ 分位数 Ablation 测试
python -m experiments.autotune.test_theta_ablation --resume llog/cs2 --round 34 --workers 24 --exec-mode direct

# 轮次状态趋势分析
python -m experiments.autotune.round_status_trend --resume llog/cs2 --start-round 30

# 窗口MASE折线图
python -m experiments.autotune.plot_window_comparison --resume llog/cs2
```

### 查看日志

```cmd
tail -f llog\run_*\full_output.log
cat llog\run_*\comparison_report.txt
```


## 七、遇到的主要问题与解决方案

| 序号 | 问题 | 解决方案 |
|------|------|----------|
| 1 | JSON 解析失败 | 强化 `_repair_json` + `json5` 兜底 + 纯正则提取 |
| 2 | 窗口数量不足 | `step_size: 22→10`，`b_subset_count: 4→2`，`rounds: 16→24` |
| 3 | RL θ 数值爆炸（Softmax Collapse） | Patch 2（自适应衰减）+ Patch 3（tanh 归一化） |
| 4 | 检查点 `completed_rounds` 未更新 | 在 RL 训练完成后立即保存 |
| 5 | θ 和 Baseline 未持久化 | 修改 `checkpointer.save()` 支持这些字段 |
| 6 | Token 消耗成本高 | 关闭 `use_llm_judge`，提高 `hard_window_ratio_threshold` |
| 7 | 子进程日志丢失 | 子进程使用独立 `ProgressLogger`，增加 `failed_windows.json` |
| 8 | 缓存构建 pickle 错误 | 提取 `_process_cache_task` 为顶层函数 |
| 9 | 新策略加入导致概率稀释和 Baseline 抖动 | 试用期冻结、分策略学习率、质量门控、概率补偿、Reward/Advantage 裁剪 |
| 10 | 缓存构建耗时 | 利用并行 `ThreadPoolExecutor`，按子集分离缓存 |
| 11 | Windows 多进程内存爆炸 | 从 `ProcessPoolExecutor` 改为 `ThreadPoolExecutor` |
| 12 | 退休机制完全失效 | 彻底禁用退休机制，所有策略只保留 ACTIVE/TRIAL |
| 13 | 已退休策略无法复用 | 加载 checkpoint 时自动修复 DEPRECATED → ACTIVE |
| 14 | 专精策略因 θ 低被饿死 | 测试时弃用 θ，仅使用语义匹配 |
| 15 | 测试逻辑未按场景分簇 | 修改测试逻辑：按簇（策略空间）取 θ top 百分比 |
| 16 | GLM-4.7 模型导致 429 | 测试文件默认模型改回 `glm-4`，信号量控制并发 |
| 17 | 策略执行超时/卡死 | 直接复用训练阶段逻辑 `SkillPolicy.execute`，不加多余包装 |
| 18 | no_rule 数据丢失（results.json被覆盖） | `_save_intermediate_results` 改为合并写入，日志用追加模式 |


## 八、犯过的严重错误（深刻反思）

### 错误1：LLM 根本不知道技能是干什么的，我却一直在分析“为什么参考策略没用”

**表现**：一直分析 `no_rule` 效果更好、参考策略为什么有锚定效应，但从未触及根本原因——LLM 看到 `chunk_ensemble` 这个名字只能靠猜，不知道具体实现逻辑和适用场景。

**正确做法**：应该在 Prompt 中给 LLM 提供每个技能的【实现逻辑】和【适用场景】，让 LLM 基于真实信息决策，而不是瞎猜。


### 错误2：数据持久化的覆盖式写入

**表现**：`_save_intermediate_results` 用 `'w'` 模式直接覆盖写入，导致 `no_rule` 和 `semantic_top1` 的数据在重新运行时被清空。

**正确做法**：应该先读取已有数据，合并后再写入。日志文件用追加模式 `'a'`。


### 错误3：添加了不必要的超时控制导致死锁

**表现**：在 `_execute_strategy` 中加了 `concurrent.futures` 超时包装，导致策略执行卡死，而训练阶段直接用 `SkillPolicy.execute` 是正常的。

**正确做法**：直接复用训练阶段的逻辑，不加任何多余包装。


### 错误4：测试文件把 LLM 当数值预测器用

**表现**：`test_semantic_vs_rl` 直接调用 `agent.predict()` 让 LLM 输出预测值，而训练阶段是让 LLM 输出策略结构后在本地执行，逻辑不一致。

**正确做法**：统一为“LLM 生成策略 → 本地执行”的流程。


### 错误5：给 LLM 传的特征不带注解，让 LLM 猜字段含义

**表现**：特征用 `trend_strength: 0.204` 而不是 `trend_strength: 0.204 (趋势: 0弱-1强)`，LLM 不知道数值含义。

**正确做法**：关键特征带简短中文解释，帮助 LLM 理解。


### 错误6：技能列表只有名字，没有说明

**表现**：Prompt 中只列出技能名称（`chunk_ensemble, multi_resolution, naive, ...`），LLM 只能凭名字猜测。

**正确做法**：每个技能附带【实现逻辑】和【适用场景】，让 LLM 根据当前窗口特征判断该用哪个。


### 错误7：没有区分 no_rule 和其他模式的 Prompt 内容

**表现**：`no_rule` 和语义匹配模式共用同一个 Prompt 构建逻辑，导致 `no_rule` 也带注解，破坏了对照组的纯净性。

**正确做法**：`no_rule` 单独构建 Prompt，不带注解、不带参考策略。


### 错误8：测试窗口数砍半后，no_rule 数据没有复用

**表现**：半窗口测试文件中，`no_rule` 没有被硬编码加载历史数据，导致重新计算浪费 11 小时。

**正确做法**：从历史日志中提取 `no_rule` 前 25 个窗口的 MASE，硬编码到测试文件中，跳过 LLM 调用。


### 错误9：results.json 被覆盖，no_rule 数据丢失

**表现**：`run()` 方法一开始就强制重新计算所有模式，且写入时覆盖已有数据。

**正确做法**：先检查已有缓存，只计算缺失的模式，写入时合并已有数据。

---

**核心教训**：所有问题都源于一个根本——**LLM 在做决策时缺乏关键信息（技能的真实含义），而我一直在分析表象，没有解决本质问题。**