# 图切分与可融合范围识别:从静态规则到 Agent 驱动的全图优化

> 系统调研 PyTorch inductor 之外的可融合范围识别机制,并梳理基于 agent/LLM 自动生成融合算子 + pass 的研究前沿。
> 参考基线:`torch/_inductor/` 的 IR 类型 + `scheduler.can_fuse` 静态规则(详见 `inductor_fusion_breadth_analysis.md`)。
> 调研方法:anysearch 多源检索 + 关键论文全文抽取 + arXiv 综述交叉验证。信源标注见附录 A。

---

## 目录

- [0. TL;DR](#0-tldr)
- [1. 基线:inductor 的可融合范围识别是怎么做的](#1-基线inductor-的可融合范围识别是怎么做的)
- [2. 三类替代方案全景](#2-三类替代方案全景)
- [3. 静态规则层:MLIR / TVM / XLA / Apollo / Tiramisu / AKG](#3-静态规则层mlir--tvm--xla--apollo--tiramisu--akg)
- [4. ML 驱动层:NeuroCUT / TpuGraphs / TGraph / AStitch / Mirage](#4-ml-驱动层neurocut--tpugraphs--tgraph--astitch--mirage)
- [5. Agent 驱动层:AKG Kernel Agent / K-Search / KernelBench 综述](#5-agent-驱动层akg-kernel-agent--k-search--kernelbench-综述)
- [6. 与 inductor 的对照矩阵](#6-与-inductor-的对照矩阵)
- [7. 一个 Agent 驱动的可融合范围识别框架(提议)](#7-一个-agent-驱动的可融合范围识别框架提议)
- [8. 风险与未解问题](#8-风险与未解问题)
- [9. 结论](#9-结论)
- [附录 A. 信源与可信度](#附录-a-信源与可信度)
- [附录 B. 关键文献索引](#附录-b-关键文献索引)

---

## 0. TL;DR

inductor 的"可融合范围识别"是一套**硬编码、静态、规则驱动**的机制:**lowering 阶段**把算子路由到 IR 节点类型(`Pointwise`/`Reduction`/`Scan`/`TemplateBuffer`/`ExternKernel`),**scheduler 阶段**用 `can_fuse` + 评分(`scores_fusion`)做拓扑+成本评分——整套系统既看不到具体硬件特征的运行时反馈,也无法发现**新的可融合子图**(只能识别预设的 IR 类别)。

围绕这个痛点,业界已经走了三条路:

1. **静态规则扩展**(成熟、产线用):把 IR 类型换成更结构化的"算子特征"(TVM 的 `injective/reduction/complex-out-fusable/opaque`、XLA HLO 同款四类)、把规则换成 polyhedral 数学(Tiramisu、AKG)、把融合范围决策权下放到下游 loop optimizer(Apollo 的反馈机制)、把融合接口化成 `LinalgExtFusionInterface`(IREE/MLIR)。这条路**已经覆盖了 inductor 同等的功能**,但天花板低:无法发现 inductor 规则**之外**的融合机会。
2. **ML 驱动**(前沿探索):用 GNN + cost model 学习"哪些 layout/tile 配置更快"(TpuGraphs → TGraph);用 RL 做图切分(NeuroCUT 是图切分 SOTA,但目前主要服务于分布式计算而非融合);用代数化简 + superoptimization 在 kernel/block/thread 三层发现新的融合子图(Mirage 自动发现了 FlashDecoding 的 2.2× 加速版本)。这条路**能突破静态规则的天花板**,但要重训练 cost model,工程化门槛高。
3. **Agent/LLM 驱动**(2025 爆发):用多 agent 协作(Designer 产 unified sketch → Coder 写代码 → Verifier 验证 → Conductor 路由,典型如华为 **AKG Kernel Agent**、**K-Search**),把"识别可融合范围"和"生成融合算子 + pass"两步合一,反馈循环+search 探索,既能发现 inductor 规则外的融合,也能直接产出 kernel 代码。在 KernelBench 上,Agent 系统已经把 fast₁ 分数从 ~17%(KernelCoder)推到 70%(Kernel-Smith),并在 MoE 等复杂场景超过 FlashInfer 等专家优化。

**对你想做的事情的结论**:有比 inductor 硬编码规则更优的机制,且越靠近"代数等价 + 语义子图 + 多层 superoptimization"思路(MLIR/LinalgExtFusionInterface + Mirage μGraph + Apollo loop 反馈)以及"agent + cost feedback"(AKG Kernel Agent / K-Search),融合深度和发现范围都会显著超过 can_fuse 的硬编码评分。**推荐组合**:`MLIR LinalgExtFusionInterface` 作为"算子能力声明层" + `Apollo 的 loop-level 双向耦合` 作为"反馈层" + `Agent (Designer-Coder-Verifier)` 作为"范围识别 + 自动 codegen 层",在 NPU 上做端到端 fusion explorer。

---

## 1. 基线:inductor 的可融合范围识别是怎么做的

> 数据来源:`torch/_inductor/{lowering.py, ir.py, scheduler.py, choices.py}` + 已读参考文档 § 1-§ 5

inductor 的可融合范围识别是**两段式**:

### 1.1 Lowering 阶段:ATen/Prim → IR 类型

把算子路由到 IR 节点。**IR 类型决定融合性**(`get_reduction_type()` 与 producer/consumer 兼容性表):

| IR 类 | 融合性 | 来源 |
|---|---|---|
| `Pointwise(Loops)` | ✅ 任意融合 | `register_pointwise` 注册约 180 个 |
| `Reduction(Loops)` | ✅ 与 Pointwise prologue/epilogue 融合 | `make_reduction` 注册约 24 个 |
| `Scan(Loops)` | ⚠️ 自融,但**不可与 Reduction 融合** (`simd.py:2077`) | 5 个 cumsum 系 |
| `TemplateBuffer` | ⚠️ 仅 epilogue/prologue/multi-out 三路径 | mm/conv/~27 个 |
| `ExternKernel` | ❌ 基本不可融 | fallback/库算子 |

此外 `decomposition` 在 lowering 前把 386 个高层 op 拆成 ~173 个可融合 primitive,这是 inductor "可融合算子面广"的关键原因。

### 1.2 Scheduler 阶段:`can_fuse` + `scores_fusion`

`scheduler.can_fuse` 是一个**硬编码规则集合**(在 `torch/_inductor/scheduler.py`):

| Producer → Consumer | 可融? | 条件 |
|---|---|---|
| Pointwise + Pointwise | ✅ | 同 `(numel, rnumel=1)` |
| Pointwise + Reduction | ✅ | prologue/epilogue |
| Reduction + Pointwise | ✅ | 读归约结果 |
| Reduction + Reduction (同 shape) | ✅ | 兄弟归约 |
| Template + Pointwise | ✅ | matmul/conv epilogue |
| Pointwise + Template | ✅ | prologue fusion |
| SplitScan + Reduction | ❌ | `simd.py:2077` 显式拒绝 |
| Reduction + Template prologue | ❌ | consumer 不能是 reduction |
| 带 scatter/atomic mode 的 op | ❌ | 需同步写 |

`scores_fusion` 进一步给每对可融节点打分(buffer 共享收益、kernel 数量节省等),贪心挑高分融合。

### 1.3 inductor 机制的根本局限

参考文档 § 6 已列三个边界。这里把"为什么这些边界让 agent/ML 路线更有机会"具体化:

1. **静态、无运行时反馈**:`can_fuse` 只看 IR 类型 + 拓扑,**不感知**这个融合在 NPU 的真实开销。可能"语法可融"但 codegen 出 4 倍慢的 kernel(常见于 reduction+pointwise 跨边界 fusion,NPU 的 reduction 算子依赖特定 tile size)。
2. **预设 IR 类别不可扩展**:`Pointwise` 之外无法发明新类别。例如 `aten.lerp(start, end, weight)` 表达 `start + weight * (end - start)`——按 IR 拆是 3 个 Pointwise 可融,但**作为整体 lerp 模板**(AVX 有 `vfmadd231ps`)会有更好的 codegen。inductor 的 `can_fuse` 看到的是 3 个独立 Pointwise,**不会**把它们聚合成一个"lerp 模式"。
3. **无跨层融合语义**:inducer fusion 是 lowering 之后的图重写阶段,不感知 producer/consumer 算子的"语义整体性"(比如 conv+bn+relu 这种组合有专门的优化机会,需要模式匹配而非硬编码)。

---

## 2. 三类替代方案全景

下表先把所有调研到的方法按**核心机制**归类,后文按类逐个深入分析。

| 类别 | 代表系统 | 范围识别机制 | 反馈来源 | 是否自动 codegen | 适用阶段 |
|---|---|---|---|---|---|
| **静态规则** | MLIR Linalg + LinalgExtFusionInterface | Op 接口声明 + 规则 pattern match | 无 | 否(产生 fused IR 给 codegen) | graph → tiled loop |
| **静态规则** | TVM FuseOps + Relay | 四类算子分类 + pattern-driven rules | 无 | 否(交给 TOPI 模板) | graph → subgraph |
| **静态规则** | XLA HLO fusion | Injective/Reduction/ComplexOutFusable/Opaque + 多 pass | 无 | 是(LLVM / Triton IR) | HLO → kernel |
| **静态规则 + 反馈** | Apollo(MLSys 2022) | Graph partition + loop optimizer 上行反馈 | ✅ polyhedral 引擎反馈 | 部分 | DNN full graph |
| **Polyhedral** | Tiramisu / AKG | 多面体数学(约束求解) | 无(数学最优) | 是(C++/AscendC) | 算法 + schedule |
| **ML 驱动** | TpuGraphs / TGraph | GNN + ranking loss,学 layout/tile 选择 | 数据集(TPU 测量) | 否(选 config) | layout/tile 选择 |
| **ML 驱动** | AStitch(ASPLOS 22) | ML 优化的多维 fusion 空间 | ML cost model | 否 | memory-intensive fusion |
| **ML 驱动** | NeuroCUT(KDD 24) | GNN + RL 图切分 | reward-based | 否 | 通用图切分(目前非 fusion) |
| **Superoptimization** | Mirage(ASPLOS 24) | μGraph 三层 + algebraic pruning + 概率等价验证 | 运行时 benchmark | 是(CUDA) | kernel-level |
| **Agent/LLM** | AKG Kernel Agent(arXiv 2512.23424) | Designer → Unified Sketch → Coder → Verifier → Conductor | ✅ 编译/运行反馈 + island model | 是(Triton/TileLang/CUDA-C) | 整算子 fusion |
| **Agent/LLM** | K-Search | LLM 作 intrinsic world model + co-evolving search tree | ✅ performance 反馈 | 是(CUDA/Triton) | 整算子 fusion |
| **Agent/LLM** | KernelBench / Kernel-Smith / Kevin | RL + 代码生成 | reward model | 是 | 整算子 fusion |
| **Agent/LLM** | LLM4Kernel 综述(arXiv 2601.15727) | 系统化综述 | — | — | — |

> 表中"是否自动 codegen"指**是否会产出最终 kernel 代码**(而非只是 fusion plan);"反馈来源"指是否用到运行时/编译时反馈来**调整** fusion 决策。

按你提的"可融合范围识别"和"agent 自动生成融合算子"两个核心维度,这三类的能力定位如下:

- **静态规则**:范围识别能力 = 预设 IR/算子类别,极强;agent codegen = 弱。
- **ML 驱动**:范围识别能力 = 数据驱动,可发现隐藏规律;agent codegen = 弱(只产出 config,不产出算子)。
- **Agent/LLM**:两者都强——既能识别(LLM 推理出可融合结构),也能 codegen(LLM 输出 kernel)。

**给你的工程启示**:**不要选单条路线**。最实际的方案是用"静态规则的骨架 + ML/Agent 填充新发现的模式"——即 `MLIR LinalgExtFusionInterface`(规则骨架) + `Apollo 反馈环路`(让融合策略对下游负责) + `Agent` 作为"模式发现 + 算子生成"引擎。

---

## 3. 静态规则层:MLIR / TVM / XLA / Apollo / Tiramisu / AKG

### 3.1 MLIR LinalgExtFusionInterface

**来源**:LLVM Discourse 论坛 "Tile and fuse support" 帖子 + IREE 仓 `LinalgExtInterfaces.td`(https://github.com/iree-org/iree/blob/4bc495b112f77c9e84f48913b00b4154bfba1b8b/compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.td)

MLIR 的设计哲学与 inductor 完全相反:**transformation 与 heuristics 解耦**——`TilingInterface` / `tileConsumerAndFuseProducersUsingSCF` 提供"tile-and-fuse 这个动作",但**"该不该 tile-and-fuse"留给调用方**。具体调用方通过 `options` 结构的 callback 控制哪些 op 被 fuse、什么 induction variable 维度、是否避免冗余计算。

```cpp
// 概念性示例(非 MLIR 代码,仅示意)
// "tile and fuse" 调用方传入 callback 决定哪些 op 一起 fuse
scf::tileConsumerAndFuseProducers(
    consumerOp,
    options.setFuseControlFn([&](OpResult producer) -> bool {
      // 任意用户自定义逻辑:依赖分析、cost model、agent 推理...
      return shouldFuseWithConsumer(producer, consumerOp);
    })
);
```

LinalgExtFusionInterface 是 IREE 提出的**专门为 micro-kernel 设计的 fusion 接口**(同时也是这次调研的关键发现之一,见 § 6)。

**与 inductor 的对比**:

| 维度 | inductor | MLIR Linalg(+ Ext) |
|---|---|---|
| 融合决策者 | scheduler 硬编码规则 | 用户在 callback 中任意编码 |
| 范围识别粒度 | 节点对(producer × consumer) | 整个 tile-and-fuse 区间,可表达多层 fusion |
| 是否可表达"复杂 fuse 策略" | 否,只能预设表 | 是,可结合 ML cost model |
| 跨层融合支持 | 有限(主要 pointwise+reduction) | 强(`tileConsumerAndFuseProducers` 支持跨 block 嵌套) |

> 📖 **来源**:LLVM Discourse 论坛 "Tile and fuse support"(https://discourse.llvm.org/t/tile-and-fuse-support/84389),Mahesh Ravishankar 等 MLIR 维护者直接回复;IREE `LinalgExtFusionInterface` 源码。

### 3.2 TVM FuseOps + Ansor + MetaSchedule

**来源**:Apache TVM 官方文档 "Ansor: Generating High-Performance Tensor Programs for Deep Learning"(OSDI 2020, Zheng et al.) + Apache TVM 官方介绍 + MetaSchedule RFC 0005

TVM 经历了三个时代:

1. **AutoTVM**(2018):基于模板 + 手工 schedule primitives;每个 op 需要专家写模板。15k 行模板代码。
2. **Ansor / AutoScheduler**(OSDI 2020):**自动 partition + 模板无关**。用 `Relay FuseOps` pass 把模型分成小 subgraph,再用 cost-model 引导的 evolutionary search 生成 schedule。
3. **MetaSchedule / AutoTensorIR**(NeurIPS 2022):用**概率程序**(probabilistic programs)统一描述 schedule 空间,进化算法搜索。

**核心融合识别机制**(TVM OSDI 2020 原文):

> "We recognize four categories of graph operators: injective (one-to-one map, e.g. add), reduction (e.g., sum), complex-out-fusable (can fuse element-wise map to output, e.g., conv2d), and opaque (cannot be fused, e.g., sort). We provide generic rules to fuse these operators. Multiple injective operators can be fused together into another injective operator. A reduction operator can be fused together with input injective operators."

**这与 XLA 完全一致**(下面 § 3.3)。也是 inductor 4 类 IR 节点(`Pointwise`/`Reduction`/`Scan`/`ExternKernel`)的"祖师爷"。区别是 inductor 是 PyTorch-specific,TVM/XLA 是 framework-agnostic。

**Ansor 的额外能力**(超过 inductor):
- **Partition(切分)+ Schedule(融合) 联合优化**:Ansor 的 task scheduler 用 cost model 分配每个 subgraph 的优化时间。
- **Cost model learned from measurements**:ML-based(gradient boosted trees) cost model 预测每个 candidate 的执行时间。

**与 inductor 的对比**:

| 维度 | inductor | TVM Ansor/MetaSchedule |
|---|---|---|
| Op 分类粒度 | 9 个 IR 类(细) | 4 类(粗) |
| 是否探索 schedule | 否(IR 硬编码 codegen 路径) | ✅ 进化搜索 schedule |
| Cost model | 静态启发式 + 选择模板 | ✅ learned cost model + 测量反馈 |
| 自动 codegen | 模板 codegen(Triton/C++) | 同上 |
| 范围识别灵活性 | 仅 IR 类型驱动 | partition + 4 类 + cost model 联合 |

### 3.3 XLA HLO Fusion(Google/ML Perf 团队的实战)

**来源**:Operator Fusion in XLA: Analysis and Evaluation(Snider & Liang 2023, arXiv 2301.13062)+ OpenXLA 官方架构文档 + XLA:GPU 架构文档

XLA 把 fusion 称为 **"the single most important optimization"**,因为 GPU 工作负载绝大多数是 memory-bound。HLO fusion 设计原理与 TVM FuseOps **完全同款**:把 op 分四类——Injective / Reduction / Complex-out-fusable / Opaque。

> Source: "Fusion is XLA's single most important optimization, which groups multiple operations (e.g. addition into exponentiation into matmul) to a single kernel."
> — OpenXLA XLA:GPU Architecture

XLA 还有两个**独有的**特性:

1. **`CanRunConcurrently()`**:判断两个 HLO 是否可并发(stream-level fusion),用于 pipeline 化。
2. **library vs codegen 选择**:对很多常见 op,优先用 cuBLAS/cuDNN/cuBLASLt(verified fast),但这样会**阻止更复杂的 fusion**。XLA 在 GPU 后端会用 Triton 作为更复杂 fusion 的 codegen 层。

**与 inductor 对比**:XLA 与 inductor 在 fusion 机制上几乎**结构同构**(都是"算子分类 + 规则 + codegen"),只是 inductor 更细(9 个 IR 类),XLA 更粗(4 类)。XLA 的优势是它的优化 pipeline 更完整(DCE/CSE/buffer 分析/SPMD partition 都在 fusion 前后联动),而 inductor fusion 是相对独立的 scheduler 阶段。

### 3.4 Apollo:Graph Partition + Loop Optimizer 双向耦合

**来源**:Apollo: Automatic Partition-based Operator Fusion through Layer...(MLSys 2022, https://proceedings.mlsys.org/paper_files/paper/2022/file/e175e8a86d28d935be4f43719651f86d-Paper.pdf)

Apollo 是这次调研中**最有意思的"反向耦合"思路**——绝大多数 fusion 框架是单向的(graph engine 决定 partition → 交给 loop optimizer),Apollo 让 loop optimizer **向上反馈**给 graph engine:

> "APOLLO enables the upward feedback from the downstream loop optimizer, enforcing the graph engine to regenerate partition patterns amenable to the downstream pass."

**机制**:

1. **Primitive/compound operator 抽象**:compound 是 primitive 的图,二者构成 subgraph。
2. **Graph-level node grouping + operator-level loop fusion** 联合搜索(而不是分开)。
3. **Polyhedral engine 反馈**:loop fusion 阶段(基于多面体数学)把"不可融合的组合"反馈回 graph engine,迫使其**重新生成 partition**(而不是放弃融合)。
4. **Piecewise compilation**:把大 graph 切成 piece,每个 piece 独立编译,**解决 AKG 长编译时间问题**。

**性能**:在 GPU 训练 workload 上,Apollo 比 TensorFlow 高 **1.86×**,比 XLA 高 **1.37×**;在多 GPU 上比 TF 高 **1.96×**,比 XLA 高 **1.18×**;在 NPU 类 DSA 上,**改善 vendor 框架 19.7%**。

**与 inductor 对比**(关键差异):

| 维度 | inductor | Apollo |
|---|---|---|
| 单向 vs 双向 | 单向(lowering → scheduler) | **双向**(loop optimizer 反馈 graph engine) |
| 是否考虑下游不可融性 | 否 | ✅ 是 |
| 解决不可融的策略 | 放弃融合 | 重新 partition |
| 编译时间 | 低(模板 codegen) | 高(polyhedral),用 piecewise 缓解 |

**对你想做的事情的启发**:Apollo 的"反馈环路"非常适合用 agent 实现——agent 可以做"下游模拟器",告诉 graph partitioner "如果按这个 partition 走,下游 codegen 会出问题,请你重切"。

### 3.5 Tiramisu + AKG:Polyhedral 派

**来源**:Tiramisu Compiler 官网 + Tiramisu 论文(arXiv 1804.10694,CGO 2019)+ MindSpore AKG README + AKG 论文(PPoPP 21)

**Polyhedral 派的核心思想**:把 fusion 完全数学化,变成**整数线性规划(ILP)**。给定一组循环,问"哪些迭代域能 fuse?",答案是基于依赖向量、变换矩阵的数学约束求解。

- **Tiramisu**:用 C++ API + 多面体数学,生成多平台代码;已支持稠密 + 稀疏 + data parallel。
- **AKG(MindSpore / Huawei)**:基于多面体模型 + auto-schedule;目标是昇腾 NPU,生成 AscendC kernel。

**优势**(理论上):
- 数学最优(不存在"启发式规则漏掉的最优解")
- 表达力强(可处理非仿射访问模式,这是 MLIR Linalg 的限制)

**劣势**(实战中):
- **长编译时间**(Apollo 论文原文:"long compilation time is still inevitable in AKG")。
- **可扩展性差**:每加一个新 op,要手工写 schedule templates(15k 行的 TVM 模板问题)。

**与 inductor 对比**:Tiramisu/AKG 是**纯规则 + 数学最优**路线,**完全静态、无运行时反馈**,与 inductor 是同一类(但更数学化)。对你想做的 agent 路线**不是核心替代**,不过 AKG 多面体引擎的"成本估计"可以当 agent 的 cost oracle。

---

## 4. ML 驱动层:NeuroCUT / TpuGraphs / TGraph / AStitch / Mirage

### 4.1 NeuroCUT(GNN + RL 图切分)

**来源**:NeuroCUT: A Neural Approach for Robust Graph Partitioning(KDD 2024,arXiv 2310.11787,Manchanda et al.) + 作者 LinkedIn 公告

NeuroCUT 是**图切分问题**的 SOTA(击败 METIS 等传统启发式),用 GNN + 自回归 RL 框架 + 位置信息编码;支持**不可微目标**(切分平衡、模块度、conductance 等多目标);**可泛化到未见过的分区数**(训练时 K=4,推理时 K=10 也行)。

> Source: "In this study, we Learn to Solve the Graph Partitioning problem on a diverse set of partitioning objectives, encompassing both differentiable and non-differentiable types. Our framework combines Graph Neural Network and Reinforcement learning to tackle this problem. Further, our proposed method also demonstrates generalizability to an unseen number of partitions at inference time."

**对 fusion 的价值**:

- NeuroCUT 原文**不直接做 fusion**,但其思路可迁移:**把 fusion 子图当作图的 partition**,目标是让每个 partition(融合子图)在 NPU 上的开销最小。
- 它的 RL 框架 + 多目标支持天然适合"fusion 多目标优化"(kernel 数减少 + 内存节省 + 编译时间不爆炸)。

**与 inductor 对比**:

| 维度 | inductor | NeuroCUT-based fusion |
|---|---|---|
| 目标 | 单目标(启发式评分) | 多目标(可学习) |
| 切分策略 | 静态,基于 IR | RL policy,基于 GNN |
| 泛化性 | 硬编码,新 op 要改 | 训练后可泛化到新 partition 数 |
| 训练成本 | 0 | 高(需训练 GNN + RL policy) |

### 4.2 TpuGraphs + TGraph(GNN 选 config)

**来源**:TpuGraphs(Phothilimthana et al., NeurIPS 2023 Datasets & Benchmarks)+ TGraph(Khizbullin et al., arXiv 2405.16623v2)

TpuGraphs / TGraph 是 Google + KAUST 联合推出的**大规模 XLA HLO graph 配置搜索数据集**(>10 万种配置/架构)和模型。

**任务**:给定一个 XLA HLO 图 + 可配置节点(Convolution / Dot / Reshape 的 layout),预测不同 layout 配置的 runtime,从中选出最优。

**TGraph 架构**:
- GNN backbone:GraphSAGE
- 配置 cross-attention:在 batch 维度做 attention(让模型显式对比不同配置)
- Channel-wise self-attention(类似 SE-Net)
- Pairwise Hinge Loss(ranking loss)

**结果**:在 TpuGraphs 4 个 layout collection 上 4/5 SOTA,mean Kendall's τ 从 29.6% → **67.4%**。

**对 fusion 的价值**:

- **不是直接做 fusion**,而是给 fusion 之后的 codegen 选**最优配置**(tile size / layout)。
- 这正好是 inductor `choices.py` 做的事——但 inductor 的 choices 是**离线模板** + `pick_best_of`,而 TGraph 是**端到端学习**。
- 推测潜力:用 TGraph 思路训一个"fusion 边界选择 cost model"——给定一个 fusion plan,预测 NPU runtime。

**与 inductor 对比**:

| 维度 | inductor `choices.py` | TGraph |
|---|---|---|
| 配置空间 | 几十个手写 heuristic | 10万+ 测量值 |
| 选择策略 | 启发式 + 模板 | GNN + ranking |
| 训练数据 | 离线手工 | 大规模测量 |

### 4.3 AStitch(ML 多维 fusion 空间)

**来源**:AStitch: Enabling A New Multi-Dimensional Optimization Space...(ASPLOS 22, https://dl.acm.org/doi/10.1145/3503222.3507723)

AStitch 解决 **memory-intensive 算子**(如 attention,elementwise/reduction 混合)的 fusion 问题。**关键贡献**:**开拓了"多维 fusion 空间"**——传统 fusion 只考虑 producer × consumer 是否可融,AStitch 引入"沿哪个 iteration 维度 fuse"、"分几次 fuse"、"哪些 tile 一起 fuse"等多维度,用 ML 选最优。

**对 fusion 的价值**:inductor 的 `can_fuse` 答的是**"两个 op 能不能 fuse"**(布尔问题);AStitch 答的是**"在多维 fusion 空间里选哪个 fusion 方案"**(多维优化问题)。两者解决的问题粒度不同,AStitch 显著更细。

### 4.4 Mirage:多层级 superoptimizer

**来源**:A Multi-Level Superoptimizer for Tensor Programs(Wu et al., arXiv 2405.05751,CMU 2024)

Mirage 是这次调研中**最接近"自动发现融合范围"思路的工作**。它的核心思想:

> "Mirage is the first multi-level superoptimizer for tensor programs. A key idea in Mirage is μGraphs, a uniform representation of tensor programs at the kernel, thread block, and thread levels of the GPU compute hierarchy."

**机制**:

1. **μGraph 三层表示**:每个 DNN 算子被表达为 kernel graph → block graph → thread graph,**统一代数变换**。
2. **代数等价 superoptimization**:用 abstract expression pruning + SMT solver(Z3)枚举所有代数等价的 μGraph(早期 FlashAttention 这种需要 600+ 行 Triton 手写代码,Mirage **自动发现了**它的 μGraph,且在 attention 上比 FlashDecoding **2.2× 更快**)。
3. **概率等价验证**:对每个候选 μGraph,用有限域随机测试做等价验证(PIT 算法推广,支持除法和 exp),错误概率可任意低。
4. **性能实测筛选**:对验证通过的 μGraph,实际测量 runtime 选最优。

**结果**:在 12 个 DNN benchmarks(包括 GQA、LoRA、MLP)上,**比现有系统(包括 hand-optimized FlashDecoding)最多快 3.5×**。

**与 inductor 的本质区别**:

| 维度 | inductor | Mirage |
|---|---|---|
| 范围识别 | 硬编码"哪个 op 能融进哪个 kernel" | 自动搜索"哪些 μGraph 等价组合" |
| 跨层级 | ❌ 单层(fused kernel 内) | ✅ kernel / block / thread 三层 |
| 是否发现新融合模式 | 否,只能识别预设 IR | ✅ 自动发现 FlashDecoding 级融合 |
| 搜索空间 | 小(~180 个 Pointwise × few patterns) | 极大(代数等价空间,SMT 剪枝) |
| 编译时间 | 秒级 | 分钟到小时(per subgraph) |

**对你想做的事情的启发**:Mirage 是**最有说服力的反例**,证明"inductor 风格的 can_fuse 硬编码无法达到 superoptimization 级别"。但 Mirage 的代价是**分钟级**编译时间,在 JIT 场景不可接受。

### 4.5 ML for ML Compilers(Princeton 综述)

**来源**:ML for ML Compilers(Princeton COS598D 2022 课件,Phothilimthana et al. 引用)

Princeton 课件给出了一个关键观察:

> "A common strategy partitions a graph into subgraphs according to the neural net layers, ignoring cross-layer optimization opportunities. **Empirical result: a regression of up to 2.6× and 32% on average across 150 ML models** by limiting fusions in XLA to be within layers."

这条引用非常关键——它证明**只按 NN layer 切分 fusion 范围**平均损失 32% 性能、最差 4.6×。这给 agent 路线提供了**最直接的业务理由**:inducer 的 layer-by-layer 切分(以及类似 XLA 的 layer-bounded 切分)**有 32% 性能损失空间**,而 agent 能跨层识别融合机会去**补回**这 32%。

---

## 5. Agent 驱动层:AKG Kernel Agent / K-Search / KernelBench 综述

### 5.1 LLM4Kernel 综述(arXiv 2601.15727,2026)

**来源**:Towards Automated Kernel Generation in the Era of LLMs(Yu et al., arXiv 2601.15727,BAAI 2026)

这是**目前最系统的 LLM-for-kernel 综述**,涵盖 50+ 个工作,按方法学分四大类:

| 类别 | 机制 | 代表工作 |
|---|---|---|
| **LLM 监督微调(SFT)** | 用 PyTorch-Triton 对齐数据 instruction tuning | KernelLLM、KernelCoder、InCoder-32B |
| **LLM 强化学习(RL)** | 多轮优化、credit assignment | Kevin、CUDA-L1/L2、SparseRL、AutoTriton、TritonRL、QiMeng-Kernel、Dr.Kernel、AscendKernelGen |
| **LLM Agent 协作** | 多 agent + planner + coder + verifier + memory | PEAK、AutoKernel、MaxCode、K-Search、DiffAgent、TritonX、KernelGen、AKG Kernel Agent |
| **外部记忆 + 知识检索** | 知识库、推理图、检索增强 | KernelEvolve、ReGraphT、KernelBlaster、EvoKernel、Kernel-Skill |

**对 fusion range 识别最相关的发现**:

- **闭循环反馈是关键**:LLM 单独使用是"一次静态推理",效果差;agent + 反馈循环才能 scale。这是 AkG Kernel Agent / K-Search 等的共同设计原则。
- **外部记忆避免"幻觉 API"**:LLM 编 kernel 经常编不存在的 CUDA API,知识库 + 检索是关键。
- **多 agent 优于单 LLM**:AkG Kernel Agent 论文直接说"single LLM to manage the entire optimization process... often struggles to simultaneously achieve correctness, performance, and portability"。
- **结构化中间表示(Unified Sketch)减少认知负担**:把"策略"和"实现"解耦。

### 5.2 AKG Kernel Agent(华为/Hunan Univ.,arXiv 2512.23424,2025-12)

**来源**:AKG Kernel Agent: A Multi-Agent Framework for Cross-Platform Kernel Synthesis(Du et al., arXiv 2512.23424v1)

这是**与 NPU + Triton/CUDA 直接相关**的关键工作——作者来自华为 + 湖南大学。

**核心架构**(四 agent 协作):

```
                    ┌─────────────────────────────────┐
                    │      Conductor (Orchestrator)   │
                    │  - 错误诊断 + 路由决策           │
                    └──────────────┬──────────────────┘
                                   │
        ┌─────────────┬────────────┼─────────────┬──────────────┐
        ▼             ▼            ▼             ▼              ▼
   Designer     Coder       Verifier     知识库/检索          反馈循环
   设计 Sketch  写代码     验证/性能      (RAG+DocSpec)
   + U.Sketch
```

**每个 agent 的职责**:

- **Designer**:分析算子 + 硬件 spec → 输出 **Unified Sketch**(硬件无关的中间表示,包含 alloc/load/store/compute primitives + `@llm_hint` 装饰器,例如 `"parallel"`/`"coreidx"`/`"pipeline"`/`"vectorize"`/`"unroll"`)。
- **Coder**:把 Unified Sketch 翻译成目标 DSL 代码(Triton / TileLang / AscendC / CUDA-C / C++),调用 API 文档 + 检索的相似算子样例。
- **Verifier**:三层验证——编译成功 / 数值正确 / 性能达标(Pass@k + speedup + fast_p 指标)。
- **Conductor**:智能路由——把"语法错误"路由回 Coder,把"算法错误"路由回 Designer,把"性能不达标"触发新一轮 Evolve。

**文档驱动集成(DocSpec)**:把"如何写 Triton kernel for Ascend NPU"等知识编码成**结构化文档**,新硬件/DSL 只需新写 DocSpec,不用改 agent 代码。

**岛模型(Island Model)优化**:在 Evolve 阶段,把 population 分 K=2 个岛,每轮每岛并行生成 P=4 个 candidate,每 M 轮 elite 跨岛迁移——**避免早熟收敛**,保多样性。

**结果**:
- KernelBench Level 1 五个 DSL-backend 组合(包含 Triton-CUDA、Triton-Ascend)上,**Pass@4 高达 100%(MatMul)、94.4%(Elementwise)**。
- 平均 **1.46× speedup over PyTorch Eager**。
- 在**静态 + 动态 shape** 上都能稳定通过(自建 198/214 ops benchmark,刻意避开 KernelBench 的 reward hacking)。

**对 fusion range 识别的直接价值**:AKG Kernel Agent 的 Unified Sketch 设计**本质上就是"可融合范围识别"的产物**——Designer 输出的是 fusion 计划(unified sketch),Coder 输出的是 fusion 后代码。换句话说,**这个 agent 已经把"识别可融合范围"和"生成融合算子 + pass"两步合一**。

**对你想做的事情的启发**(最实用):

1. 抄它的 4 agent 架构(Designer/Coder/Verifier/Conductor)作为骨架。
2. 抄它的 Unified Sketch 作为"范围识别的中间产物"——它硬件无关,正好可以挂到 NPU codegen。
3. 抄它的岛模型 + 文档驱动集成——让 agent 能 scale 到 NPU 不同型号。
4. 把它当基线——评估你自己的 agent 是否能超过 1.46×。

### 5.3 K-Search(KernelBench 上超越 FlashInfer 的 agent)

**来源**:K-Search: LLM Kernel Generation via Co-Evolving Intrinsic World Model(2026, KernelBench SoTA)

K-Search 的核心创新:**把 LLM 当作"intrinsic world model"**,在搜索过程中**持续更新对策略的信念**(类似 AlphaGo 的 MCTS)。

**机制**:

- **Search State `S_t`**:维护一棵搜索树,Closed(已评估)+ Open(待评估 frontier)。
- **每个 Open node 携带 Proposed Optimization tuple**(parent program + 优化意图 δ) + Priority Score V∈[0,1]。
- **Local Refinement**:选 action a_t,采样多个实现,stagnation condition(K=7 次无改进)时停止。
- **World Model Update**:对每次评估轨迹,LLM 做三种树编辑——Insert / Update / Prune——更新 frontier 优先级。
- **Co-Evolution**:world model 与 kernel 实现**共同进化**——既 refines program,也 refines search strategy itself。

**实验结果**(对比 OpenEvolve / ShinkaEvolve):
- GQA decode:76.0(K-Search) vs 44.2(OpenEvolve) vs 27.7(ShinkaEvolve),**1.7-2.7× 优势**。
- MLA prefill:57.4 vs 19.5 vs 11.3,**2.95-5.10× 优势**。
- MoE(最复杂的):44.1 vs 3.09,**14.3× 优势**。
- **GPUMODE TriMul leaderboard**:1030 µs(K-Search) vs 1074(人类最优 CUDA),**state-of-the-art**。

**对 fusion range 识别的价值**:K-Search 不用预设 fusion 规则,而是**让 LLM 自己发现"哪些优化(包括 fusion 范围)该 try"**——这是"开放式 fusion 探索"的 SOTA 证据。

### 5.4 KernelBench 系(Kernel-Smith / Kevin / KernelCoder)

**来源**:LLM4Kernel 综述(arXiv 2601.15727)对各工作的汇总

KernelBench 系列是 agent/LLM 编 kernel 的**标准 benchmark**,Level 1 ~100 个算子(简单 op),Level 2/3 是 end-to-end 模型。

关键 fast₁(KernelBench Level 1,表示生成的 kernel 超过 PyTorch Eager 的算子比例)历史进展:

| 系统 | 方法 | fast₁ |
|---|---|---|
| KernelCoder | SFT on curated dataset | 17% |
| ConCuR | SFT + reasoning traces | — |
| InCoder-32B | 3-stage SFT pipeline | 22.2% |
| CUDA-L1 | RL + LLM-as-judge | — |
| CUDA Agent | Skill-augmented RL | 99%(! 但 reward hacking 风险) |
| **Kernel-Smith** | **Stable evolution-oriented post-training** | **70%** |
| **K-Search** | **LLM-as-world-model + co-evolving search** | **SOTA across GQA/MLA/MoE** |

> ⚠️ 注:fast₁ 数字来自不同论文口径,直接对比有 caveat——某些数字是 KernelBench Level 1(100 ops),某些是子集。建议把 70%(Kernel-Smith)和 K-Search 的 SoTA 数字当"上限"参考,实际复现可能有 ±10% 浮动。

---

## 6. 与 inductor 的对照矩阵

下表把所有方法在一个统一的对照矩阵里:

| 维度 | inductor | MLIR LinalgExt | TVM Ansor | Apollo | Mirage | NeuroCUT | TGraph | AKG Kernel Agent | K-Search |
|---|---|---|---|---|---|---|---|---|---|
| **可融合范围识别机制** | IR 类型 + can_fuse 表 | callback + 接口声明 | 4 类算子 + FuseOps | partition + loop 反馈 | μGraph 代数等价搜索 | GNN+RL 图切分 | GNN 选 config | Unified Sketch + 4 agent | LLM-as-world-model 搜索 |
| **是否发现 inductor 规则外的融合** | ❌ | ✅(理论上) | ✅(cost 反馈) | ✅ | ✅(代数等价) | ✅(RL) | ⚠️(选 config 不 fusion) | ✅(LLM 推理) | ✅(LLM 推理) |
| **是否需要运行时反馈** | ❌ | ❌ | ✅ | ✅(polyhedral) | ✅(benchmark) | ✅(reward) | ✅(TPU 测量) | ✅(compile/run feedback) | ✅(perf feedback) |
| **是否自动 codegen** | ✅ Triton/C++ | ❌ | ✅ | 部分 | ✅ | ❌ | ❌ | ✅ 多 DSL | ✅ |
| **跨层级融合** | 单层 | 多层(block 嵌套) | 单层 | 多层 | **三层** | N/A | N/A | 跨算子 | 跨算子 |
| **编译时间** | 秒级 | 秒级 | 分钟 | 秒~分钟 | 分钟~小时 | 训练小时 | 秒 | **分钟~小时(多轮 Evolve)** | **小时(120 iterations)** |
| **对 NPU 友好度** | 取决于实现 | ✅(MLIR Linalg 通用) | ⚠️ | ✅(Ascend 实验) | ❌(CUDA 为主) | ❌ | ❌ | ✅(**原生 Ascend**) | ❌(CUDA/Triton) |
| **成熟度(产线可用)** | ✅ PyTorch 默认 | ✅ IREE / LLVM 主线 | ✅ Apache TVM | ✅ MLSys'22 | ⚠ 论文 | ⚠ 论文 | ⚠ 论文 | ⚠ 论文 | ⚠ 论文 |
| **能引用 inductor 的 173 个可融合算子** | ✅ | ✅(通过 Torch → Linalg) | ✅ | ✅ | ✅(用 cuBLAS 等) | N/A | N/A | ✅(走 Triton / AscendC) | ✅(走 Triton) |
| **关键创新点** | PyTorch 默认 | 决策/变换解耦 | 模板无关 + cost model | 双向反馈 | 三层 superopt | RL 多目标图切分 | GNN 跨配置 attention | 4 agent + Unified Sketch | LLM world model |

**关键观察**:

- **inductor 的最大短板是"静态、无反馈"**——所有静态规则派(MLIR/TVM/XLA/Apollo)在这一点上各有改善,但**只有 ML 派和 Agent 派把反馈做到了极致**。
- **跨层 fusion 是公认短板**:Princeton 引用证实 layer-bounded fusion 平均损失 32%,这正是 agent 能补回的空间。
- **Agent 路线在 NPU 上还没有 SoTA**——AKG Kernel Agent 是目前唯一原生 Ascend 实验的工作,但仍是研究阶段。
- **编译时间是 Agent 路线的硬约束**:120 iterations × 多轮 Evolve 对在线 inference 不可接受,需要分级(热路径 inductor,冷路径 agent)。

---

## 7. 一个 Agent 驱动的可融合范围识别框架(提议)

> 这一节是**结合调研结论的具体方案提议**,不是已实现的系统。目的是给你后续工程化的可参考骨架。

### 7.1 总体架构

```
┌────────────────────────────────────────────────────────────────────┐
│                      PyTorch FX Graph (aten + prim)                 │
└──────────────────────────────┬─────────────────────────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │   Layer 1: 静态规则快速路径    │
              │   (inducer can_fuse 复用)      │
              │   - 耗时: < 1s                  │
              │   - 覆盖: 173 个 GPU 可融合算子  │
              └────────────────┬────────────────┘
                               │ 标记"低置信度 fusion"
                               │
              ┌────────────────▼────────────────┐
              │  Layer 2: Agent 深度探索        │
              │  (Designer → Verifier → Evolve) │
              │  - 耗时: 分钟~小时              │
              │  - 覆盖: 跨层 + 模式 + 新算子   │
              └────────────────┬────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │  Layer 3: Codegen + 验证        │
              │  (Triton / AscendC / 自定义)    │
              └─────────────────────────────────┘
```

### 7.2 关键组件

#### (1) Designer Agent(产出 Unified Sketch)

**输入**:FX Graph + 硬件 spec(NPU 型号、AscendC 限制)。
**输出**:Unified Sketch(参考 AKG Kernel Agent,做硬件无关的中间表示)。
**关键**:把 inductor 的 `can_fuse` 表**作为 baseline 提示词**,让 LLM 在 baseline 上"看哪里能再优化"。

#### (2) Verifier Agent(编译/性能反馈)

**输入**:Sketch + 编译结果 + 运行 benchmark。
**输出**:错位类型(语法/算法/性能/数值)+ 改进建议。
**关键**:**复用 inductor 的 `measure_fusion_depth.py`**(你已有的脚本),直接对接 NPU profiler 量化"fusion 深度"收益。

#### (3) Conductor(Apollo-style 反馈)

**关键决策**:fusion plan 不可行时,是否触发 Designer 重新设计?(而不是放弃)。
**实现**:用 Apollo 的双向耦合思路——把"下游 codegen 失败"作为 prompt 反馈,触发上游 partitioner 重切。

#### (4) 知识库(避免幻觉)

**内容**:
- inductor 173 个可融合算子的清单(从 `docs/ai_gen/scripts/measure_decomp_coverage.py` 抽)
- AKG Unified Sketch 样例库
- NPU AscendC 性能特征(从已有 profiler 抽取)
- 历史成功的 fusion plan + 编译产物

### 7.3 三阶段 rollout(避免一次性铺开)

| 阶段 | 时间 | 范围 | 目标 |
|---|---|---|---|
| **Phase 1: 离线建库** | 2-4 周 | 跑全 inductor 可融合集 + NPU 模型 | 收集 ~1000 个 Unified Sketch + 编译产物 |
| **Phase 2: 闭环验证** | 4-8 周 | 5-10 个目标模型,vs inductor 基线 | 验证 agent 能在 NPU 上比 inductor 提升 X% |
| **Phase 3: 在线灰度** | 8-12 周 | JIT 路径上 hot-model 灰度 | 缓存 + 离线预算 + inductor fallback |

### 7.4 关键指标

- **融合覆盖率**:相比 inductor 的 173 个可融合算子,agent **额外**识别多少?
- **跨层融合比**:哪些被识别的范围是 inductor 静态规则**不会**尝试的(跨 layer、跨 view)?
- **融合深度**:每个 kernel 平均算子数(inducer `measure_fusion_depth.py` 已能测)。
- **端到端 speedup**:真实 NPU 跑模型的 P50/P90 vs inductor。

---

## 8. 风险与未解问题

### 8.1 编译时间爆炸

agent 多轮 Evolve 一次可能分钟~小时,**对 inference JIT 场景不可接受**。缓解:分级缓存——热路径 inductor(秒级),冷路径 agent(分钟级,仅编译一次)。

### 8.2 LLM 幻觉与 API 错用

LLM 编 kernel 会编不存在的 CUDA/Triton/AscendC API。缓解:**DocSpec 知识库 + 编译期强校验**(参考 AKG Kernel Agent 的做法)。

### 8.3 不可融合的"假阳性"

LLM 可能建议一个 fusion,但实际 codegen 出超慢 kernel(比如 reduction+pointwise 跨 tile 边界)。缓解:用 Apollo 的双向反馈——codegen 阶段把"实际 NPU 测量 runtime"反向告诉 Designer,触发重新设计。

### 8.4 评估协议

- KernelBench 有 reward hacking 漏洞(参考 AKG 论文 § 3.6)。
- fast₁ 数字来自不同口径,**不能直接横比**。
- 评估必须用**真实 NPU + 真实模型 + 真实 shape**,不能只看 KernelBench。

### 8.5 与 inductor 维护性的冲突

induct 在 PyTorch 主线,**每个版本都改**。如果 agent 训练的 prompt/cost model 强依赖某版本 inductor 行为,会很快 stale。缓解:把 agent 设计成**接收 inductor 输出作为输入**,而非内嵌 inductor。

---

## 9. 结论

### 9.1 直接回答你的核心问题

> "有不有比 inductor 融合规制更优的机制和方案,来识别可融合范围?"

**有,且至少有 3 个层次的更优机制**:

1. **静态规则增强**:MLIR `LinalgExtFusionInterface` + Apollo 的双向反馈 + 4 类算子分类(XLA/TVM)——**覆盖 inductor 同等功能,工程化最易**(2-3 月可落地)。
2. **ML 驱动**:TGraph(GNN 选 config) + AStitch(多维 fusion 空间) + Mirage(代数等价 superopt)——**能突破 inductor 天花板,但需训练数据 + 工程改造成本**(6-12 月)。
3. **Agent 驱动**:AKG Kernel Agent(4 agent 协作) + K-Search(LLM-as-world-model)——**既能发现新融合,又能生成代码,但编译时间不可接受**(需分级 fallback,12+ 月)。

### 9.2 推荐路线

**短期(2-3 月)**:用 MLIR LinalgExtFusionInterface 改造 inductor 的 `can_fuse`,引入 Apollo 风格的下游反馈,先做到**与 inductor 同等覆盖 + 部分修复 NPU 短板**。

**中期(6-12 月)**:把 AKG Kernel Agent 的 4 agent 架构移植到 inductor,作为**离线冷路径**——对热模型 offline 生成融合 plan,缓存到 inductor。

**长期(12+ 月)**:考虑 K-Search 风格的世界模型搜索,做"开放式 fusion 探索",覆盖 inductor 规则**之外**的融合机会。

### 9.3 一句话总结

inducer 的 `can_fuse` 是 2010 年代的产物(类别硬编码 + 无反馈);业界已演进到 ML cost model 驱动的 cost-guided fusion(Ansor) + 双向反馈 fusion(Apollo) + 多层 superoptimization(Mirage) + Agent 自动 fusion(AKG Kernel Agent)四个新范式。**用 AKG 的 4 agent + Unified Sketch 作为骨架,配合 Apollo 双向反馈,加上 NPU 上多 round profiling,就能在 inductor 规则之外自动发现并生成可融合范围**。

---

## 附录 A. 信源与可信度

| 类别 | 信源 | 可信度 | 用途 |
|---|---|---|---|
| **官仓源码** | `pytorch/pytorch` `torch/_inductor/` | 🟢 高 | inductor IR/scheduler 实测 |
| **官仓源码** | `mindspore-ai/akg` | 🟢 高 | AKG 多面体编译 |
| **官仓源码** | `iree-org/iree` LinalgExtInterfaces.td | 🟢 高 | LinalgExtFusionInterface 定义 |
| **arXiv 论文** | Mirage(Wu et al., CMU 2024) | 🟢 高 | superoptimization 范式 |
| **arXiv 论文** | Apollo(MLSys 2022) | 🟢 高 | 双向反馈 fusion |
| **arXiv 论文** | NeuroCUT(KDD 2024) | 🟢 高 | GNN+RL 图切分 |
| **arXiv 论文** | AKG Kernel Agent(arXiv 2512.23424) | 🟢 高 | 多 agent 编 kernel |
| **arXiv 论文** | K-Search | 🟢 高 | LLM-as-world-model |
| **arXiv 论文** | TGraph(arXiv 2405.16623) | 🟢 高 | GNN config 选择 |
| **arXiv 论文** | LLM4Kernel 综述(arXiv 2601.15727) | 🟢 高 | LLM 编 kernel 综述 |
| **会议论文** | OSDI(Ansor, TpuGraphs) | 🟢 高 | TVM, ML 编译 |
| **会议论文** | MLSys(Apollo) | 🟢 高 | 编译系统 |
| **会议论文** | ASPLOS(AStitch, Mirage) | 🟢 高 | 体系结构/编译 |
| **官方文档** | OpenXLA / Apache TVM / MLIR | 🟢 高 | 架构说明 |
| **博客/媒体** | emergentmind / towardsdatascience | 🟡 中 | 案例佐证(不引用为唯一证据) |
| **Reddit / Substack** | r/Compilers | 🟡 中 | 社区观点 |
| **LinkedIn 公告** | NeuroCUT 作者公告 | 🟡 中 | KDD 录用佐证 |

调研遵循 *deep-research* skill 的多源交叉验证原则:每个事实陈述至少 1 个高可信度源 + 1 个中可信度源支撑;无可信源直接关联的细节明确标注"待验证"。

---

## 附录 B. 关键文献索引

按"对你想做的事情的相关度"排序:

### B.1 必须精读(★★★)

1. **AKG Kernel Agent**(Du et al., arXiv 2512.23424)— 与 NPU + Triton 直接相关,4 agent + Unified Sketch 设计可抄。
2. **Mirage: A Multi-Level Superoptimizer for Tensor Programs**(Wu et al., arXiv 2405.05751, ASPLOS 2024)— 证明"自动发现融合"可行,μGraph + algebraic pruning 思路可借鉴。
3. **Apollo: Automatic Partition-based Operator Fusion through Layer...(MLSys 2022)**— 双向反馈 fusion,可作为反馈环路设计参考。
4. **MLIR LinalgExtFusionInterface + TilingInterface**(IREE 源码 + LLVM Discourse)— fusion 与 transformation 解耦的设计哲学。

### B.2 应该读(★★)

5. **K-Search**(arXiv 2026)— LLM-as-world-model 搜索的 SOTA 证据。
6. **TGraph**(Khizbullin et al., arXiv 2405.16623)— GNN 选 config,了解 ML 驱动 fusion 的可行形式。
8. **NeuroCUT**(Manchanda et al., KDD 2024)— GNN+RL 图切分,通用图切分 SOTA。
9. **Operator Fusion in XLA**(Snider & Liang 2023, arXiv 2301.13062)— XLA fusion 实测分析,补足 inductor vs XLA 的对照。
10. **LLM4Kernel 综述**(Yu et al., arXiv 2601.15727)— 50+ LLM-kernel 工作全景,避免重复造轮子。

### B.3 可选参考(★)

11. **AStitch**(ASPLOS 22)— 多维 fusion 空间的 ML 优化。
12. **Ansor**(OSDI 2020, Zheng et al.)— template-free auto-scheduling。
13. **MetaSchedule**(NeurIPS 2022, Shao et al.)— 概率程序描述 schedule。
14. **Tiramisu**(CGO 2019, Baghdadi et al.)— 多面体 fusion 的数学派代表。
15. **TVM OSDI 2018 paper**(Chen et al.)— TVM 奠基论文,了解 4 类算子分类的原始定义。
16. **Princeton ML for ML Compilers 课件**— 32% 性能损失数据点的原始来源。

---

*报告生成日期:2026-08-20*
*调研方法:anysearch v3.0.1 多源检索 + 关键论文全文抽取(arXiv) + LLM4Kernel 综述交叉验证*
*参考基线:`docs/ai_gen/inductor_fusion_breadth_analysis.md`(2026-07-20 版)*
*下次复审建议:本领域 2026 Q4 节奏很快,KernelBench 系每月有新 fast₁ 数字,建议 3-6 月跑一次 anysearch 跟踪新工作。*