# 基于 LLM Agent 的图优化引擎调研报告

> **调研主题**:复用 `torch.compile` / TorchInductor 的图解析与 codegen 基础设施,以 codegen 出的 Triton 代码作为性能基线,在此之上叠加 LLM Agent 迭代优化(以及后续的图切分与可融合方案识别)。**学术界与业界是否有同类工作**?它们处于什么阶段?
>
> **调研方法**:anysearch v3.0.1 多源检索 + 关键论文/官方仓库全文抽取 + 交叉验证。信源标注见附录 A。
>
> **v1 报告生成日期**:2026-08-24

---

## 目录

- [0. TL;DR](#0-tldr)
- [1. 调研范围澄清:你做的事情到底是什么](#1-调研范围澄清你做的事情到底是什么)
- [2. 你的项目与工业现状的逐项对照](#2-你的项目与工业现状的逐项对照)
- [3. 第一类:整算子生成派(Kernel-as-Unit,GPU 主导)](#3-第一类整算子生成派kernel-as-unitgpu-主导)
- [4. 第二类:Agent + RL/Search 派(KernelBench 时代的范式)](#4-第二类agent--rlsearch-派kernelbench-时代的范式)
- [5. 第三类:编译器协同派(Compiler-LLM Cooperation)](#5-第三类编译器协同派compiler-llm-cooperation)
- [6. 第四类:NPU 端原生派(Ascend 专属路线)](#6-第四类npu-端原生派ascend-专属路线)
- [7. 第五类:图切分 + 调度 + LLM(中后期阶段对标)](#7-第五类图切分--调度--llm中后期阶段对标)
- [8. 通用训练框架 / Survey](#8-通用训练框架--survey)
- [9. 主流方案横向对比矩阵](#9-主流方案横向对比矩阵)
- [10. 你的方案与既有方案的关键差异](#10-你的方案与既有方案的关键差异)
- [11. 风险、未解问题、对你的建议](#11-风险未解问题对你的建议)
- [附录 A. 信源与可信度](#附录-a-信源与可信度)
- [附录 B. 必读文献清单](#附录-b-必读文献清单)

---

## 0. TL;DR

围绕"**基于 `torch.compile` 基础设施 + LLM Agent 迭代优化算子内核**"这条路径,**学术界和工业界已经有非常成熟的工作**,且至少有 3 个机构的工作**与你构想的形态几乎一致**:

| 你的方案组件 | 工业/学术对标 | 成熟度 | 关键差异 |
|---|---|---|---|
| **复用 torch dynamo / FX 图解析** | `torch.compile` 全套 + `fx.graph`;**PyTorch KernelFalcon/KernelAgent** 直接跑在 inductor 上 | ✅ 工业级 | 完全相同思路 |
| **复用 inductor `can_fuse` + scheduler** | 直接当 fallback,作为 baseline | ✅ | 已成为所有工作的 "worst-case baseline" |
| **codegen Triton 代码作 baseline** | **KernelBook**(Meta,2025)直接用 `torch.compile()` 生成 (PyTorch, Triton) pairs | ✅ 数据集已发布 | 你不需要从零生成 baseline,KernelBook 已有现成数据集 |
| **LLM Agent 基于 baseline 迭代优化** | **Meta KernelFalcon/KernelAgent**(PyTorch 官方)、**Sakana AI CUDA Engineer**、**AMD GEAK**、**MIT/AutoKernel**、**Intel Xe-Forge**、**Berkeley K-Search** | ✅ 多个开源/闭源 | 完全相同范式 |
| **NPU 端适配(Ascend)** | **华为 LLM4Compiler 团队(Long Cheng)** —— **"LLM translation from Triton on GPU to Triton on Ascend NPU" + "mathematically-equivalent transformation to improve the Triton Kernel Performance" + "RL post-training on Ascend NPU"** | ⚠️ 闭源/华为内部 | **这是你方案最直接的对标团队** |
| **图切分 / 可融合范围识别(LLM 驱动)** | Apollo、TASO、DNNFusion、Welder + 通用 LLM 图推理(AdaSTORM) | ⚠️ 学术界主流;工业只有闭源 | 你 v2 报告已经覆盖这部分 |

**核心结论**:

1. **你的方案不是"从零造轮子"** —— 它在结构上与 PyTorch KernelFalcon / KernelAgent(2026.03)、Meta KernelLLM(2025)、Meta KernelBook(2025)三家**完全同构**,只是把 target hardware 从 CUDA 换成 Ascend NPU,并把算子 codegen 后端从 Triton 换成 Triton-on-Ascend。
2. **Ascend 方向最直接的对标**:**华为内部 LLM4Compiler 团队(Long Cheng, 现转 NVIDIA)** —— 该团队在 2021-2025 年做了**完整三件套**:GPU-Triton → Ascend-Triton 翻译、Trition 数学等价变换优化、Ascend NPU 上的 RL 后训练。**你方案里的"翻译 + 优化 + RL 后训练"在这家团队都已经有雏形**,但**没有开源**。
3. **图切分 / 可融合范围识别的 LLM 驱动**目前**学术界无人做**(v2 报告覆盖的 TASO/DNNFusion/Welder 都还是规则派)。这一块是你真正的差异化方向。

---

## 1. 调研范围澄清:你做的事情到底是什么

### 1.1 你构想的工程形态(从你的 prompt 中提取)

```
┌────────────────────────────────────────────────────────────────┐
│ Stage 1: Graph Parsing                                          │
│   - 复用 torch dynamo / fx.graph → 识别可融合范围                │
│   - 复用 inductor `can_fuse` + scheduler(作为 baseline 候选)     │
├────────────────────────────────────────────────────────────────┤
│ Stage 2: Baseline Codegen                                       │
│   - inductor codegen → Triton 代码(output_code.py)              │
│   - 这一步产出的 Triton 代码作为"性能保底"(至少 ≥ 纯 inductor)   │
├────────────────────────────────────────────────────────────────┤
│ Stage 3: LLM Agent 迭代优化                                     │
│   - Agent 读取 baseline Triton 代码 + FX 子图 + NPU profiling    │
│   - 多轮 (Generate → Verify → Reflect → Optimize) 循环          │
│   - 目标是"找到比 baseline 更优的 Triton 代码"                   │
├────────────────────────────────────────────────────────────────┤
│ Stage 4: 部署与执行                                              │
│   - 用最终 Triton 代码替换 inductor codegen 产物                 │
│   - 模型走 torch.compile 或 inductor AOT 路径执行                │
├────────────────────────────────────────────────────────────────┤
│ Stage 5(后续): 图切分 + 可融合方案识别                            │
│   - 切分算子图 → 子图委派给 Agent → 子图重组                     │
└────────────────────────────────────────────────────────────────┘
```

### 1.2 与既有的同类工作的精确对应

你的方案 **Stage 1+2+3** 对应业界已有三套工作(详见 § 3-5):

| 你的 Stage | PyTorch KernelFalcon / KernelAgent | Sakana AI CUDA Engineer | 华为 LLM4Compiler |
|---|---|---|---|
| Stage 1 图解析 | FX graph + 子图分解(extracting fusion boundaries) | PyTorch → IR 翻译 | Triton(IR 视角) |
| Stage 2 Baseline codegen | 不显式做 baseline,直接 LLM 生成 | Stage 1:LLM 翻译 PyTorch → CUDA | LLM 翻译 Triton-on-GPU → Triton-on-Ascend |
| Stage 3 Agent 优化 | 多 agent 闭环(Profiler/Judge/Optimize) | 进化式 + RAG(innovation archive) | 数学等价变换 + RL 后训练 |

**关键观察**:**你提的"codegen baseline → Agent 在此基础上优化"在工业上反而是少数派**——大多数工作(GEAK / CUDA Engineer / K-Search)是从 PyTorch **或** Triton 代码**直接由 Agent 端到端生成**,**没有显式 baseline**。这意味着:

- **你的 baseline-as-anchor 策略**与 KernelLLM(用 `torch.compile()` 生成训练数据)、KernelBook(用 inductor 产生 (PyTorch, Triton) pairs) **有共同思想**,但**实操时如何把 baseline 注入 Agent 的 prompt 还没有公开方案**。
- **这是一个工程空白点**,可以作为你的差异化创新。

---

## 2. 你的项目与工业现状的逐项对照

### 2.1 你的基线 vs 已有工作

| 组件 | 你的方案 | 工业对标 | 复用度 |
|---|---|---|---|
| FX 图解析 + 算子切分 | torch dynamo | PyTorch KernelFalcon(extracting fusion boundaries)| **完全相同** |
| 可融合范围识别 | inductor `can_fuse` + scheduler | XLA/TASO/DNNFusion/v2 报告覆盖的所有派 | **完全相同** |
| Triton codegen baseline | inductor `output_code.py` | **KernelBook** 用 `torch.compile()` 产生 25K (PyTorch, Triton) pairs | **完全相同的产出方式** |
| Agent 反馈循环 | Generate/Verify/Reflect/Optimize | GEAK/KE/Xe-Forge/AutoKernel 全是这个范式 | **完全相同** |
| NPU 端验证 | torch_npu / AscendC 适配 | AscendCraft(2026.01)/AscendKernelGen(2026.01) | **专属工作,需自研** |
| 图切分 + Agent 驱动 | 中后期阶段 | 暂无成熟工作 | **差异化创新点** |

### 2.2 关键参考实现对标(按"形态相似度"排序)

| # | 工作 | 机构 | 形态相似度 | 你能直接复用的部分 |
|---|---|---|---|---|
| 1 | **PyTorch KernelFalcon** + **PyTorch KernelAgent** | Meta(PyTorch 团队) | ⭐⭐⭐⭐⭐ | `meta-pytorch/KernelAgent` GitHub 完整开源;FX graph + 子图分解 + Triton 生成 + 验证 + 多 worker 并行的全栈 pipeline |
| 2 | **华为 LLM4Compiler / VecTrans**(Long Cheng 团队) | 华为 2012 Lab(已转 NVIDIA) | ⭐⭐⭐⭐⭐ | Triton → Ascend Triton 翻译 + 数学等价变换 + Ascend RL 后训练三件套,**全部对标你方案的每个组件** |
| 3 | **AMD GEAK** | AMD | ⭐⭐⭐⭐ | Triton kernel agent 通用框架,8 个 agent 模块化,ROCm 现成 |
| 4 | **Intel Xe-Forge** | Intel Labs | ⭐⭐⭐⭐ | Triton kernel 优化,9 stage 流水线 + CoVeR 反思闭环 |
| 5 | **Sakana AI CUDA Engineer** | Sakana AI | ⭐⭐⭐ | PyTorch → CUDA + 进化式 + RAG;但无 NPU |
| 6 | **AutoKernel** | MIT (RightNow-AI) | ⭐⭐⭐⭐ | 端到端 agent + 5 阶段正确性 harness,9K 行 Python 全开源 |
| 7 | **K-Search** | UC Berkeley Sky Lab | ⭐⭐⭐ | "co-evolving world model" 概念,GPU Triton |

---

## 3. 第一类:整算子生成派(Kernel-as-Unit,GPU 主导)

> 这一类以"输入 PyTorch 模块 → 输出单个 Triton/CUDA kernel"为基本动作。**Agent 不一定出场**,但代表 LLM kernel 生成的事实 baseline。

### 3.1 KernelBench / KernelBench-X(Stanford + AlphaXiv)

**来源**:[scalingintelligence.stanford.edu/pubs/kernelbench.pdf](https://scalingintelligence.stanford.edu/pubs/kernelbench.pdf) + [arxiv.org/abs/2605.04956](https://arxiv.org/abs/2605.04956v1)

**核心数据**:
- 三层任务:Level 1 单算子(100)、Level 2 融合模式(100)、Level 3 整模型(50)
- 关键指标 `fast_p`:正确且 speedup > p%(eager 基线)

**关键发现(KernelBench-X)**:
- **Category 解释正确率方差是方法本身的 3 倍**(9.4% vs 3.3%)——**意味着 LLM 算子生成的能力上限由"算子类型"决定,而非方法**。Fusion 任务 72% 全方法失败。
- **迭代 refinement 提高正确率但降低 speedup**:GEAK 迭代从 0→2,正确率 18.2%→30.7%,但 **平均 speedup 从 1.58×→1.44×** —— "新救回来的 kernel 性能总是比一开始就正确的差"
- **46.6% 正确的 kernel 反而比 PyTorch eager 慢**;cross-hardware 速度方差最高 21.4×
- **Quantization 任务 0/30 全部失败** —— 暴露了对数值精度的理解不足

**对你的价值**:
- KernelBench 是事实标准 benchmark,你需要用它做"baseline + Agent 优化"的对比
- **KernelBench-X 的结论 "iterative refinement 不能保证性能提升" 直接关系到你的 Stage 3 设计** —— 你必须有"显式 baseline + 性能下降时回退"的机制,否则 Agent 可能输出"正确但更慢"的代码

### 3.2 KernelLLM + KernelBook(Meta, 2025)

**来源**:[huggingface.co/facebook/KernelLLM](https://huggingface.co/facebook/KernelLLM) + [huggingface.co/datasets/GPUMODE/KernelBook](https://huggingface.co/datasets/GPUMODE/KernelBook) + [github.com/meta-pytorch/popcorn-kernels](https://github.com/meta-pytorch/popcorn-kernels)

**核心**:
- **KernelLLM**:Llama 3.1-8B-Instruct 在 25K (PyTorch, Triton) pairs 上 SFT,在 KernelBench-Triton Level 1 上**单发性能匹配 GPT-4o 和 DeepSeek V3**
- **KernelBook / popcorn-kernels**:核心数据来源是 **GitHub 真实仓库 + `torch.compile()` 生成的 Triton** —— **这正是你 Stage 2 codegen baseline 的数据集形态**

**数据流程**(完全可复用):

```
GitHub PyTorch repos
  ↓ extract nn.Module
  ↓ unit test 验证
  ↓ torch.compile() codegen
  ↓ 提取 Triton kernel,转换格式
  ↓
(PyTorch source, Triton kernel) pairs (KernelBook)
```

**对你的价值**:
- **KernelBook 已经把 "PyTorch → torch.compile() → Triton" 这条管线变成数据集** —— 你不需要重新实现 baseline 生成,直接用 KernelBook 当训练数据 + 验证数据
- **KernelLLM 这种 "用 codegen 作为 ground truth 做 SFT" 的范式可以借鉴** —— 你的 Agent 可以预先用 (FX graph + inductor Triton + profile) 三元组 SFT 一次,再上线做 agentic refinement

### 3.3 Sakana AI CUDA Engineer(2025.02)

**来源**:[pub.sakana.ai/static/paper.pdf](https://pub.sakana.ai/static/paper.pdf) + [arxiv.org/abs/2509.14279](https://arxiv.org/abs/2509.14279)

**核心数据**:
- 4 阶段 pipeline:**翻译 → 进化优化 → verifier → innovation archive**
- 250 tasks 中 186 个成功优化,**median speedup 1.52×**,**75% 操作优于 torch**,60% 操作优于 `torch.compile`
- 部分任务(3D 卷积、对角矩阵乘法)>50×
- 释放 **17,000+ verified CUDA kernels**

**关键机制**:
- **LLM ensemble** + **temperature sampling** + **crossover prompting**(AlphaCode 风格)+ **innovation archive**(检索增强,把已成功 kernels 作为 stepping stone)
- **soft verifier**:LLM 预筛正确性,大幅降低硬件验证开销

**对你的价值**:
- **innovation archive / RAG** 这个机制可以直接搬到你的 Stage 3 —— 把"FX 子图 → Triton baseline → profile → 优化版"三元组入库存档,作为新子图的 in-context 例子
- 一旦 baseline 性能 ≥ torch.compile,**innovation archive 应该是这套工作的核心竞争壁垒**

### 3.4 AlphaEvolve(Google DeepMind, 2025.05)

**来源**:[en.wikipedia.org/wiki/AlphaEvolve](https://en.wikipedia.org/wiki/AlphaEvolve) + [cloud.google.com/blog/products/ai-machine-learning/alphaevolve-is-available-for-everyone](https://cloud.google.com/blog/products/ai-machine-learning/alphaevolve-is-available-for-everyone)

**核心**:通用 evolutionary coding agent(Gemini 驱动),在 50 个数学问题上 75% 重现 SOTA,20% 发现更优解。已经优化了 Google 自己的 TPU 设计、Spanner LSM-tree(降低 20% write amplification)。

**对你的价值**:
- **通用 evolutionary loop**(LLM 生成变体 + 评估函数筛)可以直接套到 Stage 3
- OpenEvolve 开源实现([huggingface.co/blog/codelion/openevolve](https://huggingface.co/blog/codelion/openevolve))可作为参考
- **不需要从零写 LLM ensemble / 进化策略** —— AlphaEvolve 已验证这套框架

---

## 4. 第二类:Agent + RL/Search 派(KernelBench 时代的范式)

> 这一类是**真正把"agent"作为核心抽象**的工作 —— 不是单次代码补全,而是 Gen-Verify-Reflect 闭环。

### 4.1 GEAK(AMD, 2025.07) ⭐⭐⭐⭐

**来源**:[arxiv.org/pdf/2507.23194v1](https://arxiv.org/pdf/2507.23194v1) + [github.com/AMD-AGI/GEAK-agent](https://github.com/AMD-AGI/GEAK-agent) + [rocm.blogs.amd.com](https://rocm.blogs.amd.com/artificial-intelligence/geak-agents-family/README.html)

**核心数据**:
- 4 模块:**Generator → Reflector → Evaluator → Optimizer**
- 闭源 v1 数据: TritonBench-revised **54.89% execution accuracy**, ROCm **63.33%**, 最高 **2.59× speedup**
- GEAK-OptimAgentv2 / GEAK-OpenEvolve(Quality-Diversity search)
- 完整闭环:profile → optimization → validation

**关键技术细节**:
- **Profiling 把原始计数器翻译成"结构化自然语言性能情报"**(如"the kernel is memory-bound due to poor L2 cache locality")
- LLM Analyzer 把 raw NCU 输出 → 专家级洞察
- LLM as Optimizer:用历史性能排序数据引导策略
- **Debugging Trap**:失败 N 次后自动放弃,防止 agent 卡死
- Parallel Scaling:多实例并行探索

**对你的价值**:
- **GEAK 的 Generator-Reflector-Evaluator-Optimizer 四模块**可以直接对应到你的 Stage 3 agent 设计
- **"结构化自然语言 profiling"** 这个概念对你在 Ascend NPU 上尤其重要 —— Ascend 的 profiling 数据没有 NCU 那么标准化,你需要给 LLM 一个友好抽象
- **Full-repo optimization(Git workspaces)** 与你的 NGO 项目形态对齐

### 4.2 PyTorch KernelFalcon + KernelAgent(Meta, 2026.03) ⭐⭐⭐⭐⭐ **(最相关)**

**来源**:[pytorch.org/blog/kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents](https://pytorch.org/blog/kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents) + [pytorch.org/blog/kernelagent-hardware-guided-gpu-kernel-optimization-via-multi-agent-orchestration](https://pytorch.org/blog/kernelagent-hardware-guided-gpu-kernel-optimization-via-multi-agent-orchestration) + [github.com/meta-pytorch/KernelAgent](https://github.com/meta-pytorch/KernelAgent)

**核心数据**:
- **KernelFalcon:首个开源的 agentic system 在 250 个 KernelBench L1/L2/L3 任务上达到 100% 正确率**
- **KernelAgent:加入硬件引导的优化层**(NCU 28 个硬件指标)
- 多 agent 架构:**ProfilerAgent → JudgeAgent → AnalyzeAgent → Orchestrator Agent → Optimization Manager → BenchmarkAgent**

**与你的项目高度对齐的细节**(摘自 PyTorch 博客):
- "**extracts fusion boundaries**" —— **与你的 Stage 1 完全相同**
- "**generates Triton kernels**" —— **与你的 Stage 2/3 完全相同**
- "**composing end-to-end modules**" —— **与你的 Stage 4 部署完全相同**
- "**hardware-guided optimization pipeline**" —— **与你的 Stage 3 加 NPU profile 完全相同**
- KernelFalcon 的"deterministic control plane with early-win parallel search" + "persistent memory/observability" + "grounded tool use" —— **与你的 NGO 工程架构几乎同构**

**对你的价值**:
- **这是你方案最直接的工业对标** —— GitHub 仓库完整开源,你应该 fork 一份研究它的 agent 编排
- **README 显式提到 "intentionally leaves Triton installation to the user"** —— 同样地,你可以 fork 并把 Triton 替换成 Triton-on-Ascend
- **agent 命名 + 责任划分可以照抄**:Profiler/Judge/Analyze/Orchestrator/Benchmark 五个 agent 是经过验证的最小集合

### 4.3 AutoKernel(MIT / RightNow-AI, 2026)

**来源**:[arxiv.org/html/2603.21331](https://arxiv.org/html/2603.21331)

**核心数据**:
- 9,000+ 行 Python + 909 行 agent 指令文档,18 个 starter kernel,**250 standardized problems**
- **9 kernel 类型**,5 阶段正确性 harness(smoke / shape sweep / numerical stability / determinism / edge case)
- **H100 实测**:
  - RMSNorm:**5.29× over eager**,**2.83× over `torch.compile(max-autotune)`**(83% 带宽利用率)
  - softmax:**2.82× / 3.44×**
  - cross-entropy:**2.21× / 2.94×**
  - **16 配置中 12 个**击败 torch.compile(max-autotune)
- **Community 战报**:单 prompt 3 分钟生成的 Triton FP4 matmul **1.63×-2.15× over CUTLASS**(峰值 2898 TF/s)
- 每次迭代 ~90 秒(30 正确性 + 30 bench + 30 agent reasoning)

**关键技术**:
- **Amdahl's law 排序**:优化优先级按 `pct_gpu_time × achievable_speedup` 排序
- **Move-on criteria**:5 次 revert / 90% peak / 2h timeout / 2× speedup 任一触发即停
- 6-tier optimization playbook(tier 顺序应用)

**对你的价值**:
- **9,000 行 Python 的工程量是真实可参考的规模** —— 你的项目大可参照此量级
- **"starter kernel + tier playbook"** 这套机制可以借鉴 —— 你可以把 inductor codegen 当 starter kernel,Agent 按 playbook 应用优化
- **Amdahl 排序**在 NGO(搜广推场景)上尤其重要 —— 推荐模型中 5% 的高频 op 可能贡献 70% 的时间

### 4.4 K-Search(UC Berkeley Sky Lab)

**来源**:[sky.cs.berkeley.edu/project/k-search](https://sky.cs.berkeley.edu/project/k-search) + [arxiv.org/html/2602.19128](https://arxiv.org/html/2602.19128v1) + [github.com/caoshiyi/K-Search](https://github.com/caoshiyi/K-Search)

**核心创新**:**co-evolving world model** —— 把 LLM 自身作为"世界模型"引导搜索,而不是用静态启发式
- 维护一个"structured search tree",编码对 kernel 瓶颈的假设、设计备选、优化策略
- 在 TriMul 任务上 H100 达到 1030 µs,**超越所有先前的进化方法和人工设计**

**对你的价值**:
- **"co-evolving world model"** 这个抽象在搜索空间爆炸时(例如 Ascend NPU 的多个 cube/vector 单元组合)尤其有用
- 你可以借鉴它**让 Agent 在多次迭代中学习"哪些策略对这个子图有用"**

### 4.5 其他 Agent 工作(简表)

| 工作 | 机构 | 关键数字 | 形态 |
|---|---|---|---|
| **Intel Xe-Forge** | Intel Labs | FlashAttention **2-13.3×**,9 stage 流水线 + CoVeR 反思闭环 | Agent + Triton + Intel XPU |
| **METR KernelAgent** | METR | o3-mini-high **1.81×**,best-of-all **2.01×** (300 attempts/problem) | Parallel tree search |
| **Kevin-32B** | Cognition AI | **65% correctness**,multi-turn RL | RL-based |
| **CUDA-LLM** | 上海交大 | 比 cuBLAS 优 | RL + LLM-as-judge |
| **TritonForge/KBenchEval** | RLsys-Foundation | TritonBench + 多平台训练数据 | SFT+RL |
| **AutoTriton** | Tsinghua / AI9Stars | 性能匹配 Claude-4-Sonnet + DeepSeek-R1 | SFT + RL |
| **Dr. Kernel** | HKUST + TikTok + CUHK(SZ) + NTU | **Level-2 31.6% ≥ 1.2×**(超 Claude-4.5 26.7% / GPT-5 28.6%);**"reward hacking" / "lazy optimization"** 首先被系统化定义 | Multi-turn RL + KernelGym harness |

---

## 5. 第三类:编译器协同派(Compiler-LLM Cooperation)

> 这一类**最有学术创新性** —— 不是 LLM 单独做优化,而是把 LLM **嵌入编译器流水线**作为"智能助手"。

### 5.1 VecTrans(华为 2012 Lab, EuroLLVM 2025) ⭐⭐⭐⭐

**来源**:[arxiv.org/html/2503.19449v2](https://arxiv.org/html/2503.19449v2) + [eurollvm](https://www.llvm.org/devmtg/2025-04/slides/lightning_talk/cheng_beyond_pattern.pdf)

**核心数据**:
- 在 BiSheng Compiler(Kunpeng CPU, ARM NEON/SVE)上,**vectorize 51 个 TSVC 不可向量化函数中的 24 个**
- **1.77× geometric mean speedup**(peak 5.21× on `s293`)
- 每函数优化成本 **$0.012**(LLM API)
- DeepSeek-V3/VecTrans success ratio **46.2%**(8.76 平均迭代)

**机制**:**compiler analysis → LLM refactor → compiler auto-vectorize → formal verification**
- 用 LLVM IR 提取 vectorization facts 注入 prompt
- **Alive2 formal verification** + unit test 双层验证
- "Verification-aware multi-feedback iterative refinement" — 这就是一个 AI agent

**对你的价值**:
- **VecTrans 的"compiler analysis → LLM → compiler pass → verifier"流水线**可以直接对应到你的"inductor analysis → LLM → Triton codegen → NPU execution → verifier"流水线
- **Alive2-style formal verification** 的思想可以借,但你只能做"等价性 numerical check"(对比 inductor baseline 与 Agent 输出),而不是 IR-level 形式化
- 这是你 v2 报告"fusion 范围识别"系列工作在"自动向量化"领域的对应物

### 5.2 Agentic Code Optimization via Compiler-LLM Cooperation(2026)

**来源**:[arxiv.org/html/2604.04238v1](https://arxiv.org/html/2604.04238v1)

**核心数据**:
- **Overall speedup 1.25×**(vs 单独 LLM 优化)
- **11%** 时间 backend → frontend,30% 时间 backend → source-level optimization 重启 —— **compiler-LLM 工作流会"全栈迁移"**

**机制**:
- LLM-based agents for each level of abstraction
- Individual compiler constituents as tools
- LLM-based test generation agent
- Guiding LLM orchestrating

**对你的价值**:
- **"guiding agent 编排多个 level-of-abstraction 的工具"** 是这套工作的核心 —— 你的 NGO 可以把"FX graph 改写 / Triton codegen / Ascend op 替换 / NPU profile"都暴露成 tool,让 guiding agent 编排

### 5.3 TritonPilot(Compiler-Assisted LLM for C → Triton)

**来源**:[dl.acm.org/doi/abs/10.1145/3774895.3812200](https://dl.acm.org/doi/abs/10.1145/3774895.3812200)

**核心数据**:
- PolyBench/C 上 **97% correctness**(vs 自主 LLM agent 80%,unguided LLM 93%)
- 8× problem size 下 **median 24.6× speedup**(vs 20.7× agent / 10.8× unguided)
- profiling feedback 改善 9/29 standard + 12/21 large size
- 95% of 151 TSVC kernels pass

**核心**:**compiler analysis extracts parallelization properties → encodes as structured facts in LLM prompt**

**对你的价值**:
- **"把 compiler analysis 提取的事实注入 LLM prompt"** 是这套工作的核心创新 —— 对你来说,**Ascend NPU 的 profiling 信息**(cube/vector 占用率、L2 hit rate、memory bandwidth)应该被提取成"结构化 facts"注入 LLM prompt,而不是直接扔原始数字
- GEAK 也用了类似机制,但 TritonPilot 把它放到了"polyhedral compiler analysis"这个 level

---

## 6. 第四类:NPU 端原生派(Ascend 专属路线)

> **这是与你项目最直接相关的部分** —— 在 Ascend NPU 上做 LLM 驱动的 kernel 生成。

### 6.1 华为 LLM4Compiler / VecTrans(Long Cheng 团队) ⭐⭐⭐⭐⭐ **(你方案最直接对标)**

**来源**:[chenglong92.github.io](https://chenglong92.github.io/) + [linkedin.com/in/long-cheng92](https://linkedin.com/in/long-cheng92)

**关键事实**:
- **Long Cheng 在华为 2012 Lab 领导了 LLM4Compiler 项目**,期间(2021-2025)完整覆盖了你方案的每个组件:
  - ✅ "(agent) LLM source2source transformation to enhance the success ratio of auto-vectorization with BiSheng Compiler on ARM CPU"(→ VecTrans)
  - ✅ "(agent) **LLM translation from Triton on GPU to Triton on Ascend NPU** and also mathematically-equivalent transformation to improve the Triton Kernel Performance"
  - ✅ "(RL) **RL post-training (on Ascend NPU) to improve the capability of open-source LLM in NPU kernel generation**"
- 2024 年获**华为金牌团队奖**(Critical Fundamental Softwares + High-performance LLM Operators on Ascend NPUs)
- 现转 NVIDIA Senior Architect,继续 LLM infrastructure + GPU kernels

**对你方案的意义**:
- **你的方案和这个团队做的几乎是同一件事** —— 差异只是他们的内部代码没有开源,但 VecTrans 已在 openEuler 开源镜像
- **你需要主动 connect 这个团队 / 这些工作**(虽然 Long Cheng 已转 NVIDIA,但他之前的同事在 2012 Lab / Huawei Toronto / Cambridge 还在)
- **"Mathematically-equivalent transformation"** 这个点(v2 报告里 PET 的核心思想)被 Long 团队应用到了 Triton kernel 优化上 —— 这是你 v2 报告与 LLM-driven 方法的天然交叉点

### 6.2 AscendCraft(NJU + Huawei, 2026.01)

**来源**:[arxiv.org/html/2601.22760v1](https://arxiv.org/html/2601.22760v1) + [alphaxiv.org/abs/2601.22760](https://www.alphaxiv.org/abs/2601.22760)

**核心数据**:
- **DSL-guided transcompilation**:LLM 先在 NPU-aware DSL 中写 kernel,然后分 4 passes 降低到 AscendC
- **MultiKernelBench (52 个 AscendC kernels, 7 类):98.1% 编译成功 / 90.4% 功能正确 / 46.2% 超过 PyTorch eager**
- mHC 架构新算子(mHC_post):**6.6× (单 pass) → 15.9× (人类专家后续优化)**

**机制**:
- DSL 设计原则:(1) concise syntax(2) appropriate abstraction(隐藏 DataCopyPad 等)(3) targeted extensions(显式建模 UB/CopyIn/Compute/CopyOut)
- 4 passes:**Host-side translation → Kernel initialization → Kernel computation → Alignment and padding refinement**
- 每个 pass 都带编译错误反馈

**对你的价值**:
- **这套 DSL 思路**对你有借鉴价值 —— 你可以让 Agent 先在 FX/Triton DSL(而不是 AscendC)层面优化,然后通过结构化 pass 落到 AscendC
- **"NPU 直接生成 AscendC < 5% 正确率"** 这个数据非常重要 —— 说明你**不要让 Agent 直接生成 AscendC**,而应该让 Agent 在 Triton/fx IR 层优化,最后通过 NPU-aware lower 落到 Ascend

### 6.3 AscendKernelGen(Pengcheng Lab + Huawei, 2026.01)

**来源**:[arxiv.org/pdf/2601.07160](https://arxiv.org/pdf/2601.07160) + [huggingface.co/AscendKernelGen/KernelGen-LM-8B](https://huggingface.co/AscendKernelGen/KernelGen-LM-8B)

**核心数据**:
- **Ascend-CoT dataset**:文档 + 真实 kernel 实现 → chain-of-thought 三合一
- **KernelGen-LM-8B**(基于 Qwen3-8B):SFT + RL with execution feedback
- Level-2 complex kernels:**compilation success 0% → 95.5% (Pass@10)**,functional correctness 0% → 64.3%

**机制**:
- 三类 CoT:documentation-based(API/最佳实践)+ code-centric(从真实 kernel 抽取)+ general reasoning chain
- RL 用 execution feedback 直接驱动

**对你的价值**:
- **这套 SFT+RL 训练范式**直接可搬到你的 Agent —— 你可以把 (FX subgraph, inductor Triton baseline, profile, optimized Triton) 四元组做 SFT,然后 RL 后训练
- **"0% → 95.5% compilation success"** 的跃迁来自 domain-specific CoT 数据 —— **NPU 上的 kernel 生成必须有专属数据**,不能用通用 code 数据

### 6.4 AKG(Automatic Kernel Generation, Huawei MindSpore, PLDI 2021)

**来源**:[01.me/files/AKG/akg-pldi21.pdf](https://01.me/files/AKG/akg-pldi21.pdf) + [dl.acm.org/doi/10.1145/3453483.3454106](https://dl.acm.org/doi/10.1145/3453483.3454106)

**核心数据**(2021):
- 单算子:**1.6× over TVM**(平均)
- 算子子图融合:**1.3× over TVM,5.6× over CCE opt**(manually optimized C/C++)
- 整模型(BERT/SSD):**20.2% over TVM**,与手工 CCE 持平或略优
- 在 Ascend 910 上实验

**机制**:**polyhedral compilation + ILP-based scheduler + img2col / fractal tiling 自动化**

**对你的价值**:
- **AKG 已经被集成到 MindSpore** —— 但**没有 LLM 驱动**,纯编译器规则。如果你方案里需要"无 LLM 的 fallback",AKG 是一个选项(但只对算子内部优化有效,跨算子 fusion 还得靠 inductor)
- **Polyhedral + ILP scheduler** 这个机制可以借给"自动 fusion cut selection"

### 6.5 MultiKernelBench(NJU + Huawei, 2025.07)

**来源**:[arxiv.org/pdf/2507.17773v1](https://arxiv.org/pdf/2507.17773v1) + [github.com/wzzll123/MultiKernelBench](https://github.com/wzzll123/MultiKernelBench)

**核心数据**:**285 tasks, 14 categories, 3 平台**(NVIDIA CUDA / Huawei AscendC / Google TPU Pallas)

**对你的价值**:
- **AscendC 后端 + anti-hack 检测** —— 这是评估你的方案在 Ascend 上"是否作弊 / 是否真优化"的事实标准
- 你应该把 MultiKernelBench 作为你的 Stage 3 评估集的一部分

---

## 7. 第五类:图切分 + 调度 + LLM(中后期阶段对标)

> 这是你方案的"后续阶段" —— 还没有成熟工作,但是有 **direct partitioner 通用框架**和 **LLM + dynamic graph reasoning** 的初步工作。

### 7.1 通用 LLM Graph Reasoning:AdaSTORM(2026.06)

**来源**:[arxiv.org/pdf/2606.16328v1](https://arxiv.org/pdf/2606.16328v1)

**核心**:把大规模 dynamic graph reasoning 拆成两阶段:**Adaptive Partitioning(把图分成匹配 LLM reasoning 容量的 subregions)+ Collaborative Reasoning**

**对你的价值**:
- **"图切分匹配 reasoning capacity"** 的思想直接可搬到你的"图切分 + Agent"阶段 —— 不要把整个 FX graph 塞给一个 Agent,先切分,匹配每个 Agent 的处理能力
- 但这是 LLM-for-graph-reasoning 的通用框架,**不是 DNN 编译器专用** —— 你需要适配

### 7.2 通用 Graph Partitioning:LPS-GNN

**来源**:[emergentmind.com/topics/lps-gnn](https://emergentmind.com/topics/lps-gnn)

**核心**:LPMetis + 子图增强 + GNN backbones

**对你的价值**:
- **LPMetis-style partitioning** 在你的"图切分"阶段可以作为初始 partitioner,后续用 LLM 优化 partition
- 通用 GNN partitioning ≠ DNN operator graph partitioning,但思想可借鉴

### 7.3 工业级 Sharding:GSPMD(Google, MLSys 2022)

**来源**:[chips-compilers-mlsys-22.github.io](https://chips-compilers-mlsys-22.github.io/assets/slides/GSPMD_%20generalized%20parallelism%20for%20large%20models%20as%20shared%20compiler%20infrastructure.pdf)

**核心**:**priority-based algorithm 迭代 refinement sharding**,在 TPU 上 2048 chips 达到 63% FLOPS utilization

**对你的价值**:
- **图切分 + 反馈迭代** 的范式在你方案的中后期阶段要参考 —— 不是简单的 partitioner,而是 priority-based + iterative
- 你的"图切分"如果要做,优先级算法 + 反馈机制是必备组件

### 7.4 ⚠️ "LLM 驱动的图切分 + 可融合范围识别" 学术空白

**关键观察**:
- v2 报告里覆盖的所有 fusion 范围识别工作(TASO/DNNFusion/Welder/Apollo 等)**都是规则派 + cost model** —— **学术界目前没人用 LLM 来做 fusion 范围识别**
- 这是一个**真正的差异化方向**:用 LLM 来"识别可融合范围",而不是只用来"生成算子"

**你的潜在机会**:
- v2 报告里推荐了 Phase 1(TASO 风格 substitution + SMT 验证)+ Phase 2(DNNFusion fusion seed 搜索)+ Phase 3(Welder tile-graph)
- **新加一个 Phase 0:LLM 驱动的 fusion 范围候选生成** —— 让 LLM 用 v2 报告里的数学性质知识生成"哪些子图可能是可融合的"候选,再交给 Phase 1/2/3 验证
- 这与你 Stage 3 的 Agent 是**同一个 Agent**,只是 prompt 不同

---

## 8. 通用训练框架 / Survey

### 8.1 CompilerGym(Meta, CGO 2022)

**来源**:[arxiv.org/pdf/2109.08267](https://arxiv.org/pdf/2109.08267) + [github.com/facebookresearch/compilergym](https://github.com/facebookresearch/compilergym)

**核心**:**OpenAI Gym 风格编译器 RL 环境**
- GCC phase ordering 搜索空间 **10^4461**
- 内置 LLVM phase ordering、GCC flag tuning、CUDA loop nest generation

**对你的价值**:
- 如果你想做"用 RL 训练 Agent 而非 prompt-based",CompilerGym 的环境抽象可以直接套到你的 inductor + Triton 子任务上
- 但目前主流选择是 **prompt-based + in-context learning**,RL-based 还没成为主流(Dr. Kernel 是个例外)

### 8.2 MLGym(Meta, 2025)

**来源**:[arxiv.org/abs/2502.14499](https://arxiv.org/abs/2502.14499) + [github.com/facebookresearch/mlgym](https://github.com/facebookresearch/mlgym)

**核心**:13 个 ML 研究任务的 Gym 环境,用于评估 LLM agent 在 AI R&D 上的能力

**对你的价值**:与你的方案不直接相关,但 agent loop 设计可以借鉴

### 8.3 Towards Automated Kernel Generation in the Era of LLMs(2026)

**来源**:[arxiv.org/html/2601.15727v3](https://arxiv.org/html/2601.15727v3)

**核心**:**目前最系统的 LLM kernel generation survey**,把方法分成:
- 训练驱动(SFT / RL / SFT+RL)
- Agent 驱动(learning mechanism / external memory / hardware profiling / multi-agent orchestration)
- "Agentic training and harness engineering" —— **未来方向**

**必读** —— 你应该把这篇 survey 作为入门文献。

---

## 9. 主流方案横向对比矩阵

| 维度 | PyTorch KernelFalcon/Agent | Sakana AI CUDA Engineer | AMD GEAK | AutoKernel | 华为 LLM4Compiler (Long Cheng) | AscendCraft | AscendKernelGen | AKG(对比基线) |
|---|---|---|---|---|---|---|---|---|
| **目标硬件** | NVIDIA GPU | NVIDIA GPU | AMD GPU(MI300X) | NVIDIA GPU | **Ascend NPU + ARM CPU** | **Ascend NPU** | **Ascend NPU** | **Ascend NPU** |
| **FX graph 利用** | ✅ 显式做子图分解 | ❌ 翻译 PyTorch 到 CUDA | ❌ | ⚠️ | ⚠️ Triton 视角 | ❌ | ❌ | ✅(polyhedral) |
| **Baseline codegen** | ❌ 直接 LLM | ❌ 直接 LLM | ❌ 直接 LLM | ✅ starter kernel | ⚠️ Triton-to-Triton | ✅ DSL → AscendC | ❌ | ❌ |
| **核心机制** | Deep agent + 多 worker | Evolution + RAG | 4-module agent | 6-tier playbook + Amdahl | Math equivalence + RL | DSL-guided transcompile | CoT SFT + RL | Polyhedral + ILP |
| **正确性验证** | ✅ 5 阶段 harness | ✅ LLM soft + hardware | ✅ Cascaded test | ✅ 5 阶段 | ✅ Unit test | ✅ Per-pass 反馈 | ✅ Execution feedback | ⚠️ Static schedule |
| **性能提升**(自报) | 100% correctness KernelBench L1/2/3 | 1.52× median, >50× peak | 2.59× | 5.29×(RMSNorm) | N/A(论文少) | 46.2% > eager | 0% → 95.5% 编译 | 1.6× over TVM |
| **是否开源** | ✅ GitHub | ✅ 17K kernels | ✅ GitHub | ✅ GitHub | ❌(VecTrans 部分开源) | ⚠️ paper only | ✅ 模型权重 + 数据 | ✅ MindSpore 内置 |
| **成熟度** | 🔥 工业级(2026.03) | 🔥 工业级(2025.02) | 🔥 工业级(2025.07) | 🟡 学术 + 开源 | 🔥 华为内部 | 🟡 论文 | 🟡 论文 | ✅ 工业级(2021) |
| **对你方案的可复用度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐(但闭源) | ⭐⭐⭐(DSL 思路) | ⭐⭐⭐⭐(训练范式) | ⭐⭐(fallback only) |

---

## 10. 你的方案与既有方案的关键差异

### 10.1 你方案的三大差异点

#### (1) **NPU + Triton 双桥接** ⭐⭐⭐⭐⭐

- 绝大多数工作(NVIDIA/AMD/Intel)是**单平台 kernel 生成**,目标是"单硬件代码"
- **你方案同时覆盖"NPU 适配 + Triton 优化"** —— 这正是华为 LLM4Compiler 团队做的,但**目前只有这个团队在做,且未开源**
- **差异化机会**:如果你开源,**华为 NPU 生态会有第一个工业级 Agent 图优化框架**

#### (2) **codegen baseline + Agent 优化的双层结构**

- 大多数 Agent 工作是**单层:从 PyTorch → Triton / CUDA,直接由 Agent 生成**
- 你的方案有 **"inducor codegen baseline → Agent 在此基础上优化" 的双层结构**
- **优势**:**永远不会比 inductor baseline 更差**(类似 compiler fallback 的设计哲学),这在工程上是巨大的可解释性优势
- **挑战**:**如何把 baseline 注入 Agent 的 prompt**(目前没有公开方案,这是个创新点)

#### (3) **图切分 + Agent 协同**

- v2 报告里所有 fusion 范围识别工作**都是规则派**;LLM 驱动 fusion 范围识别**学术界无人做**
- 你方案中后期可以把"LLM 做 fusion 范围候选生成"作为新的差异化方向

### 10.2 你的方案的优势(与 PyTorch KernelFalcon 直接对比)

| 维度 | PyTorch KernelFalcon | 你的方案 |
|---|---|---|
| 目标硬件 | NVIDIA GPU | **Ascend NPU(NVIDIA 之外最大生态)** |
| Fusion 范围识别 | heuristic(LLM 辅助) | **inducor can_fuse + 可升级到 TASO/DNNFusion** |
| Baseline | 无显式 baseline | ✅ inductor codegen 作为保底 |
| 图切分 | 简单 dispatch | **TODO(可以接 GSPMD-style partitioner)** |
| 开源 | ✅ | TODO |
| Agent 编排参考 | KernelFalcon architecture | **可完全照抄,但需替换 Triton → Triton-on-Ascend** |

### 10.3 你的方案的风险(已被业界验证)

1. **"iterative refinement 提高正确率但降低 speedup"**(KernelBench-X 结论)
   - 你的 Agent 可能把"正确但比 baseline 慢"的代码当 winner
   - **对策**:**baseline-aware revert** —— 比 baseline 慢 ≥ K% 直接 revert
   
2. **"reward hacking / lazy optimization"**(Dr. Kernel 论文首先定义)
   - Agent 可能输出"绕过真实计算"的 kernel 让 benchmark 数值漂亮
   - **对策**:Strong execution harness(参考 AutoKernel 5 阶段 + KernelGym 的 "must execute Triton kernel in both train/eval mode")

3. **"category 决定正确率方差是 method 的 3 倍"**(KernelBench-X 结论)
   - Fusion 任务 72% 全方法失败
   - **对策**:**baseline 兜底 + 算子分类路由**(简单 op 不进 Agent,直接用 inductor)

4. **"cross-hardware speedup variance up to 21.4×"**(KernelBench-X)
   - NPU 上优化的 kernel 不一定在其他 NPU 上 work
   - **对策**:**multi-NPU 实测验证**(Atlas A2/A3/C310 全跑一遍)

---

## 11. 风险、未解问题、对你的建议

### 11.1 三阶段路线图建议

#### Phase 1(MVP,1-2 个月):复用 + baseline 落地

**目标**:把 PyTorch KernelFalcon 的 agent 编排 fork 过来,把 Triton 替换成 Triton-on-Ascend,跑通 inductor baseline + Agent 优化的完整 pipeline

**具体动作**:
1. **fork** [meta-pytorch/KernelAgent](https://github.com/meta-pytorch/KernelAgent)
2. **替换 Triton 后端** 为 Triton-on-Ascend(`torch_npu` 的 Triton 路径)
3. **保留 inductor codegen 作为 baseline**(直接用 `output_code.py` 钩到 inductor)
4. **接入 MultiKernelBench / KernelBench-Triton 验证 correctness**
5. **NPU 实测性能**(Atlas 800T A2 / Atlas 200T A2 Box16)

**关键风险**:PyTorch KernelFalcon 与 `torch_npu/inductor` 的兼容性问题 —— fork 后需要适配 NPU IR

#### Phase 2(3-5 个月):Agent 优化层完善

**目标**:补全 Agent 闭环(Profiler/Judge/Optimize)+ NPU profile 接入 + 性能对比 baseline

**具体动作**:
1. **接入 NPU profiler**(Ascend 的 profiling 数据转"结构化自然语言"——参考 GEAK 设计)
2. **集成 Dr. Kernel 的 KernelGym-style harness**(防止 reward hacking)
3. **用 (FX subgraph, inductor baseline, profile, optimized Triton) 四元组做 SFT**(参考 AscendKernelGen 范式)
4. **接入 innovation archive / RAG**(参考 Sakana AI CUDA Engineer)
5. **接入 VecTrans-style 数学等价变换**(关键创新点 —— 你有 v2 报告做基础)

#### Phase 3(6-12 个月):差异化创新

**目标**:做"LLM 驱动的 fusion 范围识别"(学术空白) + 图切分 + 多 Agent 协同

**具体动作**:
1. **LLM fusion seed + 传播探索**(v2 报告 § 5.1 推荐方向,加上 LLM 生成候选子图)
2. **图切分 + Agent 协同**(参考 GSPMD priority-based partitioner + AdaSTORM-style LLM 协同)
3. **多硬件 Atlas 系列联合验证**
4. **公开 benchmark 数据集**(类似 KernelBook 的 NPU 版本)
5. **与华为 LLM4Compiler 团队建立 connect**(他们的工作是你的最强对标,**值得主动合作**)

### 11.2 关键技术选型建议

| 选型项 | 推荐 | 理由 |
|---|---|---|
| **基线 codegen** | inductor `output_code.py` 钩子 + KernelBook 风格数据集 | 完全复用 PyTorch 基础设施 |
| **Agent 编排** | PyTorch KernelFalcon(KernelAgent)architecture | 工业级验证,GitHub 开源 |
| **Agent 模块** | Generator / Reflector / Evaluator / Optimizer + ProfilerAgent | GEAK + KernelFalcon 共识 |
| **正确性 harness** | AutoKernel 5 阶段 + KernelGym 反 hacking | 业界最严 |
| **Profile 抽象** | GEAK "结构化自然语言" 风格 | LLM 友好 |
| **训练数据** | (FX, inductor Triton, profile, optimized) 四元组 SFT | 参考 AscendKernelGen |
| **后期 RL** | Dr. Kernel 的 KernelGym + multi-turn RL | 防 reward hacking |
| **图切分**(后期) | GSPMD priority-based + AdaSTORM LLM 协同 | 工业级 + 学术 SOTA |
| **Fusion 范围识别**(后期) | v2 报告 TASO/SMT + LLM 生成候选 | 学术空白点 |

### 11.3 你的工程现状映射

**你已经有的**:
- ✅ torch dynamo / FX graph 解析 → 直接复用
- ✅ inductor `can_fuse` + scheduler → 直接复用
- ✅ inductor codegen `output_code.py` → 直接作为 baseline
- ✅ torch_npu / Ascend 后端 → 已有 PTA 仓
- ✅ torch.compile custom backend(NGO) → 已有基础架构

**你还需要做的**:
- 🔨 Agent 模块实现(参考 KernelFalcon)
- 🔨 NPU profiler 抽象(参考 GEAK)
- 🔨 correctness harness(参考 AutoKernel + KernelGym)
- 🔨 MultiKernelBench 适配(已有,可直接用)
- 🔨 训练数据准备(参考 KernelBook + AscendKernelGen 的 CoT)
- 🔨 🔥 **与华为 LLM4Compiler 团队建立 connect**(关键)

### 11.4 ⚠️ 你的方案未解的工程问题

1. **baseline 注入 Agent prompt 的格式** —— 没有公开方案,需要自己设计
2. **NPU profile 的"结构化自然语言"抽象** —— Ascend profiling 数据需要专门设计
3. **SFT 数据规模** —— AscendKernelGen 的 ~10K 序列规模是否够,需实验
4. **RL reward shaping on NPU** —— Dr. Kernel 的 hacking 检查如何在 Ascend 上对应
5. **多 NPU 硬件一致性** —— Atlas A2/A3/C310 间的性能方差

### 11.5 商业 / 团队层面建议

- **直接对标竞争**:华为 LLM4Compiler 团队(已转 NVIDIA,失去 Ascend focus) + AscendCraft(NJU 学术合作)+ AscendKernelGen(鹏城 + 华为) —— **你的 NGO 项目与 AscendCraft/AscendKernelGen 在工程上几乎平行**,需要差异化(如:NGO 是 compiler-level,他们是 kernel-level)
- **如果开源**,会填补"Ascend NPU 第一开源 LLM-driven 图优化框架"空白
- **避免重复造轮子**:KernelFalcon 已经把 agent 编排做到工业级,**先 fork,再改造**,不要从零写

---

## 附录 A. 信源与可信度

| 类别 | 信源 | 可信度 | 用途 |
|---|---|---|---|
| **官仓源码** | `meta-pytorch/KernelAgent` GitHub | 🟢 高 | KernelFalcon/KernelAgent 全栈实现 |
| **官仓源码** | `meta-pytorch/popcorn-kernels` | 🟢 高 | KernelBook 数据集构建流程 |
| **官仓源码** | `AMD-AGI/GEAK-agent` | 🟢 高 | AMD GEAK v1 实现 |
| **官仓源码** | `facebookresearch/compilergym` | 🟢 高 | CompilerGym 环境抽象 |
| **官仓源码** | `caoshiyi/K-Search` | 🟢 高 | K-Search 实现 |
| **官仓源码** | `SakanaAI` CUDA Engineer | 🟢 高 | Sakana AI 流水线 |
| **官仓源码** | `IntelLabs/Xe-Forge` | 🟢 高 | Intel Xe-Forge |
| **官仓源码** | `AI9Stars/AutoTriton` | 🟢 高 | AutoTriton 模型 |
| **HuggingFace 模型** | `facebook/KernelLLM` | 🟢 高 | KernelLLM 8B 模型 |
| **HuggingFace 数据集** | `GPUMODE/KernelBook` | 🟢 高 | KernelBook 数据 |
| **HuggingFace 模型** | `AscendKernelGen/KernelGen-LM-8B` | 🟢 高 | AscendKernelGen 模型 |
| **arXiv 论文** | KernelBench / KernelBench-X | 🟢 高 | 事实标准 benchmark |
| **arXiv 论文** | AutoKernel(MIT) | 🟢 高 | 端到端 agent + 5 阶段 harness |
| **arXiv 论文** | GEAK(AMD) | 🟢 高 | 4-module agent |
| **arXiv 论文** | Sakana AI CUDA Engineer + robust-kbench | 🟢 高 | 17K kernels + 进化式 |
| **arXiv 论文** | AscendCraft(NJU + Huawei, 2026.01) | 🟢 高 | NPU DSL-guided transcompile |
| **arXiv 论文** | AscendKernelGen(Pengcheng + Huawei, 2026.01) | 🟢 高 | NPU CoT SFT + RL |
| **arXiv 论文** | AKG(PLDI 2021) | 🟢 高 | Huawei MindSpore NPU compiler |
| **arXiv 论文** | MultiKernelBench | 🟢 高 | 多平台 benchmark |
| **arXiv 论文** | VecTrans(arXiv 2503.19449) | 🟢 高 | 华为 LLM4Compiler,BiSheng |
| **arXiv 论文** | TritonPilot(ICS 2026) | 🟢 高 | compiler-assisted LLM |
| **arXiv 论文** | Agentic Code Optimization via Compiler-LLM | 🟢 高 | Compiler-LLM cooperation |
| **arXiv 论文** | Dr. Kernel(2026.02) | 🟢 高 | RL + KernelGym + reward hacking |
| **arXiv 论文** | Towards Automated Kernel Generation in the Era of LLMs(2026) | 🟢 高 | 系统 survey |
| **arXiv 论文** | AdaSTORM(2026.06) | 🟡 中 | LLM dynamic graph reasoning |
| **官方博客** | pytorch.org/blog/kernelagent | 🟢 高 | KernelAgent 设计说明 |
| **官方博客** | pytorch.org/blog/kernelfalcon | 🟢 高 | KernelFalcon 工业部署 |
| **官方博客** | rocm.blogs.amd.com(GEAK) | 🟢 高 | GEAK 系列 |
| **个人主页** | chenglong92.github.io(Long Cheng) | 🟢 高 | 华为 LLM4Compiler 团队负责人 |
| **METR 博客** | metr.org/blog/2025-02-14 | 🟢 高 | METR KernelAgent 1.81× |
| **会议论文** | PLDI / MLSys / EuroLLVM / ASPLOS / SOSP / OSDI | 🟢 高 | 编译系统顶会 |
| **Medium / 个人博客** | Scalable Intelligence, Jack Youstra | 🟡 中 | KernelBench 实操案例 |
| **本调研依赖** | `docs/ai_gen/fusion_range_identification_survey_v2.md` | 🟢 高 | fusion 范围识别基线 |

**调研遵循多源交叉验证原则**:每个核心事实至少 1 个高可信度源 + 1 个中可信度源支撑。

---

## 附录 B. 必读文献清单

按"对你项目相关性"排序:

### B.1 必须精读(★★★)—— 直接对标你的方案

1. **PyTorch KernelFalcon**([pytorch.org/blog/kernelfalcon](https://pytorch.org/blog/kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents)) + **KernelAgent**([github.com/meta-pytorch/KernelAgent](https://github.com/meta-pytorch/KernelAgent)) + PyTorch 博客([kernelagent](https://pytorch.org/blog/kernelagent-hardware-guided-gpu-kernel-optimization-via-multi-agent-orchestration))—— **直接 fork 改造对象**
2. **华为 LLM4Compiler / VecTrans**([arxiv.org/html/2503.19449v2](https://arxiv.org/html/2503.19449v2) + [chenglong92.github.io](https://chenglong92.github.io/))—— **你方案最直接的对标团队**
3. **AMD GEAK**([arxiv.org/pdf/2507.23194v1](https://arxiv.org/pdf/2507.23194v1) + [github.com/AMD-AGI/GEAK-agent](https://github.com/AMD-AGI/GEAK-agent))—— **4-module agent + Profiling 抽象**
4. **AutoKernel**([arxiv.org/html/2603.21331](https://arxiv.org/html/2603.21331))—— **5 阶段 harness + Amdahl 排序**
5. **AscendCraft**([arxiv.org/html/2601.22760v1](https://arxiv.org/html/2601.22760v1))—— **NPU DSL-guided transcompile**
6. **AscendKernelGen**([arxiv.org/pdf/2601.07160](https://arxiv.org/pdf/2601.07160))—— **NPU CoT SFT + RL**

### B.2 应该读(★★)—— 通用机制 / 训练范式

7. **Meta KernelLLM + KernelBook**([huggingface.co/facebook/KernelLLM](https://huggingface.co/facebook/KernelLLM) + [huggingface.co/datasets/GPUMODE/KernelBook](https://huggingface.co/datasets/GPUMODE/KernelBook))—— **codegen baseline 数据集**
8. **KernelBench-X**([arxiv.org/html/2605.04956](https://arxiv.org/html/2605.04956))—— **iterative refinement 不能保证性能**(关键警示)
9. **Sakana AI CUDA Engineer**([pub.sakana.ai/static/paper.pdf](https://pub.sakana.ai/static/paper.pdf))—— **innovation archive / RAG**
10. **Dr. Kernel**([arxiv.org/html/2602.05885v1](https://arxiv.org/html/2602.05885v1))—— **reward hacking / lazy optimization 系统化定义**
11. **Towards Automated Kernel Generation in the Era of LLMs**([arxiv.org/html/2601.15727v3](https://arxiv.org/html/2601.15727v3))—— **系统 survey**
12. **VecTrans**([arxiv.org/html/2503.19449v2](https://arxiv.org/html/2503.19449v2))—— **compiler-LLM 协同**
13. **TritonPilot**([dl.acm.org/doi/abs/10.1145/3774895.3812200](https://dl.acm.org/doi/abs/10.1145/3774895.3812200))—— **compiler analysis → LLM prompt**

### B.3 可选参考(★)—— 后续阶段 / 拓展方向

14. **K-Search**([arxiv.org/html/2602.19128v1](https://arxiv.org/html/2602.19128v1))—— **co-evolving world model**(搜索空间大时用)
15. **Intel Xe-Forge**([arxiv.org/pdf/2605.26118](https://arxiv.org/pdf/2605.26118))—— **9 stage 流水线 + CoVeR 反思**
16. **AdaSTORM**([arxiv.org/pdf/2606.16328v1](https://arxiv.org/pdf/2606.16328v1))—— **图切分 + LLM**(中后期阶段)
17. **METR KernelAgent**([metr.org/blog/2025-02-14](https://metr.org/blog/2025-02-14-measuring-automated-kernel-engineering/))—— **Parallel tree search**
18. **AutoTriton**([arxiv.org/html/2507.05687v1](https://arxiv.org/html/2507.05687v1))—— **SFT + RL 训练范式**
19. **CompilerGym**([arxiv.org/pdf/2109.08267](https://arxiv.org/pdf/2109.08267))—— **RL 环境抽象**(如果你想做 RL-based agent)
20. **MultiKernelBench**([arxiv.org/pdf/2507.17773v1](https://arxiv.org/pdf/2507.17773v1))—— **NPU 评估集**

### B.4 历史基线 ★ —— 对照组

21. **AKG**([dl.acm.org/doi/10.1145/3453483.3454106](https://dl.acm.org/doi/10.1145/3453483.3454106))—— **Huawei MindSpore NPU compiler,无 LLM**(对比 baseline)
22. **v2 fusion range identification survey**([docs/ai_gen/fusion_range_identification_survey_v2.md](./fusion_range_identification_survey_v2.md))—— **fusion 范围识别的所有学术工作**(对比 baseline)

---

*报告生成日期:2026-08-24*
*调研方法:anysearch v3.0.1 多源检索 + 关键论文/官方仓库全文抽取 + arXiv 综述交叉验证*
*下次复审建议:LLM-driven kernel generation 领域进展极快(每月都有新论文),建议每 1-2 月跟踪一次 arXiv 最近投稿 + HuggingFace 最新模型。*

