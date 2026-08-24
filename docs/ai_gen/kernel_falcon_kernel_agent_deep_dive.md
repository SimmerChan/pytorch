# PyTorch KernelFalcon / KernelAgent 深度调研

> **调研主题**:Meta PyTorch 团队发布的两个开源深度 Agent 系统 —— KernelFalcon(代码生成,2025.11)与 KernelAgent(性能优化,2026.03)—— 的完整架构、组件实现、与你的项目的可复用度对比。
>
> **调研范围**:① baseline 是什么 ② 图切分 / 可融合范围识别方案 ③ 具体使用方式(是否用户无感 / 不改模型代码) ④ 完整目录结构与关键组件源码剖析 ⑤ 可直接复用到 Ascend NPU 的部分 ⑥ 你的方案与它的差异点。
>
> **信源**:
> - PyTorch 官方博客 KernelFalcon(2025.11)
> - PyTorch 官方博客 KernelAgent(2026.03)
> - GitHub 仓库 [meta-pytorch/KernelAgent](https://github.com/meta-pytorch/KernelAgent)(Apache 2.0, 522 ⭐, 1.27 MB)
> - 仓库内核心源码:`Fuser/`、`triton_kernel_agent/`、`kernel_perf_agent/`
>
> **v1 报告生成日期**:2026-08-24

---

## 目录

- [0. TL;DR](#0-tldr)
- [1. 两个系统的定位差异](#1-两个系统的定位差异)
- [2. 仓库元数据 + 顶层架构](#2-仓库元数据--顶层架构)
- [3. KernelFalcon 架构详解](#3-kernelfalcon-架构详解)
  - [3.1 核心思想:深度 Agent + Code-to-Code](#31-核心思想深度-agent--code-to-code)
  - [3.2 四阶段 Pipeline](#32-四阶段-pipeline)
  - [3.3 Stage 1:FuserAgent](#33-stage-1fuseragent)
  - [3.4 Stage 2:ExtractorAgent](#34-stage-2extractoragent)
  - [3.5 Stage 3:Dispatcher + KernelAgent](#35-stage-3dispatcher--kernelagent)
  - [3.6 Stage 4:ComposerAgent](#36-stage-4composeragent)
  - [3.7 验证机制:Sentinel "ALL_TESTS_PASSED"](#37-验证机制sentinel-all_tests_passed)
- [4. KernelAgent 架构详解(性能优化)](#4-kernelagent-架构详解性能优化)
  - [4.1 核心思想:硬件信号驱动的优化闭环](#41-核心思想硬件信号驱动的优化闭环)
  - [4.2 六个 Agent 的职责拆分](#42-六个-agent-的职责拆分)
  - [4.3 Beam Search 探索 + Best-of-K](#43-beam-search-探索--best-of-k)
- [5. ⚠️ Baseline 是什么](#5-️-baseline-是什么)
- [6. ⚠️ 图切分 / 融合范围识别方案](#6-️-图切分--融合范围识别方案)
- [7. ⚠️ 使用方式:用户改不改模型代码](#7-️-使用方式用户改不改模型代码)
- [8. 关键目录与源码剖析](#8-关键目录与源码剖析)
  - [8.1 `Fuser/` —— KernelFalcon 主入口](#81-fuser--kernelfalcon-主入口)
  - [8.2 `triton_kernel_agent/` —— 核心 Agent](#82-triton_kernel_agent--核心-agent)
  - [8.3 `kernel_perf_agent/` —— 性能分析](#83-kernel_perf_agent--性能分析)
  - [8.4 `oink/` —— 待确认子项目](#84-oink--待确认子项目)
  - [8.5 `examples/` + `tests/`](#85-examples--tests)
- [9. 平台抽象:为什么它能直接扩到 NPU](#9-平台抽象为什么它能直接扩到-npu)
- [10. 完整调用流程(从用户 CLI 到最终 kernel)](#10-完整调用流程从用户-cli-到最终-kernel)
- [11. 关键设计模式总结(对你 NGO 的借鉴清单)](#11-关键设计模式总结对你-ngo-的借鉴清单)
- [12. ⚠️ 你的方案与 KernelFalcon/KernelAgent 的差异](#12-️-你的方案与-kernelfalconkernelagent-的差异)
- [13. 风险与未公开细节](#13-风险与未公开细节)
- [14. 行动建议](#14-行动建议)
- [附录 A. 信源清单](#附录-a-信源清单)
- [附录 B. 文件清单与行数](#附录-b-文件清单与行数)

---

## 0. TL;DR

**PyTorch KernelFalcon(2025.11)+ KernelAgent(2026.03)** 是 Meta PyTorch Labs 的两个深度 Agent 系统,**同一个仓库** `meta-pytorch/KernelAgent` 维护,代码完全开源(Apache 2.0, 522 ⭐)。

| 问题 | 直接答案 |
|---|---|
| **baseline 是什么?** | **没有显式的 `torch.compile` codegen baseline**。KernelFalcon 的 baseline 就是**原始 PyTorch 模型**(eager 模式);KernelAgent 的 baseline 是**已经验证过的 Triton kernel**(来自 KernelFalcon / 手工)。⚠️ 这与你的"inducor codegen baseline + Agent 优化"的设计**不同** |
| **图切分 / 融合范围识别方案?** | **Stage 1 FuserAgent 用 LLM 做 code-to-code 融合**(在 Python 源码层重组,保留 if/while/动态 shape),**Stage 2 ExtractorAgent 用 LLM 生成 JSON subgraph spec**(含 op 序列 + shape contract);**不是 inductor 风格的 IR 层融合**。⚠️ 这与你的"复用 inductor can_fuse"假设**不同** |
| **是否用户无感?** | **否**。两种使用方式:① **显式调用 CLI/SDK**(`python -m Fuser.pipeline --problem ...`),② **Gradio UI**。**没有 `torch.compile(backend="kernelagent")` 这种零侵入的 backend 集成**。代码会写到一个独立目录,需要手动替换 |
| **可直接复用到 NPU 的部分?** | **很多**:`platform_config.py` 已经是 **registry 模式**(`cuda` / `xpu` 双注册),你**加一个 `npu` 条目 + `nvidia.py` 复制成 `npu.py` 即可扩到 Ascend**。Router 决策、orchestrator、worker protocol 都与硬件解耦 |
| **与你的方案差异?** | 详见 § 12:**最关键的差异 —— KernelFalcon 不接 inductor,你的方案要接**;**最关键的可复用 —— 平台 registry 与多 agent 编排模式** |

---

## 1. 两个系统的定位差异

**同一个 GitHub 仓库** 维护,分成两套不同目标的 pipeline:

| 系统 | 目标 | 输出 | 来源 |
|---|---|---|---|
| **KernelFalcon** | **代码生成**:从 PyTorch 模块生成 Triton kernel,**100% 正确率**(KernelBench L1/L2/L3) | 可部署的 Triton kernel + 自检 harness | [PyTorch 博客 2025.11](https://pytorch.org/blog/kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents/) |
| **KernelAgent** | **性能优化**:从已有 Triton kernel 出发,**用 NCU profiling 反馈迭代优化** | 更快版本的同一 Triton kernel | [PyTorch 博客 2026.03](https://pytorch.org/blog/kernelagent-hardware-guided-gpu-kernel-optimization-via-multi-agent-orchestration) |

**KernelFalcon 数字**:
- KernelBench L1(100/100)/ L2(100/100)/ L3(50/50)**全部 100% 正确率**
- "Performance metrics (speedup, latency) will be covered in our follow-up post"(KernelAgent 博客就是 follow-up)

**KernelAgent 数字**:
- 100 L1 KernelBench tasks:**2.02× speedup** over KernelFalcon-only baseline
- **1.56× over default `torch.compile`**(静态 shape, 关闭 CUDA graphs)
- **89% of H100 roofline efficiency**
- **outperform `torch.compile` in 65/100 tasks**
- 案例研究:matrix-vector mult `A @ x`(M=2048, K=1M, BF16)
  - `torch.compile` baseline:**2.09 ms**
  - KernelFalcon(correctness-only):**9.52 ms**(没优化,只是让它对)
  - LLM-only 8 轮 sequential(opus-4.5):**3.20 ms**
  - **KernelAgent(4 workers × 8 rounds, opus-4.5):1.95 ms**

---

## 2. 仓库元数据 + 顶层架构

### 2.1 元数据

| 项 | 值 |
|---|---|
| **GitHub** | [meta-pytorch/KernelAgent](https://github.com/meta-pytorch/KernelAgent) |
| **License** | Apache 2.0 |
| **Stars / Forks** | 522 / 88 |
| **Open issues** | 13 |
| **Size** | 1.27 MB |
| **Language** | Python |
| **Created** | 2025-07-10 |
| **Last push** | 2026-07-15 |
| **Python** | 3.10 – 3.12 |
| **Dependencies** | openai / anthropic / jinja2 / python-dotenv / gradio>=5.5.0 / requests / numpy / omegaconf |
| **CLI 入口** | `fuser-ui` / `kernel-agent` / `pipeline-ui` / `optimization-ui` / `list-models` |

### 2.2 顶层目录(全部公开)

```
meta-pytorch/KernelAgent/
├── Fuser/                            # KernelFalcon 主入口(代码生成)
│   ├── auto_agent.py       (33 KB)    # AutoRouter:静态分析 → 选哪条 pipeline
│   ├── orchestrator.py     (13 KB)    # 多 worker 并行 orchestrator
│   ├── subgraph_extractor.py(14 KB)   # Stage 2:LLM 生成 subgraph JSON
│   ├── dispatch_kernel_agent.py(18 KB)# Stage 3:分发到 KernelAgent workers
│   ├── compose_end_to_end.py(20 KB)   # Stage 4:Composer 缝合
│   ├── pipeline.py          (7 KB)    # 一站式 CLI:extract → dispatch → compose
│   ├── code_extractor.py    (3 KB)    # 从 LLM 输出抠 Python 代码块
│   ├── worker.py            (9 KB)    # 单 worker 子进程
│   ├── runner.py            (9 KB)    # 在 sandbox 子进程跑 candidate kernel
│   ├── runner_util.py       (5 KB)
│   ├── dedup.py             (2 KB)    # 代码 SHA 去重
│   ├── event_adapter.py     (8 KB)
│   ├── logging_utils.py     (2 KB)
│   ├── paths.py             (2 KB)
│   ├── constants.py         (1 KB)
│   ├── cli.py               (6 KB)
│   ├── config.py            (2 KB)
│   ├── config/                        # 配置目录
│   └── __init__.py
│
├── triton_kernel_agent/              # KernelAgent 核心(代码生成 + 优化)
│   ├── agent.py            (21 KB)    # TritonKernelAgent 类
│   ├── worker.py           (30 KB)    # 单 worker:LLM ↔ 测试循环
│   ├── manager.py          (10 KB)    # Worker 池管理
│   ├── opt_manager.py      (28 KB)    # 优化阶段的 manager(beam search)
│   ├── opt_worker.py       (18 KB)    # 优化阶段的 worker
│   ├── worker_util.py      (8 KB)
│   ├── prompt_manager.py   (15 KB)    # Jinja prompt 渲染
│   ├── platform_config.py  (4 KB)     # ⚠️ 平台注册表(cuda/xpu)
│   ├── platform/                      # 平台实现细节
│   │   ├── __init__.py     (3 KB)
│   │   ├── interfaces.py   (9 KB)
│   │   ├── registry.py     (9 KB)
│   │   ├── noop.py         (6 KB)    # 默认占位
│   │   └── nvidia.py       (23 KB)   # NVIDIA CUDA 专属逻辑
│   ├── opt_worker_component/          # KernelAgent 优化层
│   │   ├── profiling/                 # NCU profiler
│   │   ├── orchestrator/              # 优化主循环
│   │   ├── prescribing/               # Bottleneck 处方
│   │   ├── searching/                 # beam search / greedy
│   │   └── benchmarking/              # CUDA event timing
│   └── templates/                     # ⚠️ Jinja prompt 模板(6 个)
│       ├── kernel_generation.j2 (6 KB)
│       ├── kernel_optimization.j2 (5 KB)
│       ├── kernel_refinement.j2 (4 KB)
│       ├── reflexion_prompt.j2 (3 KB)
│       ├── test_generation.j2 (10 KB)
│       └── triton_guidelines.j2 (17 KB)
│
├── kernel_perf_agent/                # 性能分析工具(roofline 等)
│   └── kernel_opt/
│       └── roofline/                  # SOL 分类与早停
│
├── oink/                             # ⚠️ 待确认的子项目(NPU 相关?)
│   ├── README.md
│   ├── pyproject.toml
│   ├── src/
│   ├── tests/
│   ├── benchmarks/
│   └── .codex/
│
├── examples/                         # 示例问题
│   ├── run_opt_manager.py   (7 KB)    # 优化入口示例
│   ├── configs/
│   ├── optimize_01_matvec/            # 案例研究:matrix-vector
│   ├── optimize_02_rmsnorm/
│   ├── optimize_03_max_pooling/
│   ├── triton_01_element_add.py
│   ├── triton_02_fused_reduction_gemm.py
│   └── triton_03_fused_dcpp.py
│
├── scripts/                          # UI / 工具脚本
├── utils/                            # LLM providers
├── tests/
├── e2e_test.py            # 端到端示例
├── pyproject.toml
├── LICENSE (Apache 2.0)
└── README.md
```

### 2.3 子项目关系图

```
┌──────────────────────────────────────────────────────────────────────┐
│                          用户入口(4 个)                              │
│                                                                      │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐         │
│  │ fuser-ui       │  │ kernel-agent   │  │ pipeline-ui    │         │
│  │ (Gradio)       │  │ (Gradio)       │  │ (Gradio)       │         │
│  └────────┬───────┘  └────────┬───────┘  └────────┬───────┘         │
│           │                   │                   │                  │
│  ┌────────▼───────────────────▼───────────────────▼──────────┐      │
│  │  Fuser.auto_agent (AutoRouter 决策)                          │      │
│  │   ↓ 静态 AST 分析 → 选 ① or ②                               │      │
│  └─────┬───────────────────────────────────────────┬────────────┘      │
│        │                                           │                  │
│  ┌─────▼──────────────┐                  ┌─────────▼───────────┐       │
│  │ ① Fuser pipeline   │                  │ ② Direct KernelAgent │      │
│  │ (完整 4 阶段)       │                  │ (单 op 直接生成)      │       │
│  │ extract→dispatch    │                  │ 走 triton_kernel_    │      │
│  │   →compose          │                  │   agent/manager.py   │      │
│  └─────┬──────────────┘                  └─────────┬───────────┘       │
│        │                                           │                  │
│        └─────────────┬─────────────────────────────┘                  │
│                      ▼                                                │
│  ┌──────────────────────────────────────────────────────────┐        │
│  │ triton_kernel_agent.TritonKernelAgent                     │        │
│  │  ├─ agent.py(主类)                                       │        │
│  │  ├─ manager.py + worker.py(worker 池 + 验证循环)         │        │
│  │  ├─ prompt_manager.py(Jinja prompt 渲染)                │        │
│  │  ├─ platform_config.py(cuda/xpu 双注册)                 │        │
│  │  └─ opt_manager.py + opt_worker.py(优化层,可选)         │        │
│  └──────────────────────────────────────────────────────────┘        │
│                      │                                                │
│                      ▼                                                │
│  ┌──────────────────────────────────────────────────────────┐        │
│  │ 输出:.fuse/<run_id>/                                     │        │
│  │   compose_out/composed_kernel.py(可直接部署的 Triton)    │        │
│  │   + kernels_out/<subgraph>/<worker>(中间产物)            │        │
│  └──────────────────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. KernelFalcon 架构详解

### 3.1 核心思想:深度 Agent + Code-to-Code

**KernelFalcon 博客原文摘要**(核心论点):

> "**Preserve Python semantics**. We stay code-to-code in PyTorch, so if/else, while, data-dependent routing and dynamic shapes remain valid."
>
> "**Verifier-first loop**. A KernelAgent compiles and tests candidate kernels; failures feed back locally; we early-exit on the first numerically correct kernel."
>
> "**Compose and verify end-to-end**. Fused kernels drop in for the original ops, followed by whole-model parity checks before acceptance."

**与传统编译器的根本差异**:

| 维度 | 传统编译器(TorchInductor / XLA / TVM) | KernelFalcon |
|---|---|---|
| **处理对象** | FX Graph / SSA IR(冻结控制流) | **PyTorch 源码**(保留 if/while/动态 shape) |
| **融合范围识别** | `can_fuse` 表 + scheduler 启发式 | **LLM 提议融合方案 + Python 执行验证** |
| **失败模式** | Tracing 把控制流塌缩成单分支 | LLM 提议 → Python 直接 run → 真值 |
| **优化空间** | 预设 IR 类目 + pattern matching | **代码 LLM 任意重写** |
| **Kernel 生成** | 自动 codegen(Triton 模板) | **LLM 直接生成** + 独立 test harness |

### 3.2 四阶段 Pipeline

```
PyTorch nn.Module
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 1: FuserAgent                                         │
│   input:  原始 PyTorch 模型(任意控制流)                     │
│   process: LLM 提议融合 → Python 直接执行验证              │
│   output: Fused PyTorch module(显式 subgraph 边界)        │
│                                                            │
│   关键创新:融合保持在 Python 源层,不降到 IR                 │
└──────────────────────────────────────────────────────────────┘
       │  FusedModel(nn.Module) + 显式 subgraph 函数
       ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 2: ExtractorAgent                                     │
│   input:  FusedModel                                        │
│   process: LLM 分析 fused code → 推断 shape → JSON         │
│   output: subgraphs.json                                    │
│                                                            │
│   关键创新:JSON 是 typed contract                          │
│   - 每个 subgraph 有独立 id + 完整 shape + dtype + 权重     │
│   - dedup by (ops + shapes + weights) 签名                  │
└──────────────────────────────────────────────────────────────┘
       │  List[SubgraphSpec]
       ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 3: Dispatcher + KernelAgent (parallel)               │
│   for each subgraph:                                        │
│     spawn N 个 KernelAgent worker(默认 4)                   │
│     每个 worker 独立 LLM 采样 + sandbox 执行 + 反馈         │
│     第一个 PASS 的 worker → 触发 winner event → 其他 cancel │
│                                                            │
│   output: verified Triton kernel + test harness            │
└──────────────────────────────────────────────────────────────┘
       │  List[VerifiedKernel]
       ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 4: ComposerAgent                                      │
│   input:  original problem + subgraphs.json + kernels       │
│   process: LLM 合成 single Triton program                   │
│   output: composed_kernel.py                                │
│           (含 kernel_function + self-test harness)         │
│                                                            │
│   验证:执行 composed_kernel.py,check "PASS" + exit 0       │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
composed_kernel.py (可直接部署)
```

### 3.3 Stage 1:FuserAgent

**FuserAgent 的本质**:LLM 提议的 PyTorch code refactoring + Python 直接执行验证。

**输入**:原始 PyTorch 模型(可能含 if/while/dynamic shape)。

**处理流程**(从 `Fuser/orchestrator.py` + 博客):
1. 解析 PyTorch 模型的 AST
2. **LLM 提议融合方案**(输出新的 PyTorch 代码,带显式 subgraph 边界)
3. **Python 直接执行** 验证(catch 异常、比较输出)
4. 通过 → 进入下一阶段;失败 → 反馈给 LLM 重试

**关键代码(从博客提取)**:

```python
# 输入(原始 PyTorch,任意控制流)
class Model(nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            x = self.conv(x)
            x = self.bn(x)
            x = torch.tanh(x)
            x = F.max_pool2d(x, 2)
        return self.norm(x)

# 输出(LLM 重构后,显式 subgraph 边界)
class FusedModel(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.branch = ConvBnTanhMaxPool(channels=channels)
        self.norm = ChannelwiseNorm(channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:  # 控制流保持
            x = self.branch(x)
        return self.norm(x)

# 对应 subgraph 子模块(LLM 也生成)
class ConvBnTanhMaxPool(nn.Module):
    def forward(self, x):
        x = F.conv2d(x, self.conv_w, stride=1, padding=1)
        x = F.batch_norm(x, self.bn_rm, self.bn_rv, self.bn_w, self.bn_b,
                         training=False, eps=self.eps)
        x = torch.tanh(x)
        return F.max_pool2d(x, 2)
```

**与传统编译器融合的对比**(博客原文):

> "Unlike our prompt-driven approach that keeps the Python `if` and just inserts fused submodules inside it, traditional compiler-based fusion tends to either specialize to a single branch during tracing or require significant manual effort to encode control flow explicitly."

| 维度 | TorchDynamo / torch.compile | KernelFalcon FuserAgent |
|---|---|---|
| 控制流处理 | 塌缩成 graph break + guard | **保持 Python if** |
| 失败回退 | 重 trace(recompilation) | LLM 重提议 + Python 重试 |
| 调试性 | 看 IR(变量名丢失) | **看 Python 源码** |

**Deep Agent 原则体现**:Deterministic control plane(orchestration logic 全是 Python 代码,LLM 只生成 candidate code)。

### 3.4 Stage 2:ExtractorAgent

**本质**:**LLM 把 fused PyTorch code 翻译成 JSON contract**。

**输出 JSON 格式**(从博客原文):

```json
[
  {
    "id": "sg_conv_bn_tanh_pool_1",
    "type": "Conv2d_BN_Tanh_MaxPool",
    "data_layout": "NCHW",
    "dtype": "float32",
    "ops": [
      {"op": "conv2d", "kernel_size": [3, 3], "stride": [1, 1],
       "padding": [1, 1], "dilation": [1, 1], "groups": 1, "bias": false},
      {"op": "batch_norm", "eps": 1e-5, "momentum": 0.1},
      {"op": "tanh"},
      {"op": "max_pool2d", "kernel_size": [2, 2], "stride": [2, 2]}
    ],
    "input_shape": ["B", "C_in", "H", "W"],
    "output_shape": ["B", "C_out", "H_out", "W_out"],
    "weights_original": {
      "conv.weight": ["C_out", "C_in", 3, 3],
      "batch_norm.weight": ["C_out"],
      "batch_norm.bias": ["C_out"],
      "running_mean": ["C_out"],
      "running_var": ["C_out"]
    },
    "weights_fused": null,
    "count": 1,
    "where": "Model.forward conditional branch",
    "source": {
      "module": "FusedConvBnTanhPool",
      "code": "def forward(self, x):\n  x = F.conv2d(...)..."
    }
  }
]
```

**关键设计**:
- **dedup by (ops + shapes + weights) signature**:相同 op 序列 + 相同 shape → 同一个 subgraph,只生成一次 kernel
- 每个 subgraph 自带 `source.code` 字段(原始 fused module 的代码)→ 给 Stage 3 worker 当 prompt 上下文

**对你方案的意义**:
- **这套 JSON schema 可以直接借鉴** —— 你的 Stage 2 输出可以是同样的格式
- **区别在于 source 来源**:KernelFalcon source = fused PyTorch;**你的 source 可以是 inductor codegen 出的 Triton baseline**

### 3.5 Stage 3:Dispatcher + KernelAgent

**核心机制**(并行探索 + 早停):

```
subgraphs.json (from Stage 2)
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ Dispatcher (Fuser/dispatch_kernel_agent.py)                 │
│                                                              │
│   for each subgraph in subgraphs.json:                       │
│     spec = build_prompt(subgraph)                            │
│     spawn N workers (default 4) ──────────┐                  │
│       each with different temperature:    │                  │
│       temp = [0.8, 0.9, 1.0, ...]         │                  │
│                                            │                  │
│     ┌──────────────────────────────────────▼───────────────┐ │
│     │  per-worker (worker.py):                            │ │
│     │    1. LLM 生成 kernel.py + test_kernel.py           │ │
│     │    2. 写到自己 workdir(kernel.py + test_kernel.py)  │ │
│     │    3. subprocess run test_kernel.py                  │ │
│     │       exit 0 + "ALL_TESTS_PASSED" → 推送 winner    │ │
│     │       fail → 错误信息反馈回 LLM,下一轮              │ │
│     │    4. max_rounds (default 10) 后放弃                │ │
│     └──────────────────────────────────────────────────────┘ │
│                                            │                  │
│     winner event ──────────────────────────┘                  │
│       └─ 触发其他 workers 的 terminate                       │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
   kernels_out/<subgraph_id>/<worker_id>/
     ├── kernel.py              (verified Triton kernel)
     ├── test_kernel.py        (LLM 生成的 test harness)
     ├── prompt_round_*.txt    (历史 prompt)
     ├── reply_round_*.txt     (历史 LLM 响应)
     └── summary.json          (成功/失败 + 耗时)
```

**关键设计**:

1. **Local error feedback 防 context pollution**:
   - 每个 worker 在自己的 workdir 工作
   - Worker 2 的错误**只反馈给 Worker 2**,其他 worker 上下文干净
   - 用 subprocess 跑测试 → 隔离 GPU 资源

2. **Early termination 节省算力**:
   - 共享 `success event`(multiprocessing.Event)
   - 第一个 PASS 的 worker 设置 event,其他 worker 在下一轮检查 event 后主动退出

3. **Temperature 多样性探索**:
   - 不同 temperature 引导 worker 走不同优化策略(保守 vs 探索)

4. **DISALLOWED_TORCH_PATTERNS 防御**(摘自 `triton_kernel_agent/worker.py`):
   - 静态正则检查禁止 kernel 里出现 `import torch.nn` 等高层 API
   - 因为 `@triton.jit` 函数不能含 PyTorch 操作(否则编译失败),这是结构性保证

### 3.6 Stage 4:ComposerAgent

**本质**:LLM 缝合多个 verified Triton kernel 成一个完整 program。

**输入**:
- 原始 problem(PyTorch 模型)
- `subgraphs.json`(Stage 2 输出)
- 每个 subgraph 的 verified Triton kernel(Stage 3 输出)

**输出结构**:
```python
# composed_kernel.py

import torch
import triton
import triton.language as tl

@triton.jit
def fused_conv_bn_tanh_pool_kernel(...):
    # Stage 3 生成的 Triton kernel #1
    ...

@triton.jit
def channelwise_norm_kernel(...):
    # Stage 3 生成的 Triton kernel #2
    ...

def kernel_function(input_a, weight, ...):
    """
    包装函数,匹配原始 Model.forward 的签名
    """
    out1 = torch.empty(...)
    fused_conv_bn_tanh_pool_kernel[grid1](input_a, weight, out1, ...)
    out2 = torch.empty(...)
    channelwise_norm_kernel[grid2](out1, out2, ...)
    return out2

# Self-test harness
if __name__ == "__main__":
    torch.manual_seed(42)
    ref = build_pytorch_reference(...)
    out = kernel_function(*build_inputs(...))
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
    print("ALL_TESTS_PASSED")
```

**验证**(博客原文):
- `--verify` flag 开启时:**Python 立即执行 `composed_kernel.py`**,检查 `PASS` in stdout + exit 0
- dtype-specific tolerance:**fp32 严 / fp16 松**,匹配 PyTorch 内部测试标准

### 3.7 验证机制:Sentinel "ALL_TESTS_PASSED"

**这是 KernelFalcon 的核心正确性保障**(摘自 `Fuser/runner.py`):

```python
_SENTINEL = "ALL_TESTS_PASSED"
_PASS_REGEX = re.compile(r"\bPASS\b")

@dataclass(frozen=True)
class RunResult:
    rc: int
    ...
```

**运行流程**:
1. Worker 生成 `kernel.py` + `test_kernel.py`
2. Worker 在 sandbox subprocess 执行 `python test_kernel.py`
3. 成功标准:`exit_code == 0` AND `stdout contains "ALL_TESTS_PASSED"`
4. 失败:`stderr` 截断 20KB,反馈给 LLM 进下一轮
5. 防止"作弊":LLM 生成的 test harness 自己也要 **真比较 kernel 输出 vs PyTorch reference**(reference 来自 `subgraphs.json.source.code` 的 fused PyTorch)

**为何可信**:
- **`@triton.jit` 函数**不能含 PyTorch ops(结构性禁止)
- **subprocess 隔离**:失败/超时不会污染主进程
- **真实 GPU 执行**:不是 LLM 模拟判断,而是真跑 NVIDIA CUDA

---

## 4. KernelAgent 架构详解(性能优化)

### 4.1 核心思想:硬件信号驱动的优化闭环

**KernelAgent 博客原文**(三大设计原则):

> "**Ground everything in hardware metrics**. Both bottleneck diagnosis and optimization prescriptions must be derived from real profiling data."
>
> "**Explore optimization paths in parallel**. Given the same hardware signals, multiple valid optimization strategies may exist. KernelAgent evaluates these strategies concurrently."
>
> "**Learn across rounds through shared memory**. Optimization agents reflect on what succeeded and failed in each round, summarizing insights into a shared memory that guides subsequent iterations."

### 4.2 六个 Agent 的职责拆分

```
                ┌──────────────────────────────────────────────────────────┐
                │                    优化循环每轮                           │
                └──────────────────────────────────────────────────────────┘

   ┌─────────────────┐
   │ ProfilerAgent   │  NCU 收集 28 个硬件指标
   │ (opt_worker_    │  (compute utilization, memory bandwidth, cache hit
   │  component/     │   rate, occupancy, stall breakdowns)
   │  profiling/)    │
   └────────┬────────┘
            │ NCU metrics dict
            ▼
   ┌─────────────────┐
   │ JudgeAgent      │  Roofline 分析 → 分类瓶颈类别
   │ ("Diagnose")    │  (memory-bound / compute-bound / underutilized)
   │                 │  + LLM-based reasoning 找 root cause
   └────────┬────────┘
            │ BottleneckReport
            │   {category, efficiency%, root_causes[]}
            ▼
   ┌─────────────────┐
   │ AnalyzeAgent    │  LLM 基于 BottleneckReport + GPU 规格
   │ ("Prescribe")   │  + curated optimization DB
   │                 │  → 生成具体修复方案 + rationale
   └────────┬────────┘
            │ List[Fix] with rationale
            ▼
   ┌─────────────────┐
   │ Orchestrator    │  合成历史 + 当前 prescription
   │ Agent           │  + 选搜索策略(beam / greedy)
   │                 │  + Reflexion: "was_diagnosis_correct?" 等
   └────────┬────────┘
            │ 最终 prompt
            ▼
   ┌─────────────────┐
   │ Optimization    │  维护 top-K kernels
   │ Manager         │  spawn M 个 worker × N 个不同 fix
   │ (opt_manager.py)│  并行探索
   └────────┬────────┘
            │ 多 candidate kernels
            ▼
   ┌─────────────────┐
   │ BenchmarkAgent  │  correctness check + warmup(25)
   │ ("Measure")     │  + 100 iterations timing
   │                 │  + divergence-based revert
   └────────┬────────┘
            │ best-so-far kernel
            ▼
   ┌─────────────────┐
   │ 持久化到 .optimize/workers/<worker_id>/<run_id>/   │
   │   artifacts/kernel_round_N.py                       │
   │   artifacts/round001_opt_prompt.txt                 │
   │   artifacts/round001_opt_reply.txt                  │
   │   artifacts/round001_strategy.json                  │
   └────────────────────────────────────────────────────┘
```

### 4.3 Beam Search 探索 + Best-of-K

**Algorithm**(博客原文 + README):

```python
# pseudocode
top_k = [baseline_kernel]
for round in range(MAX_ROUNDS):
    # 1. 对每个 top-K kernel,做 NCU profile → bottleneck diagnosis
    jobs = []
    for kernel in top_k:
        for fix in propose_fixes(kernel):  # M 个 fix
            jobs.append(apply_fix(kernel, fix))
    # 2. 并行执行(默认 4 worker × M fix)
    candidates = parallel_run(jobs)
    # 3. correctness + benchmark 筛
    valid = [c for c in candidates if c.correctness and c.runtime < best_so_far * 1.0]
    # 4. divergence-based revert: 退化 ≥ 阈值就 revert
    top_k = select_top_k(valid + top_k, k=K)
    # 5. 早停条件:达到 roofline (≥95% SOL) 或性能收敛
    if reached_roofline(top_k[0]): break
```

**实际案例**(博客 matrix-vector case study):

```
Round 1: 4/4 workers succeeded, best new: 7.8 ms (9.52 → 7.8)
Round 2: 4/4 workers succeeded, best new: 4.05 ms
Round 3: 4/4 workers succeeded, best new: 3.11 ms
...
Final: 1.95 ms (vs 2.09 ms torch.compile baseline)
```

**与 LLM-only baseline 对比**:
- LLM-only(sequential 8 rounds, opus-4.5):**3.20 ms**(陷入 local minima)
- KernelAgent(4 workers × 8 rounds):**1.95 ms**(多 worker 并行逃出 local minima)

---

## 5. ⚠️ Baseline 是什么

### 5.1 直接答案

**KernelFalcon 没有"inducor codegen baseline"概念**。Baseline 是以下两种之一:

| 系统 | Baseline | 用途 |
|---|---|---|
| **KernelFalcon** | **原始 PyTorch 模型**(eager 模式) | Stage 4 compose 时,`composed_kernel.py` 跑出来跟原始 PyTorch 比 |
| **KernelAgent** | **已经验证过的 Triton kernel**(由 KernelFalcon 生成 / 人工写) | 每轮:优化的 kernel 跑得比 baseline 快多少 |

### 5.2 ⚠️ 这与你的方案的关键差异

**你的方案**:
```
FX graph → inductor codegen → Triton baseline (保底) → Agent 在此基础上优化
```

**KernelFalcon / KernelAgent**:
```
PyTorch module → 直接 LLM 生成 Triton kernel(没有 inductor 这一步)
                  ↑↓ 跟原始 PyTorch 对比验证
```

**没有"性能保底"概念**:
- 如果 LLM 生成的 kernel **编译失败 / 跑错**:重试
- 如果 LLM 生成的 kernel **正确但比 eager 慢**:照样接受(100% correctness 优先)
- **不接受"比 baseline 慢"的 kernel 被自动 revert**(没有 baseline-aware revert)

### 5.3 KernelAgent 的 baseline-aware revert 机制

**只对 KernelAgent(优化层)有**,不在 KernelFalcon(代码生成层):

> "**Benchmarking Agent** validates correctness and measures real performance for each kernel variant produced during exploration... tracking best-so-far with **divergence-based revert**"

但这里的 baseline **不是 inductor codegen**,而是**当前轮的 best-so-far kernel**:
- 每轮:用 NCU 测所有 candidate
- 比 baseline(best-so-far)**慢 ≥ 阈值** → revert(保留前一轮)
- "divergence-based" 具体阈值未在博客公布,但语义是"不能比上一轮差太多"

### 5.4 ⚠️ 你的方案可以借鉴/改进的地方

| 你方案的特性 | KernelFalcon/KernelAgent 是否支持 | 你的优势 |
|---|---|---|
| inductor codegen 作为保底 baseline | ❌ 不支持 | **你的方案更强** —— 至少不会比 inductor 慢 |
| 比 baseline 慢就 revert | ⚠️ KernelAgent 只对 best-so-far revert | **你的方案更安全** |
| 比 inductor fused baseline 还慢 | ❌ KernelFalcon 不感知 inductor | **你的方案能进一步优化 inductor 没融合的部分** |

**结论**:**你的"inducor codegen baseline + Agent 优化"设计比 KernelFalcon 更稳健**。可以**借鉴 KernelAgent 的 best-so-far 机制,再叠加 inductor baseline 兜底**。

---

## 6. ⚠️ 图切分 / 融合范围识别方案

### 6.1 KernelFalcon 的方案:LLM-driven code refactoring

**KernelFalcon 根本没用 inductor 风格的 IR 融合**。它的"融合范围识别"分两层:

#### 第一层:FuserAgent 提议融合边界(LLM)

- **Stage 1 输入**:原始 PyTorch 模型源码(可能含 if/while/动态 shape)
- **Stage 1 处理**:LLM 重写 PyTorch 源码,把可融合的算子组合成显式 submodule(例如 `ConvBnTanhMaxPool`)
- **Stage 1 输出**:`FusedModel`(原始控制流 + 显式 subgraph 边界)

**与传统融合的本质差异**(博客原文):

> "Traditional compiler-based fusion tends to either specialize to a single branch during tracing or require significant manual effort to encode control flow explicitly... When TorchScript lowers to SSA form, your carefully named `hidden_states` becomes `t0`."

#### 第二层:ExtractorAgent 提取 subgraph spec(LLM)

- **Stage 2 输入**:FusedModel
- **Stage 2 处理**:LLM 推断每个 submodule 的 op 序列、shape、dtype、权重,生成 JSON
- **Stage 2 输出**:`subgraphs.json`(typed contract,含 dedup signature)

### 6.2 ⚠️ 你的方案与 KernelFalcon 的差异

| 维度 | 你的方案 | KernelFalcon |
|---|---|---|
| 图解析起点 | `torch.compile` / dynamo / FX graph | **PyTorch 源码**(不经过 dynamo) |
| 可融合范围识别 | **复用 inductor `can_fuse` + scheduler** | **LLM 提议融合**(无 inductor) |
| 控制流处理 | 走 inductor(dynamo graph break) | **保持 Python if** |
| Shape 推断 | inductor 内置 | **LLM 推断**(可能不准确) |
| 与 inductor 兼容 | ✅ 完全兼容 | ❌ 完全不依赖 inductor |

### 6.3 ⚠️ KernelFalcon 不走 inductor 的代价

**优势**:
- 保留 Python 语义(动态 shape、MoE routing、TreeLSTM 等)
- 不依赖 inductor 维护性(每次 PyTorch 升级 inductor 改 rules,KernelFalcon 不受影响)
- 调试友好(看到的是 Python,不是 IR)

**劣势**:
- **没有 inductor 的 codegen baseline 兜底**
- **不能利用 inductor 已经做的 fusion 优化**
- **LLM 提议的融合质量不稳定**(没有 can_fuse 那种结构化保证)
- **ExtractorAgent 的 shape 推断可能错**(LLM 幻觉)

### 6.4 AutoRouter:静态分析选择走哪条路

**`Fuser/auto_agent.py` 全文 docstring**(路由器决策逻辑):

```python
"""
Auto-routing agent that decides whether to:
  - Solve a KernelBench-style problem directly with KernelAgent
  - Or run the full Fuser pipeline (extract → dispatch → compose)

Decision is based on a lightweight static analysis of the problem file:
  - Parse the problem as Python AST, inspect Model.__init__/forward
  - Count presence of ops commonly hard to fuse (conv_transpose2d, attention, group_norm chains)
  - Detect control flow in forward (if/for/while)
  - Approximate operation chain length (number of sequential transformations)

Routing policy (conservative):
  - Route to Fuser if any of:
      * attention-like patterns (softmax over QK, multihead attention, einsum with bmm)
      * conv_transpose2d present
      * group_norm used together with conv/conv_transpose or long chains (>=4 steps)
      * explicit control flow in forward
  - Otherwise route to KernelAgent directly

If the chosen path fails, the agent can optionally fall back to the other path.
"""
```

**决策矩阵**(从 docstring 提取 + 推理):

| 特征 | → Fuser pipeline | → 直接 KernelAgent |
|---|---|---|
| attention-like(softmax over QK / multihead / einsum bmm) | ✅ | |
| conv_transpose2d | ✅ | |
| group_norm + conv/conv_transpose OR 长链(≥4 步) | ✅ | |
| **控制流**(if/for/while in forward) | ✅ | |
| 简单 op 链(无控制流,无非 conv/attention) | | ✅ |

**对你方案的借鉴**:
- 你的 `AutoRouter` 可以**完全照抄这套决策**,只是把"→ Fuser pipeline"换成"→ 你方案的 fusion 优化 pipeline"

### 6.5 AutoAgent 的 fallback 机制

**`--no-fallback` flag** 控制:**如果选的那条路失败,要不要 fallback 到另一条**。
默认应该是 fallback(更稳)。

---

## 7. ⚠️ 使用方式:用户改不改模型代码

### 7.1 直接答案:**用户不改模型代码,但要改调用方式**

**没有 `torch.compile(backend="kernelagent")` 这种零侵入集成**。

### 7.2 三种使用方式

#### 方式 ①:CLI 一站式(最常用)

```bash
# AutoRouter:静态分析自动选路径
python -m Fuser.auto_agent \
  --problem /abs/path/to/KernelBench/level1/19_ReLU.py \
  --no-router-cache \
  --verify

# 显式三阶段:
python -m Fuser.pipeline \
  --problem /abs/path/to/problem.py \
  --extract-model gpt-5 \
  --dispatch-model o4-mini \
  --dispatch-jobs auto \
  --compose-model o4-mini \
  --workers 4 \
  --max-iters 5 \
  --verify

# Intel XPU 平台:
python -m Fuser.pipeline \
  --problem /abs/path/to/problem.py \
  --target-platform xpu \
  [--extract-model gpt-5 ...]
```

**输出**:`.fuse/<run_id>/compose_out/composed_kernel.py`

#### 方式 ②:Python SDK(直接调用 KernelAgent)

```python
from triton_kernel_agent import TritonKernelAgent

agent = TritonKernelAgent(num_workers=4, max_rounds=8, model_name="gpt-5")
result = agent.generate_kernel(
    problem_description="Implement ReLU over a contiguous 1D tensor of length 1024"
)

if result["success"]:
    print("Kernel path:", result["kernel_path"])
    print("Session directory:", result["session_dir"])
```

#### 方式 ③:Gradio UI(交互式)

```bash
kernel-agent        # 或 python scripts/triton_ui.py
fuser-ui            # 或 python scripts/fuser_ui
pipeline-ui         # 或 python scripts/pipeline_ui
optimization-ui     # 或 python scripts/optimization_ui --port 8085
```

### 7.3 ⚠️ 使用方式的限制

**没有的集成模式**:
- ❌ `torch.compile(backend="kernelagent")` —— **不是 torch.compile backend**
- ❌ `model = KernelAgentModel(model)` 装饰器模式 —— 不存在
- ❌ 自动替换算子 / 透明融合 —— **不能**

**实际的工作流**:
1. 用户写好 PyTorch 模型
2. 把模型(或单个 op)喂给 KernelFalcon / KernelAgent
3. Agent 生成 `composed_kernel.py`(独立 Triton 文件)
4. 用户**手动**修改模型,用 Triton kernel 替换原算子

### 7.4 与你的方案对比

| 维度 | KernelFalcon/KernelAgent | 你的 NGO + Agent 方案 |
|---|---|---|
| 用户改模型代码? | ❌ Agent 输出独立文件,手动替换 | **✅ 你可以做成 `torch.compile(backend="ngo")`**(因为复用 dynamo) |
| 调用方式 | CLI / SDK / Gradio | **直接 `torch.compile(model, backend="ngo")`** |
| 部署复杂度 | 中等(用户要把输出文件接回模型) | **低(零侵入)** |
| 性能保证 | LLM 生成的 kernel,无兜底 | **inducor codegen 兜底** |

**结论**:**你的方案在使用方式上比 KernelFalcon 更优** —— 你可以做到真正的 `torch.compile` backend 集成,KernelFalcon 不能。

---

## 8. 关键目录与源码剖析

### 8.1 `Fuser/` —— KernelFalcon 主入口

| 文件 | 行数 | 作用 |
|---|---|---|
| `pipeline.py` | ~7 KB | 一站式 CLI:extract → dispatch → compose |
| `auto_agent.py` | ~33 KB | AutoRouter + 完整流程编排(代码量最大) |
| `orchestrator.py` | ~13 KB | 多 worker 并行 orchestrator(用 `multiprocessing`) |
| `subgraph_extractor.py` | ~14 KB | Stage 2:LLM 生成 subgraph JSON + dedup |
| `dispatch_kernel_agent.py` | ~18 KB | Stage 3:分发 subgraph 到 KernelAgent workers |
| `compose_end_to_end.py` | ~20 KB | Stage 4:Composer 缝合 + verify |
| `worker.py` | ~9 KB | Fuser 的单 worker(不同于 KernelAgent worker) |
| `runner.py` | ~9 KB | Sandbox subprocess 执行 candidate kernel |
| `code_extractor.py` | ~3 KB | 从 LLM 输出抠 ```python 代码块 + canonicalize(SHA dedup) |
| `runner_util.py` | ~5 KB | Multiprocess runner util |
| `cli.py` | ~6 KB | CLI 参数 |
| `config.py` | ~2 KB | `OrchestratorConfig` + `WorkerConfig` dataclass |
| `paths.py` | ~2 KB | Run 目录管理 |
| `dedup.py` | ~2 KB | Code dedup by SHA |
| `event_adapter.py` | ~8 KB | 多进程 event adapter(winner 通知) |
| `logging_utils.py` | ~2 KB | JSONL/JSON 日志 |
| `constants.py` | ~1 KB | Sentinel "ALL_TESTS_PASSED" 等 |

**关键 dataclass**(`Fuser/config.py`):

```python
@dataclass
class OrchestratorConfig:
    problem_path: Path
    model: str
    workers: int = 4
    max_iters: int = 10
    llm_timeout_s: int = 120
    run_timeout_s: int = 180
    stream_mode: str = "all"  # all|winner|none
    store_responses: bool = False
    isolated: bool = False
    deny_network: bool = False
    enable_reasoning_extras: bool = True
    target_platform: str = "cuda"  # ← 这是你加 "npu" 的入口

@dataclass
class WorkerConfig:
    run_id: str
    worker_id: str
    variant_index: int
    model: str
    max_iters: int
    llm_timeout_s: int
    run_timeout_s: int
    store_responses: bool
    isolated: bool
    deny_network: bool
    enable_reasoning_extras: bool
    stream_dir: Path
    workspace_dir: Path
    shared_digests_dir: Path
    target_platform: str = "cuda"
```

**`store_responses` flag**:是否保存每轮 LLM 的 prompt/response(调试用,默认 false 节省磁盘)。

### 8.2 `triton_kernel_agent/` —— 核心 Agent

| 文件 | 行数 | 作用 |
|---|---|---|
| `agent.py` | ~21 KB | `TritonKernelAgent` 主类(API 入口) |
| `manager.py` | ~10 KB | Worker 池 + multiprocessing 调度 |
| `worker.py` | ~30 KB | 单 worker:LLM → 生成 → 测试 → 反馈循环 |
| `opt_manager.py` | ~28 KB | 优化层 manager(beam search) |
| `opt_worker.py` | ~18 KB | 优化层 worker(NCU + LLM + 测试) |
| `prompt_manager.py` | ~15 KB | Jinja2 prompt 渲染 |
| `worker_util.py` | ~8 KB | Format test code for LLM 等工具 |
| `platform_config.py` | ~4 KB | ⚠️ 平台注册表(cuda/xpu) |
| `platform/` | 5 文件 | 平台实现接口 + NVIDIA 专属 |

**`TritonKernelAgent` 主类签名**(摘自 `agent.py`):

```python
class TritonKernelAgent:
    """Main agent for generating and optimizing Triton kernels."""

    def __init__(
        self,
        num_workers: int | None = None,
        max_rounds: int | None = None,
        log_dir: str | None = None,
        ...
    ):
        ...

    def generate_kernel(
        self,
        problem_description: str,
        ...
    ) -> dict:
        """Generate a Triton kernel from problem description."""
        # Returns: {"success": bool, "kernel_path": str, "session_dir": str, "message": str}
```

### 8.3 `kernel_perf_agent/` —— 性能分析

**Roofline 分析 + SOL 分类 + 早停**:
- `kernel_perf_agent/kernel_opt/roofline/`
- 输入:NCU 28 个指标
- 输出:`{category: "memory"|"compute"|"underutilized", efficiency_pct: float}`

### 8.4 `oink/` —— 待确认子项目

**结构**:
```
oink/
├── README.md (5 KB)
├── pyproject.toml (1.6 KB)
├── src/
├── tests/
├── benchmarks/
└── .codex/  (?)
```

⚠️ **oink 目录独立 pyproject,不在主 pyproject 里**。`oink` 字面意思可能是"On-device INference Kernel"或类似。**未抓取 README 内容**(curl 被拦截),**需要后续确认是否与 NPU 相关**。

### 8.5 `examples/` + `tests/`

**examples**:
- `run_opt_manager.py` —— 优化层入口示例
- `triton_01_element_add.py` —— 简单元素加
- `triton_02_fused_reduction_gemm.py` —— 融合 reduction + GEMM
- `triton_03_fused_dcpp.py` —— 融合 DCPP(?)

**examples/configs/**:`optimize_01_matvec/`、`optimize_02_rmsnorm/`、`optimize_03_max_pooling/`

### 8.6 Jinja prompt 模板(6 个)

| 模板 | 大小 | 用途 |
|---|---|---|
| `kernel_generation.j2` | 6 KB | 初次生成 kernel 的 prompt |
| `kernel_optimization.j2` | 5 KB | 优化轮 prompt(基于 NCU 反馈) |
| `kernel_refinement.j2` | 4 KB | refine prompt(基于失败反馈) |
| `reflexion_prompt.j2` | 3 KB | self-analysis("was_diagnosis_correct?" 等) |
| `test_generation.j2` | 10 KB | 让 LLM 生成 test harness 的 prompt |
| `triton_guidelines.j2` | 17 KB | ⚠️ Triton 编程指南(最大,工程量集中) |

**`triton_guidelines.j2` 17 KB** —— 这是 KernelFalcon 维护的 Triton 编程指南,作为 in-context examples 给 LLM 用。**这份模板是你可以直接借鉴的最有价值的资产** —— 你把它改成"AscendC 编程指南"或"Triton-on-Ascend 编程指南"就能直接用。

---

## 9. 平台抽象:为什么它能直接扩到 NPU

### 9.1 PlatformConfig 注册表(`triton_kernel_agent/platform_config.py`)

```python
@dataclass(frozen=True)
class PlatformConfig:
    name: str
    device_string: str           # "cuda" or "xpu"
    guidance_block: str          # 平台特定的 CRITICAL REQUIREMENTS
    kernel_guidance: str         # 平台特定的优化指南
    cuda_hacks_to_strip: tuple = field(default_factory=tuple)

PLATFORMS: dict[str, PlatformConfig] = {
    "cuda": PlatformConfig(
        name="cuda",
        device_string="cuda",
        guidance_block="",
        kernel_guidance="",
        cuda_hacks_to_strip=(),
    ),
    "xpu": PlatformConfig(
        name="xpu",
        device_string="xpu",
        guidance_block=_XPU_GUIDANCE,
        kernel_guidance=_XPU_KERNEL_GUIDANCE,
        cuda_hacks_to_strip=_XPU_CUDA_HACKS,
    ),
}

def get_platform(name: str) -> PlatformConfig:
    if name not in PLATFORMS:
        raise ValueError(f"Unknown platform '{name}'. Available: {', '.join(sorted(PLATFORMS.keys()))}")
    return PLATFORMS[name]
```

**对你扩 NPU 的指引**(从 `xpu` 条目推断):

```python
# 你需要加:
_NPU_GUIDANCE = """
**CRITICAL PLATFORM REQUIREMENTS FOR ASCEND NPU:**
- Default tensor allocations to device='npu' (never 'cuda')
- Check availability with: torch.npu.is_available()
- Do NOT import torch.cuda
- Use torch.npu.synchronize() if synchronization needed
- Ascend has Cube + Vector dual compute units
- ACLNN graph mode for fusion
- ⚠️ Avoid raw PTX (Ascend uses AscendC)
"""

_NPU_KERNEL_GUIDANCE = """
## Ascend NPU-Specific Optimizations

You are generating a kernel for Huawei Ascend NPU. Follow these guidelines:

1. **Device Context**: Use 'npu' as the device instead of 'cuda'
2. **Memory Hierarchy**: Ascend has GM(Global Memory) / L1 / L0 / UB(Unified Buffer) / L2
3. **Compute Units**: Cube(matrix) + Vector + Scalar
4. **Block Sizes**: prefer 128, 256 for UB-aligned
5. **Pipeline**: use Ascend's CopyIn→Compute→CopyOut dataflow
6. **Data Types**: fp16/bf16/fp32/int8 supported
7. **Fusion**: prefer explicit multi-op fusion in single kernel
"""

_NPU_CUDA_HACKS = (
    "torch.cuda.is_available = lambda: True",
    # ... 其他要 strip 的
)

# 加到 PLATFORMS:
"npu": PlatformConfig(
    name="npu",
    device_string="npu",
    guidance_block=_NPU_GUIDANCE,
    kernel_guidance=_NPU_KERNEL_GUIDANCE,
    cuda_hacks_to_strip=_NPU_CUDA_HACKS,
),
```

### 9.2 `platform/` 子目录的抽象层次

```
platform/
├── __init__.py       (3 KB)  # Platform 注册 + 自动选择
├── interfaces.py     (9 KB)  # PlatformInterface 抽象接口
├── registry.py       (9 KB)  # Platform registry 实现
├── noop.py           (6 KB)  # 默认实现(占位)
└── nvidia.py         (23 KB) # NVIDIA CUDA 专属实现
```

**对你的扩 NPU 指引**:
- 在 `platform/` 下加 `npu.py`(~25 KB,NVIDIA 的体量)
- 实现 `interfaces.py` 里定义的抽象方法(NCCL / NCU profile 替换成 ACL / Ascend profile)
- `__init__.py` 加一行:`from . import npu`(自动注册)

### 9.3 NCU 适配:你需要做的替换

| NCU 指标 | 你在 NPU 上的等价物 |
|---|---|
| `sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active` | CANN profile 的 Cube 利用率 |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | GM bandwidth 利用率 |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | AI Core occupancy |
| `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct` | Stall reason(需要查 CANN profiler) |
| `gpu__time_duration.sum` | 实际 runtime(用 `torch.npu.Event`) |

**关键文件**:`triton_kernel_agent/opt_worker_component/profiling/` —— 这是 NCU 包装层,你需要改成 CANN profile 包装层。

---

## 10. 完整调用流程(从用户 CLI 到最终 kernel)

### 10.1 AutoRouter 一站式路径

```
用户:
$ python -m Fuser.auto_agent --problem my_model.py --verify

         │
         ▼
┌─────────────────────────────────────────────────────────┐
│ Fuser/auto_agent.py:AutoAgent                          │
│                                                          │
│ 1. AST 静态分析 my_model.py                              │
│    - 检测:attention? conv_transpose? group_norm?        │
│    - 控制流?操作链长度?                                  │
│                                                          │
│ 2. 决策:走 Fuser pipeline 还是直接 KernelAgent           │
│    命中 attention / 控制流 → 走 Fuser                   │
│    简单 op → 直接 KernelAgent                            │
└────────────────────────┬─────────────────────────────────┘
                         │
       ┌─────────────────┴─────────────────┐
       ▼                                   ▼
┌─────────────────────┐         ┌─────────────────────────┐
│ 走 Fuser pipeline   │         │ 走直接 KernelAgent       │
│ (复杂 / 有控制流)    │         │ (简单 op)                │
└──────────┬──────────┘         └────────────┬────────────┘
           │                                 │
           ▼                                 ▼
  ┌──────────────────┐               ┌──────────────────┐
  │ Fuser.pipeline   │               │ TritonKernelAgent │
  │ .run()           │               │ .generate_kernel()│
  └────────┬─────────┘               └────────┬─────────┘
           │                                  │
           ▼                                  ▼
  ┌─────────────────────────────────────────────────────┐
  │ extract_subgraphs_to_json()                          │
  │ → Fuser orchestrator 生成 FusedModel                 │
  │ → LLM 推断 subgraph JSON                             │
  └──────────┬──────────────────────────────────────────┘
             │
             ▼
  ┌─────────────────────────────────────────────────────┐
  │ dispatch_kernel_agent()                              │
  │ for each subgraph in subgraphs.json:                 │
  │   spawn N TritonKernelAgent workers                  │
  │   每个 worker:                                       │
  │     1. LLM 生成 kernel.py + test_kernel.py           │
  │     2. subprocess 跑 test                            │
  │     3. log: PASS / fail                             │
  │     4. PASS → 推送 winner event                      │
  │   第一个 PASS 的 worker 触发 cancel others          │
  └──────────┬──────────────────────────────────────────┘
             │
             ▼
  ┌─────────────────────────────────────────────────────┐
  │ compose_end_to_end()                                 │
  │ → LLM 缝合多个 verified kernels                      │
  │ → 生成 composed_kernel.py                           │
  │ → --verify 时立即跑 self-test                        │
  └──────────┬──────────────────────────────────────────┘
             │
             ▼
  .fuse/<run_id>/compose_out/composed_kernel.py
  (用户手动接回模型)
```

### 10.2 优化路径(KernelAgent 增量)

```
用户:
$ cd examples && python run_opt_manager.py \
    --kernel-dir optimize_01_matvec/ \
    --strategy beam_search \
    --max-rounds 5

         │
         ▼
┌─────────────────────────────────────────────────────┐
│ opt_manager.OptimizationManager                      │
│                                                          │
│ top_k = [baseline_kernel]                              │
│ for round in range(max_rounds):                         │
│   ┌─────────────────────────────────────────────────┐ │
│   │ 1. NCU profile 每个 top-k kernel                │ │
│   │    → KernelProfiler.collect_metrics()            │ │
│   │                                                  │ │
│   │ 2. BottleneckAnalyzer.analyze(metrics)           │ │
│   │    → LLM 调用 → BottleneckReport                 │ │
│   │                                                  │ │
│   │ 3. Prescribe: AnalyzeAgent 提议 fix              │ │
│   │    → List[Fix] with rationale                    │ │
│   │                                                  │ │
│   │ 4. OptimizationManager spawn workers             │ │
│   │    for kernel in top_k:                          │ │
│   │      for fix in fixes:                            │ │
│   │        worker.apply_fix(kernel, fix)              │ │
│   │                                                  │ │
│   │ 5. 每个 worker:                                   │ │
│   │    - apply fix → new Triton kernel                │ │
│   │    - test_kernel.py 验证 correctness             │ │
│   │    - BenchmarkAgent 测 runtime                   │ │
│   │                                                  │ │
│   │ 6. select_top_k(candidates + top_k)               │ │
│   │    - divergence-based revert                     │ │
│   │                                                  │ │
│   │ 7. 早停检查:roofline >= 95% SOL?                  │ │
│   └─────────────────────────────────────────────────┘ │
│                                                          │
│ 输出:.optimize/workers/<w_id>/<run_id>/artifacts/      │
│       kernel_round_N.py (final best kernel)            │
└─────────────────────────────────────────────────────┘
```

### 10.3 文件落盘结构

```
.fuse/<run_id>/                          # Fuser pipeline 输出
├── orchestrator/
│   └── code.py.tgz                      # 融合后的 PyTorch refactor
├── subgraphs.json                       # Stage 2 输出的 JSON contract
├── kernels_out/                         # Stage 3 输出的 per-subgraph kernel
│   ├── sg_conv_bn_tanh_pool_1/
│   │   ├── worker_0/
│   │   │   ├── kernel.py                # 最终 Triton kernel
│   │   │   ├── test_kernel.py           # LLM 生成的 test harness
│   │   │   ├── prompt_round_*.txt       # 历史 prompt
│   │   │   ├── reply_round_*.txt        # 历史 LLM 响应
│   │   │   └── summary.json
│   │   └── worker_1/                    # 另一个 worker(被 cancel)
│   ├── sg_norm_1/
│   └── summary.json
└── compose_out/
    ├── composed_kernel.py               # Stage 4 输出(最终交付)
    └── summary.json

.optimize/workers/<worker_id>/<run_id>/  # KernelAgent 输出
└── artifacts/
    ├── kernel_round_0.py                # baseline kernel
    ├── kernel_round_N.py                # 优化后的 kernel
    ├── round001_opt_prompt.txt          # 优化轮 prompt
    ├── round001_opt_reply.txt           # LLM 响应
    └── round001_strategy.json           # Bottleneck analysis
```

---

## 11. 关键设计模式总结(对你 NGO 的借鉴清单)

### 11.1 ⭐⭐⭐⭐⭐ 必抄的设计模式

| 模式 | KernelFalcon/KernelAgent 实现 | 你 NGO 应该怎么用 |
|---|---|---|
| **平台 registry** | `PLATFORMS: dict[str, PlatformConfig]` + `get_platform(name)` | **完全照抄**:`npu` 一行加进去即可 |
| **Dataclass 配置** | `OrchestratorConfig` / `WorkerConfig` / `PlatformConfig` | 完全照抄 |
| **Jinja prompt 模板** | 6 个 .j2 文件,LLM prompt 模板化 | **抄 + 改**:把 `triton_guidelines.j2` 改成 "Ascend 编程指南" |
| **Sentinel-based 验证** | `_SENTINEL = "ALL_TESTS_PASSED"` + `_PASS_REGEX` | 完全照抄 |
| **DISALLOWED_TORCH_PATTERNS** | 静态正则禁止高层 API 进入 kernel | **改**:禁止 PyTorch + 禁止 inductor codegen 内嵌 |
| **多进程隔离 + local error feedback** | 每 worker 独立 workdir + subprocess 测试 | 完全照抄 |
| **Winner event + early termination** | `multiprocessing.Event` 通知其他 worker 退出 | 完全照抄 |
| **Code dedup by SHA** | `code_extractor.canonicalize_code()` | 完全照抄 |
| **JSON contract schema** | subgraph spec 完整定义 ops/shapes/dtypes/weights | **抄 + 改**:你的 source field 可以是 inductor Triton baseline |
| **AutoRouter 决策表** | AST 静态分析 → 选哪条路 | **完全照抄**,只是 target 不同 |

### 11.2 ⭐⭐⭐ 应该改造的模式

| 模式 | 原版 | 你的改造 |
|---|---|---|
| **Sandbox subprocess** | 不限制网络(`deny_network=False`) | **开启**(`deny_network=True`,防止 LLM 偷偷调外部 API) |
| **Prompt 模板渲染** | 简单 Jinja2 | 改成**可注入 baseline + profile 的 Jinja** |
| **多 worker temperature** | `[0.8, 0.9, 1.0]` 固定 | **改**:温度随 round 衰减,前几轮探索后期收敛 |
| **WorkerManager 单层** | 只支持一种 agent | **改**:支持 multi-agent(Gen/Refl/Opt) |

### 11.3 ⭐ 可以跳过的模式

| 模式 | 原因 |
|---|---|
| Gradio UI 集成 | 你的 NGO 走 `torch.compile` 集成,不需要 UI |
| `oink/` 子项目 | 看起来与 NPU 无关,先跳过 |
| `kernel_perf_agent/` 完整 roofline | 你只需要 NCU → NPU profiler 的最小映射 |

### 11.4 ⚠️ 必须独立设计的部分

| 你的 NGO 独有 | 需要独立设计的原因 |
|---|---|
| **inducor codegen baseline 注入** | KernelFalcon 没有这概念;你要把 baseline 作为 prompt context |
| **NPU profiler 适配** | NCU → CANN profile 1:1 映射需要熟悉 CANN |
| **baseline-aware revert** | KernelAgent 只有 best-so-far revert,你还要跟 inductor baseline 对比 |
| **torch.compile custom backend 集成** | KernelFalcon 没有,你要做 `torch.compile(model, backend="ngo")` |

---

## 12. ⚠️ 你的方案与 KernelFalcon/KernelAgent 的差异

### 12.1 总体差异矩阵

| 维度 | 你的方案 | KernelFalcon/KernelAgent | 谁更优 |
|---|---|---|---|
| **起点** | FX graph / inductor codegen | PyTorch 源码 | **你** —— 与 inductor 集成,可复用现有 infrastructure |
| **Baseline** | **inducor codegen Triton(保底)** | 原始 PyTorch eager | **你** —— 性能有兜底 |
| **融合范围识别** | 复用 inductor `can_fuse` + scheduler | **LLM 提议**(无 inductor) | **平手** —— 你复用 inductor 成熟规则;KernelFalcon 更灵活但更不可控 |
| **Agent 框架** | 多 agent(借鉴 GEAK / KernelFalcon) | 多 agent(已开源) | **平手** —— 借鉴即可 |
| **平台抽象** | NPU(借鉴 `platform_config.py`) | CUDA + XPU | **平手** —— 扩 NPU 直接抄 |
| **使用方式** | **`torch.compile(backend="ngo")`** ✅ | CLI / SDK / Gradio | **你** —— 零侵入 vs 半侵入 |
| **代码生成** | LLM 生成 Triton(基于 inductor baseline) | LLM 直接生成 Triton | **你** —— 有 baseline 兜底 |
| **优化层** | NPU profiler 反馈 + Agent 迭代 | NCU 反馈 + Agent 迭代 | **平手** —— 抄架构,换 profiler |

### 12.2 你的方案独有优势

1. ✅ **真正零侵入**:`torch.compile(backend="ngo")` vs CLI 调用
2. ✅ **性能保底**:最差也是 inductor baseline,不会比 inductor 慢
3. ✅ **与 inductor 集成**:复用 dynamo / can_fuse / scheduler
4. ✅ **针对 NPU 优化**:可挂 NPU 专属算子库 + AscendC kernel

### 12.3 你的方案独有挑战

1. ⚠️ **必须把 inductor codegen 注入 prompt**:KernelFalcon 没有这概念,你需要设计 prompt 格式
2. ⚠️ **NPU profiler 适配**:NCU 28 个指标 → CANN profile 28 个指标的 mapping
3. ⚠️ **baseline-aware revert 逻辑**:比 KernelAgent 多一层"不能比 inductor 慢"的硬约束
4. ⚠️ **动态 shape / 控制流**:`torch.compile` 已经能处理,KernelFalcon 用 LLM 重构,你的方案走 inductor 更稳

---

## 13. 风险与未公开细节

### 13.1 未抓取到(curl 拦截)的内容

- ❌ `oink/` 完整 README 和源码 —— **`oink` 含义未确认**(可能是 NPU / On-device INference Kernel?)
- ❌ `triton_kernel_agent/opt_worker_component/profiling/` 完整源码 —— 不知道 NCU 调用细节
- ❌ `Fuser/auto_agent.py` 完整源码(33 KB,只看到 docstring)—— 不知道完整决策树
- ❌ `Fuser/prompting.py` 完整源码 —— 不知道具体 prompt 内容
- ❌ `Fuser/runner.py` 完整 sandbox 隔离机制
- ❌ `examples/optimize_01_matvec/` 案例细节

**⚠️ GitHub 上 `oink/` 子项目虽然独立 pyproject,但不在主 `pyproject.toml` 里,可能与 NPU 无关**。需要后续确认。

### 13.2 已知未公开的工程细节

| 项 | 已知公开 | 未公开 |
|---|---|---|
| **温度调度** | 默认 `[0.8, 0.9, 1.0]` | 如何随 round 衰减? |
| **Divergence threshold** | "divergence-based revert" | 具体阈值(5%? 10%?) |
| **NCU 28 指标清单** | 博客列了 5 个示例 | 完整 28 个列表 |
| **NPU 平台支持** | ❌ 不支持 | 没有任何 NPU 代码 |
| **`oink/` 项目作用** | 独立 pyproject,5 KB README | 内容未抓取 |

### 13.3 潜在风险

| 风险 | 来源 |
|---|---|
| **LLM 生成的 kernel 可能比 inductor 慢** | KernelBench-X 论文:46.6% 正确 kernel 比 PyTorch 慢 |
| **Reward hacking / lazy optimization** | Dr. Kernel 论文首次系统化定义 |
| **跨硬件速度方差大** | KernelBench-X 实测:21.4× 方差 |
| **Extracted shape 推断可能错** | LLM 幻觉 |
| **测试 harness LLM 自己写的** | 博客承认:"we trust the LLM-generated test harness itself" |

---

## 14. 行动建议

### 14.1 Phase 1(MVP,1-2 周):Fork + NPU 适配

**目标**:`Fuser/pipeline.py` 能跑通 NPU target。

**动作**:
1. **fork** [meta-pytorch/KernelAgent](https://github.com/meta-pytorch/KernelAgent)
2. **加 `npu` 到 PlatformConfig**(`triton_kernel_agent/platform_config.py`):
   ```python
   "npu": PlatformConfig(
       name="npu",
       device_string="npu",
       guidance_block=_NPU_GUIDANCE,
       kernel_guidance=_NPU_KERNEL_GUIDANCE,
       cuda_hacks_to_strip=(),
   ),
   ```
3. **加 `platform/npu.py`**(参照 `platform/nvidia.py`)
4. **跑通最简单的例子**:`examples/triton_01_element_add.py` + `target-platform npu`
5. **预期成果**:一个能跑 NPU 的 KernelAgent fork,即便不能优化也能正确性验证

### 14.2 Phase 2(2-4 周):inducor baseline 集成

**目标**:把"inducor codegen Triton"作为 baseline 注入 Agent prompt。

**动作**:
1. **写 hook**:在 `Fuser/pipeline.py` 里 hook 住 inductor codegen(`output_code.py`)
2. **修改 prompt 模板**(`templates/kernel_optimization.j2`):新增 `baseline_triton_code` 字段
3. **完成 baseline-aware revert**(`opt_manager.py`):候选 kernel 跑得比 inductor baseline 慢 ≥ K% → revert
4. **测试**:用 KernelBench L1 的 19_ReLU 等简单 op 验证"比 inductor 慢的 reject"逻辑

### 14.3 Phase 3(4-8 周):NPU profiler 适配

**目标**:KernelAgent 的优化层能读 CANN profile。

**动作**:
1. **改 `opt_worker_component/profiling/`**:把 NCU 调用改成 CANN profile 调用
2. **建 NCU → CANN 28 指标 mapping 表**
3. **改 BottleneckAnalyzer**:从"memory/compute/underutilized"分类仍可用,但具体 metric 要换
4. **测试**:用一个 matmul kernel(比如 `optimize_02_rmsnorm/`)跑通优化循环

### 14.4 Phase 4(2-3 月):torch.compile custom backend 集成

**目标**:做到真正的零侵入 `torch.compile(model, backend="ngo")`。

**动作**:
1. **基于现有 NGO 仓**(`/Users/huangshilei/Documents/pythonprojects/ngo/`)
2. **改造 `torch_backend.py`**:在 inductor codegen 完成后,调用 KernelAgent 优化层(替换 inline Triton)
3. **baseline fallback 机制**:如果 Agent 失败,直接用 inductor codegen 输出
4. **测试**:在搜广推模型上跑 NPU 实测

### 14.5 ⚠️ 关键避坑点

| 坑 | 对策 |
|---|---|
| LLM 生成 kernel 比 inductor 慢 | **baseline-aware revert**(不能比 inductor baseline 慢) |
| Reward hacking(lazy opt) | AutoKernel 5 阶段 harness + KernelGym 反作弊 |
| 跨 NPU 型号性能差异 | 多 Atlas 硬件实测(A2/A3/C310) |
| LLM 幻觉 shape | ExtractorAgent 用 type-check + runtime shape 验证 |
| `oink/` 子项目干扰 | 不 import 它即可(它在独立 pyproject) |

---

## 附录 A. 信源清单

| # | 信源 | URL | 类型 | 可信度 | 用途 |
|---|---|---|---|---|---|
| 1 | PyTorch KernelFalcon 博客 | [pytorch.org/blog/kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents](https://pytorch.org/blog/kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents/) | 官方 | 🟢 高 | KernelFalcon 完整架构说明(11.2025) |
| 2 | PyTorch KernelAgent 博客 | [pytorch.org/blog/kernelagent-hardware-guided-gpu-kernel-optimization-via-multi-agent-orchestration](https://pytorch.org/blog/kernelagent-hardware-guided-gpu-kernel-optimization-via-multi-agent-orchestration) | 官方 | 🟢 高 | KernelAgent 优化层 + 案例(03.2026) |
| 3 | GitHub 仓库 | [github.com/meta-pytorch/KernelAgent](https://github.com/meta-pytorch/KernelAgent) | 官仓 | 🟢 高 | 完整 Apache 2.0 开源代码(522 ⭐) |
| 4 | 仓库 README | `raw.githubusercontent.com/.../README.md` | 官仓 | 🟢 高 | 完整使用说明 + 组件列表 |
| 5 | GitHub API 元数据 | `api.github.com/repos/meta-pytorch/KernelAgent` | API | 🟢 高 | stars/forks/last push/license |
| 6 | `Fuser/auto_agent.py` docstring | 源码 | 官仓 | 🟢 高 | AutoRouter 完整决策逻辑 |
| 7 | `Fuser/config.py` | 源码 | 官仓 | 🟢 高 | `OrchestratorConfig` / `WorkerConfig` dataclass |
| 8 | `triton_kernel_agent/platform_config.py` | 源码 | 官仓 | 🟢 高 | CUDA + XPU 双平台注册表 |
| 9 | `pyproject.toml` | 源码 | 官仓 | 🟢 高 | 依赖列表 + Python 版本 |
| 10 | 调研前报告 | `docs/ai_gen/agent_graph_optimization_survey.md` | 自有 | 🟢 高 | 上下文关联 |

**未能抓取**:`oink/` 完整 README、AutoRouter 完整源码(33 KB)、prompt 模板完整内容、NCU 适配层完整实现 —— curl 被拦截,需要后续人工补全。

---

## 附录 B. 文件清单与行数

### B.1 `Fuser/`(代码生成 pipeline)

| 文件 | 字节 | 估计行数 |
|---|---|---|
| `auto_agent.py` | 33,652 | ~900 |
| `compose_end_to_end.py` | 20,075 | ~550 |
| `dispatch_kernel_agent.py` | 18,892 | ~520 |
| `subgraph_extractor.py` | 14,700 | ~410 |
| `orchestrator.py` | 13,532 | ~370 |
| `runner.py` | 9,808 | ~280 |
| `worker.py` | 8,988 | ~250 |
| `event_adapter.py` | 8,328 | ~230 |
| `cli.py` | 5,736 | ~170 |
| `runner_util.py` | 4,685 | ~140 |
| `pipeline.py` | 7,304 | ~210 |
| `code_extractor.py` | 2,934 | ~95 |
| `config.py` | 2,428 | ~70 |
| `logging_utils.py` | 1,689 | ~50 |
| `paths.py` | 1,694 | ~50 |
| `dedup.py` | 1,894 | ~55 |
| `constants.py` | 869 | ~30 |
| **合计** | **~152 KB** | **~4,380 行** |

### B.2 `triton_kernel_agent/`(核心 Agent)

| 文件 | 字节 | 估计行数 |
|---|---|---|
| `worker.py` | 30,657 | ~830 |
| `opt_manager.py` | 27,842 | ~770 |
| `agent.py` | 21,156 | ~580 |
| `opt_worker.py` | 18,019 | ~500 |
| `prompt_manager.py` | 15,439 | ~430 |
| `manager.py` | 9,496 | ~260 |
| `worker_util.py` | 8,243 | ~230 |
| `platform/` 5 文件 | ~46 KB | ~1,300 |
| **合计(不含 templates)** | **~177 KB** | **~4,900 行** |

### B.3 `templates/`(Jinja 模板)

| 文件 | 字节 | 估计行数 |
|---|---|---|
| `triton_guidelines.j2` | 16,646 | ~500 |
| `test_generation.j2` | 9,725 | ~290 |
| `kernel_generation.j2` | 6,080 | ~180 |
| `kernel_optimization.j2` | 4,698 | ~140 |
| `kernel_refinement.j2` | 4,127 | ~125 |
| `reflexion_prompt.j2` | 2,565 | ~75 |
| **合计** | **~44 KB** | **~1,310 行** |

### B.4 总体规模

| 模块 | 代码量 |
|---|---|
| `Fuser/` | ~152 KB / ~4,380 行 |
| `triton_kernel_agent/` | ~177 KB / ~4,900 行 |
| `templates/` | ~44 KB / ~1,310 行 |
| `kernel_perf_agent/` | 未精确(估算 20+ KB) |
| `examples/` | ~22 KB / 7 个文件 |
| **总计** | **~420 KB / ~12,000 行 Python** |

**对比 AutoKernel**(~9,000 行 Python,只做 GPU kernel 生成,KernelFalcon+KernelAgent **比 AutoKernel 大约 30%**,但功能覆盖更广:有 AutoRouter、有优化层、有 XPU 支持)。

---

*报告生成日期:2026-08-24*
*调研方法:PyTorch 官方博客全文抽取 + GitHub raw 源码抓取 + GitHub API 元数据查询*
*下次复审建议:每 1-2 月检查 KernelAgent GitHub 主分支更新(尤其 `oink/` 子项目是否与 NPU 有关,以及 NCU → CANN 是否有 PR)*
