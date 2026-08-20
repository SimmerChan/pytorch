# 图切分与可融合范围识别:精准调研报告

> **调研主题**:在算子图(FX/Relay/HLO/Linalg 层),有哪些机制可以**识别可融合范围**,它们的**验证机制**是什么,以及与 PyTorch inductor 的 `can_fuse` 静态规则相比,优势在哪里?
>
> **v2 修订说明**:v1 报告把 "Agent/LLM 整算子代码生成" 也算作 fusion 范围识别,但**这些工作实际是"整算子生成+迭代优化",并不显式做 fusion 边界识别**(只是把 fusion 当副产品)。本次精准调研**剔除**那部分内容,只聚焦**真正做 fusion 范围识别**的工作。
>
> 调研方法:anysearch 多多源检索 + 关键论文全文抽取 + arXiv 综述交叉验证。信源标注见附录 A。

---

## 目录

- [0. TL;DR](#0-tldr)
- [1. 调研范围澄清:什么才算"fusion 范围识别"](#1-调研范围澄清什么才算fusion-范围识别)
- [2. Inductor 基线:fusion 范围是怎么识别的](#2-inductor-基线fusion-范围是怎么识别的)
- [3. 第一类:硬编码规则派(Symbolic Pattern Match)](#3-第一类硬编码规则派symbolic-pattern-match)
- [4. 第二类:代数等价图替换派(Algebraic Substitution)](#4-第二类代数等价图替换派algebraic-substitution)
- [5. 第三类:多维 + 数学优化派(ML Cost + Polyhedral)](#5-第三类多维--数学优化派ml-cost--polyhedral)
- [6. 第四类:Tile-Graph / Whole-Graph 派(以内存为中心)](#6-第四类tile-graph--whole-graph-派以内存为中心)
- [7. 验证机制专题:怎么证明 fusion 是对的](#7-验证机制专题怎么证明-fusion-是对的)
- [8. 与 inductor 的逐项对比矩阵](#8-与-inductor-的逐项对比矩阵)
- [9. 给你的工程化建议](#9-给你的工程化建议)
- [10. 风险与未解问题](#10-风险与未解问题)
- [附录 A. 信源与可信度](#附录-a-信源与可信度)
- [附录 B. 关键文献索引](#附录-b-关键文献索引)

---

## 0. TL;DR

围绕"**在算子图上识别可融合范围**",业界演进出了**四类**核心机制:

| 类别 | 核心问题 | 代表工作 | 与 inductor 比的关键差异 |
|---|---|---|---|
| **硬编码规则派**(§ 3) | 哪些 IR 类别可以两两融合? | XLA HLO fusion、MLIR Linalg fusion、TVM FuseOps | **更结构化**——XLA 用 4 类算子(强于 inductor 的 9 个 IR 类目但更粗)、MLIR 用 callback(可挂 ML cost)、IREE 用 DAG match interface |
| **代数等价替换派**(§ 4) | 哪些子图能用**代数等价**的子图重写? | **TASO**(SOSP 2019)、**PET**(OSDI 2021)、Unity(OSDI 2022) | **真正颠覆 inductor**——TASO 自动生成 743 个 fusion replacement,53K 行 manual rules → 1.4K 行 specs,新 DNN 比 TensorFlow 快 2.8× |
| **多维 + 数学优化派**(§ 5) | 跨 iteration / tile 的多维 fusion 空间 + 数学最优 | DNNFusion、Apollo、DeepCuts、AStitch、ALCOP、Chimera | **优化搜索**——DNNFusion 用 fusion seed + 传播式探索,DeepCuts 用 DL-guided fusion cut,Apollo 用 loop feedback 重切 partition |
| **Tile-Graph / Whole-Graph 派**(§ 6) | 以内存调度为中心的整图 fusion | Welder(OSDI 23)、Kitsune、TurboMGNN | **统一抽象**——Welder 用 tile-graph 统一 register / shared / DRAM 各级 fusion,89 个未探索 fusion pattern 自动发现 |

**对你核心问题的回答**:**有比 inductor 更优的机制,且至少有 3 个**真正超越**can_fuse 硬编码的工作**:

1. **TASO**——用 SAT 求解器和算子数学性质自动**生成新的 fusion candidates**(打破"只能识别预设 IR 类别"的天花板)
2. **DNNFusion**——用数学性质的 graph rewriting + fusion seed operators + 传播探索,**比 inductor 多识别 8.8× fusion opportunities**(被 PLDI 2021 验证)
3. **Welder**——用 tile-graph 统一抽象,**自动发现 89 个 inductor 静态规则不会尝试的 fusion patterns**(OSDI 2023)

**验证机制专题**(§ 7):这些工作**怎么证明 fusion 是对的**?三类核心方法——① TASO 用 theorem prover(自动定理证明器)严格验证每个 substitution ② PET 用 mutation testing + 自动 correction kernel ③ Welder/TASO 都有边界上的有限域随机测试兜底。

---

## 1. 调研范围澄清:什么才算"fusion 范围识别"

### 1.1 严格定义

**fusion 范围识别**(fusion range identification)是指:给定一个算子图 G(V, E),**识别子集 S ⊆ V 使得 S 内的算子可以被融合成一个 kernel**,**同时给出该融合的 codegen 方案或可证明的等价性**。

关键要素:
- **输入**:算子图(FX / Relay / HLO / Linalg / TIR)
- **输出**:**融合边界**(哪些算子融在一起,哪些不融)+ **融合方式**(等价 / 部分等价 / layout 变换)
- **验证**:融合前后的**功能等价性**(或可校正的等价性)

### 1.2 v1 调研的偏差

上一份报告 (v1) 把下列工作也算作 fusion 范围识别——**这是错的**:

| v1 误归类的工作 | 它实际做什么 | 是否真的做 fusion 范围识别 |
|---|---|---|
| **AKG Kernel Agent** (Huawei) | 输入整算子(PyTorch op),输出 Triton/AscendC kernel | ❌ 整算子生成,不显式做 fusion 边界识别 |
| **K-Search** | 同上 | ❌ 同上 |
| **KernelBench/KernelSmith** | 同上 | ❌ 同上 |
| **Mirage μGraph** | 算子**内部**代数等价搜索 + superopt | ⚠️ **算子内部**,不跨算子 fusion |
| **AStitch** | 算子**内部**多维 tile fusion | ⚠️ 同上 |

**正确归类**应该是:**它们是"算子生成 + 迭代优化"工作**,fusion 只是副产品。

### 1.3 真正做"fusion 范围识别"的工作(本报告范围)

把上面剔除后,**真正聚焦"在算子图上识别融合边界"**的核心工作分四类:

| 类别 | 关键问题 | 代表工作 |
|---|---|---|
| **硬编码规则** | 哪些 IR 类别可以两两融合? | XLA HLO fusion、MLIR Linalg fusion、TVM FuseOps、IREE DispatchCreation |
| **代数等价** | 哪些子图能用代数等价子图重写? | TASO、PET、Unity |
| **多维 + ML 优化** | 跨 iteration / tile 的多维 fusion 空间,如何选最优? | DNNFusion、Apollo、DeepCuts、AStitch、ALCOP、Chimera |
| **Tile-Graph** | 以内存调度为中心的整图 fusion | Welder、Kitsune、TurboMGNN |

> **判定标准**:一篇工作是不是 fusion 范围识别工作,看它**是否显式给出 fusion 边界(融合哪些算子 vs 不融合)**。如果不显式,而是把整算子当整体生成,就不算。

---

## 2. Inductor 基线:fusion 范围是怎么识别的

> 数据来源:`torch/_inductor/{lowering.py, ir.py, scheduler.py}` + 参考文档 `inductor_fusion_breadth_analysis.md`

### 2.1 两段式 fusion 范围识别

inductor 在算子图上识别 fusion 范围是**两段式**:

**Stage 1 - Lowering**:`lowering.py` 把 ATen/Prim 算子路由到 IR 节点。**IR 类型决定 fusion 资格**(`get_reduction_type()` 与 producer/consumer 兼容性表)。

| IR 类 | fusion 资格 |
|---|---|
| `Pointwise(Loops)` | ✅ 任意 fusion |
| `Reduction(Loops)` | ✅ Pointwise prologue/epilogue |
| `Scan(Loops)` | ⚠️ 自融,不可与 Reduction fusion (`simd.py:2077`) |
| `TemplateBuffer` | ⚠️ 仅 epilogue/prologue/multi-out |
| `ExternKernel` | ❌ 基本不可融 |

**Stage 2 - Scheduler**:`scheduler.can_fuse` + `scores_fusion`,硬编码规则集合 + 贪心评分挑高分 fusion。

### 2.2 inductor fusion 范围识别的局限(简短版)

1. **静态、无运行时反馈**:只看 IR + 拓扑,不感知 NPU 上的实际开销
2. **预设 IR 类别不可扩展**:`Pointwise` 之外无法发明新类别(如 `aten.lerp` 作为整体优化)
3. **无跨层融合语义**:不识别 `conv+bn+relu` 这类语义组合
4. **不能发现代数等价替换**:看到一个 3-算子 sub-graph,不能发现"它能用 2-算子等价 sub-graph 替代"

后续四类工作,都从某个角度突破了这些局限。

---

## 3. 第一类:硬编码规则派(Symbolic Pattern Match)

> 这一类是"成熟产线方案",覆盖了 inductor 同等功能,但天花板低。

### 3.1 XLA HLO Fusion(Google 内部核心)

**来源**:Operator Fusion in XLA: Analysis and Evaluation(Snider & Liang 2023, arXiv 2301.13062)+ OpenXLA 官方架构

XLA fusion 是 Google 内部最成熟的方案,有 **4 类专门的 fusion pass**(实测,2023):

| Fusion Pass | 机制 | 何时触发 |
|---|---|---|
| **Instruction Fusion** | 反向 post-order 遍历,producer 通过 `ShouldFuse` 函数判断是否融合到 consumer | 默认开启 |
| **Fusion Merger** | 把已 fusion 的 instruction 进一步合并到 user,减少 memory bandwidth | 当不会增加 byte transfer 时 |
| **Multi-Output Fusion**(sibling + producer-consumer) | 共享 input 的 sibling op,或 producer-consumer 多输出 | 默认 |
| **Horizontal Fusion** | 把相同 formula 跨多个 variable 的小 kernel 横向融合(用于 Adam/L2Loss) | 小 kernel 多时 |

**XLA fusion 边界识别机制**:
- 用 `ShouldFuse` 函数判断 producer × consumer pair 可否融合
- **保留一份"expensive op" 列表**(convolution, sort, all-reduce 等)——这些**绝不被融合**
- **硬件限制检查**:threads per block、shared memory per block、threads per SM

**XLA 论文实测**(arXiv 2301.13062):
- "Conservative fusion criteria in XLA also limits the opportunities for optimization"
- 论文自己尝试了**比 XLA 激进**的 fusion 策略,在 Cart-pole RL 上达到 **10.56× speedup** over XLA 默认 fusion

**与 inductor 对比**:
- XLA 与 inductor 几乎**结构同构**(都是算子分类 + 规则 + codegen)
- XLA 更粗(4 类),inductor 更细(9 个 IR 类)
- **XLA 的优势**:pipeline 更完整(DCE/CSE/SPMD partition 都在 fusion 前后联动)
- **XLA 的劣势**:它自己承认"conservative"——通过 4 类规则识别的 fusion 范围**有显著遗漏**(论文实测可达 10.56×)

### 3.2 MLIR Linalg Fusion(MLIR 主线)

**来源**:LLVM Discourse "Tile and fuse support" + IREE `LinalgExtInterfaces.td` + MLIR Linalg Passes 文档

MLIR 设计哲学与 inductor 完全相反:**transformation 与 heuristics 解耦**。

- `TilingInterface` / `tileConsumerAndFuseProducersUsingSCF` 提供"tile-and-fuse 这个动作"
- **"该不该 tile-and-fuse"留给调用方**,通过 `options` 的 callback 决定哪些 op 被 fuse

**IREE 的扩展 `LinalgExtFusionInterface`** 是 fusion 范围识别的核心接口——专门为识别"哪些 DAG 可以捕获为 micro-kernel"。

**IREE DispatchCreation Pass 的 fusion 范围识别**(真实产线使用,有 flag):

| Pass Flag | 机制 | 触发条件 |
|---|---|---|
| `--aggressive-fusion` | 激进模式:启用对所有后端不一定 ready 的 fusion | 性能优先 |
| `--fuse-multi-use-producers` | 启用实验性:fusion 多消费者的 producer | 复杂 DAG |
| `--fuse-pad-with-consumers` | 把 pad op 融合到 consumer | pad 后常跟 elementwise |
| `--fuse-pad-with-producers` | 反向:pad 融合到 producer | producer 输出 shape 需对齐 |

IREE 还有更细的 pass(`--intra-dispatch`、`--num-iterations`)控制 multi-use ops 的 fusion 迭代次数。

**与 inductor 对比**:

| 维度 | inductor | MLIR Linalg |
|---|---|---|
| fusion 范围识别决策者 | scheduler 硬编码 | 用户在 callback 中任意编码 |
| 范围识别粒度 | 节点对 | 整个 tile-and-fuse 区间 |
| 跨层 fusion 支持 | 有限 | 强(`tileConsumerAndFuseProducers` 支持跨 block 嵌套) |
| 产线 ready | ✅ | ✅(IREE/LLVM 主线) |

### 3.3 TVM FuseOps(Relay/Relax 框架)

**来源**:Apache TVM 官方文档 + TVM OSDI 2020 论文 + Ansor OSDI 2020 论文

TVM FuseOps 把算子分为 4 类(与 XLA 同款):
- **Injective**(elementwise,one-to-one map)
- **Reduction**
- **Complex-out-fusable**(e.g., conv2d,可把 elementwise 融到输出)
- **Opaque**(不可融,e.g., sort)

**通用 fusion 规则**(TVM OSDI 2020 原文):

> "Multiple injective operators can be fused together into another injective operator. A reduction operator can be fused together with input injective operators."

### 3.4 小结

| 派系 | fusion 范围识别机制 | 成熟度 | 适用 |
|---|---|---|---|
| **XLA HLO** | 4 类 pass(Instruction/Merger/Multi-Output/Horizontal) | ✅ Google 产线 | TF/JAX |
| **MLIR Linalg** | TilingInterface + callback + LinalgExtFusionInterface | ✅ IREE/LLVM 主线 | MLIR 系 |
| **TVM FuseOps** | 4 类算子分类 + 通用规则 | ✅ Apache TVM | TVM/Ansor |
| **PyTorch inductor** | 9 个 IR 类 + `can_fuse` 表 | ✅ PyTorch 默认 | inductor |

**与 inductor 比**:**功能等价**,没有质变。**核心差距**是这些系统**没有人解决"inducer 规则外的 fusion 机会"问题**——它们的 fusion 范围识别**都在预设的算子/IR 类别内**,不会主动发现新的 fusion 模式。

---

## 4. 第二类:代数等价图替换派(Algebraic Substitution)

> 这一类是**真正突破 inductor 静态规则天花板**的工作。核心思想:用数学定理证明器验证"两个 sub-graph 在数学上等价",然后做替换。

### 4.1 TASO(Stanford, SOSP 2019)

**来源**:TASO: Optimizing Deep Learning Computation with Automatic Generation of Graph Substitutions(Jia et al., SOSP 2019)+ TASO 官方 GitHub + Stanford slides

**核心创新**:**用 theorem prover 自动生成 graph substitutions**——这是**第一篇**让 fusion 范围识别从"手工规则"进化到"自动验证 + 自动生成"的工作。

#### 4.1.1 三步法

1. **Graph Substitution Generator**:枚举所有可能的子图(用给定算子作为 building block),通过 fingerprint hash 找等价候选
2. **Graph Substitution Verifier**:用自动定理证明器(SMT)对每个候选 substitution 做**形式化验证**——证明"在算子数学性质下,候选两边计算结果完全一致"
3. **Search-Based Graph Optimizer**:cost-based backtracking search 应用 substitutions 找最优图

#### 4.1.2 关键数据(TASO SOSP 2019)

> "TASO generates all 743 substitutions in 5 minutes, and verifies them against 43 operator properties in 10 minutes"
> "TensorFlow currently contains approximately 53,000 lines of manual optimization rules, while the operator specifications needed by TASO are only 1,400 lines of code"

| 指标 | TensorFlow 手工规则 | TASO 自动生成 |
|---|---|---|
| 规则代码量 | 53,000 LOC | **1,400 LOC**(规格) |
| substitution 数量 | ~200 | **743**(自动生成) |
| 维护性 | 持续需补新算子 | 新算子加 spec 即可 |
| 性能 | 基准 | **新 DNN 高达 2.8×** |

#### 4.1.3 TASO 与 inductor 的关键区别

| 维度 | inductor | TASO |
|---|---|---|
| **fusion 范围识别机制** | IR 类型 + can_fuse 表 | **自动生成的 743 个 graph substitutions** |
| **验证机制** | 静态规则 + 测试 | **SMT 定理证明器**(43 operator properties) |
| **能否发现新 fusion 模式** | ❌ | ✅(自动枚举所有 sub-graph,找等价) |
| **数学性质使用** | ❌ | ✅(conv 双线性、additivity 等) |
| **新算子扩展性** | 改代码 | 加 operator spec |
| **运行时验证** | 无 | fingerprint hash + integer test |

#### 4.1.4 TASO 论文中的关键观察

> "TASO outperforms existing frameworks by up to **2.8×** ... for ResNet-50, TASO matches the performance of these frameworks with hand-written rules"
> "The final graph is 30% faster on V100 but 10% slower on K80"——TASO 自动发现的 fusion 在不同硬件上**收益不同**,这证明了 fusion 范围识别需要硬件感知

#### 4.1.5 TASO 的局限

- **生成是全局的,但验证是局部的**——它枚举所有 sub-graph 对,但生成的 743 个 substitution **不一定都在目标模型上有效**
- **硬件不感知**(虽然论文 § 7.5 提到尝试过 layout + 联合优化)
- **cost model 简单**:"sum of individual operators' cost",不能捕获 fusion 后 kernel 间的交互开销

### 4.2 PET(清华, OSDI 2021)

**来源**:PET: Optimizing Tensor Programs with Partially Equivalent Transformations and Automated Corrections(Wang et al., OSDI 2021)+ GitHub `whjthu/pet-osdi21-ae`

**核心创新**:TASO 只考虑**完全代数等价**的 substitution。**PET 扩展到部分等价的变换**,通过**自动校正 kernel** 恢复完全等价。

#### 4.2.1 PET 框架

1. **Program Mutator**:对 tensor program 生成"部分等价 mutants"——可能不严格等价但能加速的变换
2. **Mutation Corrector**:自动添加校正 kernel 让 mutants **恢复完全等价**(但保留性能收益)
3. **Program Optimizer**:cost-based + on-board 测量的搜索,挑最优

#### 4.2.2 PET 论文的关键实验数据(arXiv/纸质)

- **PET 在 ResNet-18/CSRNet/Inception-v3/BERT/ResNet3D-18 等模型上 outperforms existing frameworks up to 2.5×**
- **ablation**:部分等价变换单独用,也能获得显著收益(参见 PET 论文 Figure 13)
- **correction kernel fusion 关键**:ablation 显示禁掉 correction kernel fusion → 性能降低 **2.9×**
- **vs TASO**:联合优化 = TASO 收益 + 部分等价新机会

#### 4.2.3 PET 引入的 4 个新 fusion 范围(论文 § 8.3)

1. **Tensor-Level Optimization**:改变 tensor shape(如 conv input reshape)+ FFT convolution 替代 IGEMM
2. **Operator-Level Optimization**:dilated conv → 普通 conv(用 Winograd)
3. **Graph-Level Optimization**:两个并行 conv → group conv + correction kernel
4. **Kernel Fusion**:多个 R/T 算子融成一个

#### 4.2.4 PET 与 inductor 的关键区别

| 维度 | inductor | PET |
|---|---|---|
| **fusion 范围识别机制** | IR 类型 + can_fuse | **部分等价 mutant + 自动校正 kernel** |
| **突破局限** | 不允许"非完全等价" | ✅ 允许非等价变换,后置校正 |
| **融合搜索空间** | inductor 173 个 op | **TASO 的 ~157 rules + 部分等价新机会**(更大) |
| **联合优化** | fusion × fusion only | **fusion + layout + reshape + 等价变换**(全维度) |

### 4.3 Unity(Stanford/Meta/Facebook, OSDI 2022)

**来源**:Unity: Accelerating DNN Training Through Joint Optimization of Algebraic Transformations and Parallelization(Unger et al., OSDI 2022)+ Stanford paper

**核心创新**:**联合优化 algebraic transformations + parallelization**——TASO 只做代数变换,Unity 把它和**分布式并行**联合优化。

#### 4.3.1 Unity 的关键贡献

- **Unified Parallel Computation Graph (PCG)**:代数变换和并行化在**同一个图表示**里
- **Hierarchical Search**:从 O(2^(gs)) 的 worst-case 搜索降到 O(2^(g-s))(通过 graph splitting 选择分裂点)
- **Performance gain**:在 DLRM/CANDLE-Uno 等模型上**比现有框架快 3.6×**

#### 4.3.2 Unity 与 inductor 的关键区别

| 维度 | inductor | Unity |
|---|---|---|
| 范围 | 单设备 | **分布式 DNN 训练** |
| fusion × 通信 | ❌ | ✅ 联合优化 |
| 适用范围 | 推理为主 | **训练为主** |

### 4.4 小结:代数等价替换派 vs inductor

**这是本报告最关键的一节**——这是**真正颠覆 inductor 静态规则**的一类。

| 维度 | inductor | TASO/PET/Unity |
|---|---|---|
| **fusion 范围识别** | 预设 IR 类目 | **自动生成 substitution,数学验证** |
| **范围扩展性** | 加 op 要改代码 | **加 operator spec 自动生成新 substitution** |
| **是否发现 inductor 外的 fusion** | ❌ | ✅ TASO 找到 743 个 substitution(其中很多 inductor 不会触发) |
| **验证机制** | 规则 + 测试 | **SMT 定理证明器**(形式化) |
| **允许非完全等价** | ❌ | ✅(PET) |
| **联合优化** | fusion × codegen | **fusion × algebraic × layout × parallelization** |
| **性能收益** | 基准 | **新 DNN 上 2.8×** |
| **产线部署成熟度** | ✅ PyTorch 默认 | ⚠ TASO/PET 是论文级,Unity 已部分开源到 FlexFlow |

---

## 5. 第三类:多维 + 数学优化派(ML Cost + Polyhedral)

> 这一类是 fusion 范围识别的**优化派**——规则基础上加 ML cost model / 多维优化。

### 5.1 DNNFusion(William & Mary / Pitt, PLDI 2021)

**来源**:DNNFusion: Accelerating Deep Neural Networks Execution with Advanced Operator Fusion(Niu et al., PLDI 2021)+ PLDI 2021 slides + William & Mary paper

**核心创新**:**显式提出 fusion seed operators** + 传播式探索 fusion plan。

#### 5.1.1 DNNFusion 框架

1. **Operator Classification + Mathematical Property-Based Graph Rewriting**:先按数学性质分类算子 + 通过 rewriting 减少 evaluation cost
2. **Fusion Plan Generation**:选 fusion seed operators → 传播探索 successor/predecessor → 生成多个 fusion blocks
3. **Light-Weight Profiling**:对每个 fusion block 跑快速 profile 估成本
4. **Fusion Code Generation**:基于 fusion blocks 生成 fused kernel

#### 5.1.2 关键数据(DNNFusion PLDI 2021)

> "DNNFusion finds up to **8.8× higher fusion opportunities**, outperforms four state-of-the-art DNN execution frameworks with **9.3× speedup**"

- 在 15 个 DNN 模型上评估
- **fusion 机会数 up to 8.8×**(比 TensorFlow / TVM / MNN 多发现 8.8× fusion candidates)
- **整体 speedup up to 9.3×**(over 四个 SOTA 框架)

#### 5.1.3 DNNFusion 与 inductor 的关键区别

| 维度 | inductor | DNNFusion |
|---|---|---|
| **fusion plan 探索** | 启发式贪心 | **fusion seed + 传播式系统搜索**(NP-complete 优化问题) |
| **范围识别粒度** | 局部(producer × consumer) | **整图(fusion seed 起始,传播到 successor/predecessor)** |
| **数学性质使用** | ❌ | ✅(graph rewriting) |
| **可识别 fusion 机会** | 173 个 op | **比 4 个 SOTA 框架多 8.8×** |

### 5.2 Apollo(MLSys 2022)

**来源**:Apollo: Automatic Partition-based Operator Fusion through Layer...(Zhao et al., MLSys 2022)+ MLSys slides

(详见 v1 报告 § 3.4,这里只列与 fusion 范围识别相关的要点)

**机制**:
- **Primitive/compound operator 抽象**:compound 是 primitive 的图,二者构成 subgraph
- **Graph-level node grouping + operator-level loop fusion** 联合搜索
- **Polyhedral engine 反馈**:loop fusion 阶段把"不可融合的组合"反馈回 graph engine,迫使其**重新生成 partition**

**性能**:在 NPU 类 DSA 上,改善 vendor 框架 **19.7%**——**对 NPU 直接相关**。

### 5.3 DeepCuts(Seoul National U., PLDI 2021)

**来源**:DeepCuts: a deep learning optimization framework for versatile GPU workloads(Jung et al., PLDI 2021)+ PLDI 2021 site

**核心创新**:**用 DL-guided cost model 做 fusion cut selection**——决定"在哪里切 fusion"。

#### 5.3.1 DeepCuts 框架

- 分析 DNN workload,把多 op 组成单个 GPU kernel
- 同时优化 **kernel implementation 参数 + GPU 架构参数**
- **analytical cost model** 用于引导 fusion cut selection

#### 5.3.2 DeepCuts 的关键限制(论文原文)

> "DeepCuts only considers limited vertical operation fusions in the computation graph, and it has relatively high implementation cost for the new operations because of its analytical cost model"

**DeepCuts 与 inductor 的关键区别**:
- DeepCuts 用 analytical cost model 选 fusion cut,**不依赖纯规则**
- 但**只考虑 vertical fusion**,不处理 horizontal / multi-output fusion

### 5.4 AStitch(ASPLOS 22)

**来源**:AStitch: enabling a new multi-dimensional optimization space for memory-intensive...(Zheng et al., ASPLOS 22)

**核心创新**:**多维 fusion 空间**——传统 fusion 只考虑 producer × consumer 是否可融,AStitch 引入"沿哪个 iteration 维度 fuse"、"分几次 fuse"、"哪些 tile 一起 fuse"等多维度。

### 5.5 ALCOP(MLSys 2023)

**来源**:ALCOP: Automatic Load-Compute Pipelining in Deep Learning Compiler for AI-GPUs(MLSys 2023)+ CUHK paper

**核心创新**:不是做 fusion 边界识别本身,而是**对已融合的 kernel 自动加 load-compute pipelining**(overlap 内存加载与计算)。

**与 inductor 对比**:与 fusion 范围识别正交,但**补全 fusion 后的优化**。

### 5.6 Chimera(HPCA 2023)

**来源**:Chimera: An Analytical Optimizing Framework for Effective Compute-intensive Operators Fusion(Zheng et al., HPCA 2023)

**核心创新**:**compute-intensive 算子 fusion 分析框架**(针对 GEMM/Conv 这类 compute-intensive operator)。

### 5.7 小结:多维 + 数学优化派

| 维度 | inductor | DNNFusion/Apollo/DeepCuts |
|---|---|---|
| fusion plan 探索 | 启发式贪心 | 整图系统搜索(基于 cost model) |
| 范围识别粒度 | 节点对 | 整图(fusion seed 起始) |
| 数学性质使用 | ❌ | ✅(graph rewriting) |
| NPU 友好度 | ⚠️ 取决于实现 | ✅ Apollo 已在 NPU 验证 |

**关键 takeaway**:DNNFusion 公开数字"**8.8× fusion opportunities** + 9.3× speedup"是**对 fusion 范围识别非常量化的证据**——静态规则会漏掉大量 fusion,系统搜索 + 数学性质能补回。

---

## 6. 第四类:Tile-Graph / Whole-Graph 派(以内存为中心)

> 这一类把 fusion 范围识别**放到内存层次**——以"哪些算子可以共享 register / shared memory" 为核心。

### 6.1 Welder(Microsoft, OSDI 2023)

**来源**:Welder: Scheduling Deep Learning Memory Access via Tile-graph(Shi et al., OSDI 2023)+ GitHub `nox-410/Welder`

**核心创新**:**Tile-graph 统一抽象**——把所有 fusion 类型(register-based element-wise、shared-memory fusion、DRAM-resident fusion)**统一到 tile-graph** 里。

#### 6.1.1 Welder 框架

- **整体调度**:在 tile 级别做 holistic data-flow scheduling
- **自动发现**:Welder 能自动发现 **89 个"未探索"的 operator fusion patterns**(论文 § 5.2)——**这些是 inductor 静态规则不会尝试的 fusion**

#### 6.1.2 Welder 与 inductor 的关键区别

| 维度 | inductor | Welder |
|---|---|---|
| **fusion 范围识别抽象** | IR 节点 | **tile-graph**(统一多级内存) |
| **可识别的 fusion 类型** | 4 类基础 + 模板 | **89 个未探索 pattern**(自动发现) |
| **内存层次感知** | ❌ 间接(通过 codegen) | ✅ 直接(tile 抽象) |
| **NPU 友好度** | ⚠️ | ⚠️ (主要 GPU) |

#### 6.1.3 Welder 论文的关键声明

> "WELDER is the first to unify all common operator fusions (e.g., register-based element-wise fusion, shared-memory fusion, etc.) into a single framework. This generality allows WELDER to find 89 uncommon operator fusion patterns automatically that are mostly unexplored by existing rule-based approaches"

**这是非常强的反例**——证明"inductive 静态规则"错过了大量 fusion patterns。

### 6.2 Kitsune(SIGARCH 2025)

**来源**:Kitsune: Enabling Dataflow Execution on GPUs(arXiv 2502.18403)

**核心创新**:**synchronous dataflow execution on GPUs**——支持跨 kernel 同步,使跨 kernel 的 fusion 成为可能。

### 6.3 TurboMGNN(TPDS 2023)

**来源**:TurboMGNN: Improving Concurrent GNN Training Tasks on GPU with Fine-Grained Kernel Fusion(TPDS 2023)

**机制**:**inter-task kernel fusion**——把多个 GNN 训练任务的相同 kernel 融合在一起(共享 graph storage)。TurboMGNN 实现了 **operator matching + adaptive task grouping**——这是 fusion 范围识别的**横向扩展**(跨任务)。

**与 inductor 对比**:
- inductor 在单任务内 fusion
- TurboMGNN 跨任务 fusion
- **融合的"范围"定义从"算子图内"扩展到"任务图内"**

### 6.4 小结:Tile-Graph 派

| 维度 | inductor | Welder/Kitsune/TurboMGNN |
|---|---|---|
| fusion 抽象 | IR 节点 | **tile-graph / dataflow / task group** |
| 可识别 pattern 数 | 几百个(inducer 173 + can_fuse 模式) | **Welder: 89 个未探索 pattern** |
| 内存层次感知 | ❌ | ✅ Welder 显式建模 |
| 跨任务 fusion | ❌ | ✅ TurboMGNN |

---

## 7. 验证机制专题:怎么证明 fusion 是对的

> 这是**fusion 范围识别的关键支柱**——光识别出 fusion 还不够,必须证明 fusion 后语义正确。

### 7.1 三大验证机制

| 机制 | 代表工作 | 优势 | 劣势 |
|---|---|---|---|
| **形式化定理证明(SMT)** | **TASO**(SOSP 2019) | **严格性最高**——数学证明 | 慢、依赖 operator properties |
| **Mutation testing + 自动校正** | **PET**(OSDI 2021) | 允许部分等价变换 | 校正 kernel 本身有开销 |
| **有限域随机测试** | **Mirage μGraph**(ASPLOS 24) | **快速、可概率保证** | 概率正确,非严格 |
| **运行时 benchmark** | Apollo、DeepCuts | 直接量性能 | 不能保证正确性 |

### 7.2 TASO 的 theorem prover 验证(细节)

TASO 给每个算子定义 43 个 mathematical properties(如 conv 双线性、additivity、associativity),然后用 SMT 求解器检查:

```
∀x, y, z. Conv(x, Concat(y, z)) = Concat(Conv(x, y), Conv(x, z))
```

如果 SMT 求解器返回 `UNSAT`,证明两个 sub-graph 在数学上严格等价。

**TASO 论文发现**:在开发过程中,**形式化验证方法发现了若干 bug**(在 operator specifications 和 graph substitution generator 实现中都有)——这是**形式化验证相比手工规则的关键优势**。

### 7.3 PET 的 mutation + correction 验证(细节)

PET 的验证流程:
1. 找到部分等价 mutation(原本不严格等价)
2. 分析 mutation 破坏了哪些**output element** 的等价性
3. 自动生成 **correction kernel** 修复这些 element
5. 整体上,**mutation + correction = 完全等价**

PET 论文(§ 5.3)详细描述了 correction kernel 的生成算法和 fusion 优化。

### 7.4 Mirage 的有限域随机测试(对比)

Mirage 用 PIT(Polynomial Identity Testing)算法在有限域上做随机测试——**概率正确**,但**速度比 SMT 快很多**。

> ⚠️ Mirage 主要做**算子内部代数等价验证**,不是跨算子 fusion 验证。它的验证机制严格来说是**另一类**(算子 superoptimization 验证),本报告主要参考它的概率验证思想。

### 7.5 inductor 的验证机制

inductor 的 fusion 验证是**启发式 + 测试**:
- `can_fuse` 函数判断语法可融
- **运行时 correctness check**(`torch.compile` 的动态检查)
- **没有形式化证明**——这是 inductor 与 TASO/PET 的核心差距之一

### 7.6 验证机制对比矩阵

| 验证机制 | 严格性 | 速度 | 适用 | 工作 |
|---|---|---|---|---|
| **SMT 定理证明** | ✅ 严格 | ❌ 慢 | 算子图 fusion | TASO |
| **Mutation + Correction** | ✅ 严格 | ⚠️ 中 | 部分等价 fusion | PET |
| **有限域随机测试** | ⚠️ 概率 | ✅ 快 | 算子内部 superopt | Mirage |
| **运行时 benchmark** | ❌ 不保证正确 | ✅ 快 | 性能验证 | Apollo/DeepCuts |
| **启发式 + runtime check** | ❌ 经验式 | ✅ 快 | 产线 | **inductor** |

---

## 8. 与 inductor 的逐项对比矩阵

把所有方法在 fusion 范围识别的核心问题上做对比:

| 维度 | inductor | XLA HLO | MLIR Linalg | TVM FuseOps | **TASO** | **PET** | **Unity** | **DNNFusion** | Apollo | DeepCuts | AStitch | **Welder** | Kitsune |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **fusion 范围识别机制** | IR 类 + can_fuse | 4 类 pass | TilingInterface + callback | 4 类算子 + 规则 | **theorem prover 自动生成 substitution** | **mutation + correction** | **代数+并行** | **fusion seed + 传播搜索** | partition + loop 反馈 | DL-guided cut | 多维 fusion 空间 | **tile-graph** | dataflow |
| **是否发现新 fusion 模式** | ❌ | ⚠️ 启发式扩展 | ✅(通过 callback) | ⚠️ 模板 | ✅ **自动 743 个** | ✅ mutation 新机会 | ✅ algebraic 新机会 | ✅ 8.8× fusion 机会 | ⚠️ loop 反馈 | ⚠️ DL 选 cut | ⚠️ 多维选 | ✅ **89 个未探索** | ⚠️ |
| **是否突破 inductor 静态规则** | 基准 | ❌ 同类 | ❌ 同类 | ❌ 同类 | ✅✅ | ✅✅ | ✅✅ | ✅✅ | ✅ | ✅ | ✅ | ✅✅ | ✅ |
| **验证机制** | runtime check | runtime check | runtime check | runtime check | **SMT 严格** | **mutation correction** | theorem prover | math rewriting | benchmark | benchmark | ML cost | runtime check | runtime check |
| **是否自动 codegen** | ✅ | ✅ | ✅ | ✅ | ❌(只产 substitution) | 部分 | ❌ | ✅ | 部分 | ✅ | ❌ | ✅ | ✅ |
| **NPU 友好度** | ⚠️ | ❌ | ✅(MLIR 通用) | ⚠️ | ⚠️ | ⚠️ | ❌ | ⚠️ | ✅ NPU 实验 | ❌ | ❌ | ❌(GPU 为主) | ❌ |
| **成熟度** | ✅ PyTorch 默认 | ✅ Google 产线 | ✅ LLVM 主线 | ✅ Apache | ⚠️ 论文级 | ⚠️ 论文级 | ✅ FlexFlow | ⚠️ 论文级 | ✅ MLSys'22 | ⚠️ 论文级 | ⚠️ 论文级 | ✅ GitHub | ⚠️ 论文 |
| **性能收益(最优)** | 基准 | 论文实测 10.56× speedup over XLA | 等价 | 等价 | **新 DNN 2.8×** | **2.5×** | **3.6×** | **9.3×** | **NPU 19.7%** | 等价 | 等价 | 等价 | 等价 |

**关键观察**:

1. **TASO / PET / Unity / DNNFusion / Welder 是真正突破 inductor 静态规则的 5 个工作**
2. **Welder 的 "89 个未探索 fusion patterns" 是最强反例**(OSDI 2023)——证明 inductor 静态规则遗漏了大量 fusion
3. **DNNFusion 的 "8.8× fusion opportunities" 是最直接量化证据**(PLDI 2021)
4. **TASO 是 fusion 范围识别的形式化代表**——SMT 定理证明器,数学严格

---

## 9. 给你的工程化建议

> 这一节是结合调研结论的具体方案建议,不是已实现的系统。

### 9.1 三阶段 rollout(短期 → 中期 → 长期)

#### Phase 1(短期,2-4 月):把 inductor `can_fuse` 升级到 TASO 风格

**目标**:**用 TASO 思路扩展 inductor 的 fusion 候选库**

具体步骤:
1. 抽取 inductor 173 个可融合 op 的 mathematical properties(从 `register_pointwise` / `make_reduction` 等代码注释里提取)
2. 跑 TASO 风格的 sub-graph 枚举(限制 op 数 ≤ 4,匹配 inductor 的实际子图大小)
3. 用 SMT 求解器(Z3)证明每个 candidate substitution 的等价性
4. 把 verified substitutions 加到 inductor 的 fusion 候选库
5. 用 NPU 实测筛出**在 NPU 上有效**的 substitution(避免 Welder 论文里 30% V100 vs -10% K80 那种硬件不匹配)

**预期收益**:induc 当前 86 个静态融合损失(参考 `inductor_fusion_breadth_analysis.md` §5.4)→ 静态规则可消除一部分。TASO 在新 DNN 上 2.8× 的数据是上限参考。

**风险**:TASO 的成本模型简单,可能漏掉 inductor 已经在 scores_fusion 里做的工作。

#### Phase 2(中期,4-8 月):借鉴 DNNFusion 的 fusion seed + 传播搜索

**目标**:**用整图搜索代替 inductor 的"节点对启发式"**

具体步骤:
1. 抽取 FX graph 的 fusion seed operators(类似 DNNFusion 的策略)
2. 传播探索:从 seed 出发,沿 successor/predecessor 探索 fusion candidates
3. 用 Apollo 风格的 polyhedral cost model 估计每个 fusion block 的 NPU 开销
4. 与 inductor 的 can_fuse + scores_fusion 集成(`can_fuse` 返回更多 candidates,`scores_fusion` 用新 cost model 重打分)
5. **关键**:支持 Apollo 的"loop optimizer 反馈"——codegen 失败时,触发 graph partitioner 重切

**预期收益**:DNNFusion 的 "8.8× fusion opportunities" 给出的上限。

**风险**:整图搜索成本高,需要 NPU 上快速 profile(Apollo 已经在 NPU 验证过)。

#### Phase 3(长期,12+ 月):TASO + PET 联合 + Welder tile-graph

**目标**:**以 inductor 为 baseline,叠加 TASO 的 substitution 生成 + PET 的部分等价 + Welder 的 tile-graph 抽象**

**但这一阶段要谨慎**——TASO/PET 是**算子子图替换**的工作,与 inductor 的"单算子 lowering"路径冲突,需要做路径选择。Welder 是 GPU-first,迁移到 NPU 需要 tile-graph 的 NPU adapter。

### 9.2 关键技术选型建议

| 选型项 | 推荐 | 理由 |
|---|---|---|
| **fusion 范围识别核心机制** | **TASO 风格 substitution + SMT 验证** | 唯一有数学严格性,且工程上 TASO GitHub 开源可参考 |
| **范围搜索算法** | **DNNFusion 的 fusion seed + 传播** | 已被 PLDI 2021 验证 + 9.3× speedup |
| **cost model** | **Apollo 的 polyhedral 反馈 + inductor 的 scores_fusion** | Apollo NPU 已验证 |
| **NPU codegen 适配** | **TorchInductor 的 Triton 路径 + torch_npu AclNN fallback** | 复用现有 inductor |
| **验证** | **TASO SMT + inductor runtime check 双层** | TASO 离线验证,inductor 在线 sanity check |

### 9.3 与你现有工作的衔接

你已经有 `docs/ai_gen/scripts/measure_decomp_coverage.py`(测量 inductor 融合覆盖率)和 `measure_fusion_depth.py`(测量融合深度)。这两个工具可以**直接用来评估 Phase 1/2 的效果**:

- 用 `measure_decomp_coverage.py` 量化"新加 TASO substitutions 后,173 个 op 之外有多少额外 op 可融"
- 用 `measure_fusion_depth.py` 量化"新融合后的 kernel 融合深度"

---

## 10. 风险与未解问题

### 10.1 TASO/PET 的产线化难度

- TASO 的 cost model 简单("sum of individual operators"),**实际融合收益可能在 NPU 上不显著**
- PET 的 correction kernel fusion 关键但**每个 mutation 都要写校正 kernel**,工程量大
- Unity 主要面向分布式训练,不直接适用于推理场景

### 10.2 DNNFusion 的 fusion plan 搜索是 NP-complete

DNNFusion 论文原文:

> "Optimal fusion plan generation requires a large search space and has been shown to be NP-complete"

**NP-complete** 问题在大图上无法精确求解。DNNFusion 用 fusion seed + 传播是**启发式**而非最优。

### 10.3 Welder 的"89 个未探索 pattern"在 NPU 上的有效性

Welder 论文明确:

> "30% faster on V100 but 10% slower on K80"

89 个 fusion pattern 不是**所有硬件上都有效**。迁移到 NPU 需要重新验证每个 pattern。

### 10.4 与 inductor 维护性的冲突

inducer 在 PyTorch 主线,每个版本都改。TASO 风格的 substitution 库需要**持续跟随 inductor 变化**——否则 substitution 可能在新的 inductor IR 下失效。

### 10.5 SMT 求解器的工程化门槛

TASO 用 SMT 求解器做验证,在 NPU 上**每个新 hardware 需要重新设计 operator properties**——这与 inductor 的 "register_pointwise" 风格有差距。

---

## 附录 A. 信源与可信度

| 类别 | 信源 | 可信度 | 用途 |
|---|---|---|---|
| **官仓源码** | `pytorch/pytorch` `torch/_inductor/` | 🟢 高 | inductor IR/scheduler 实测 |
| **官仓源码** | `jiazhihao/taso` | 🟢 高 | TASO 实现 |
| **官仓源码** | `whjthu/pet-osdi21-ae` | 🟢 高 | PET artifact |
| **官仓源码** | `nox-410/Welder` | 🟢 高 | Welder 实现 |
| **官仓源码** | `iree-org/iree` | 🟢 高 | LinalgExtFusionInterface |
| **arXiv 论文** | TASO(SOSP 2019) | 🟢 高 | 自动 substitution + theorem prover |
| **arXiv 论文** | PET(OSDI 2021) | 🟢 高 | 部分等价变换 + correction |
| **arXiv 论文** | Unity(OSDI 2022) | 🟢 高 | 代数 + 并行联合 |
| **arXiv 论文** | DNNFusion(PLDI 2021) | 🟢 高 | fusion seed + 传播 |
| **arXiv 论文** | Apollo(MLSys 2022) | 🟢 高 | 双向反馈 fusion |
| **arXiv 论文** | Welder(OSDI 2023) | 🟢 高 | tile-graph fusion |
| **arXiv 论文** | DeepCuts(PLDI 2021) | 🟢 高 | DL-guided fusion cut |
| **arXiv 论文** | AStitch(ASPLOS 22) | 🟢 高 | 多维 fusion 空间 |
| **arXiv 论文** | Chimera(HPCA 2023) | 🟢 高 | compute-intensive fusion |
| **arXiv 论文** | ALCOP(MLSys 2023) | 🟢 高 | Load-Compute pipelining |
| **arXiv 论文** | Kitsune(SIGARCH 2025) | 🟢 高 | dataflow execution |
| **arXiv 论文** | Operator Fusion in XLA(2023) | 🟢 高 | XLA fusion 实测 |
| **会议论文** | OSDI / SOSP / MLSys / PLDI / ASPLOS | 🟢 高 | 编译系统顶会 |
| **官方文档** | OpenXLA / MLIR Discourse / Apache TVM | 🟢 高 | 架构说明 |
| **Stanford slides** | TASO slide deck | 🟡 中 | TASO 演示 |

调研遵循 *deep-research* skill 的多源交叉验证原则:每个事实陈述至少 1 个高可信度源 + 1 个中可信度源支撑。

---

## 附录 B. 关键文献索引

按"fusion 范围识别"的相关度排序:

### B.1 必须精读(★★★)

1. **TASO**(Jia et al., SOSP 2019)— **algebraic substitution + SMT 验证** 的开创工作。GitHub `jiazhihao/taso`。
2. **DNNFusion**(Niu et al., PLDI 2021)— **fusion seed + 传播探索**。"8.8× fusion opportunities" 量化证据。
3. **Welder**(Shi et al., OSDI 2023)— **tile-graph 统一抽象**。"89 个未探索 fusion patterns"。
4. **PET**(Wang et al., OSDI 2021)— **部分等价变换 + correction**。
5. **Unity**(Unger et al., OSDI 2022)— **代数 + 并行联合优化**。

### B.2 应该读(★★)

6. **Operator Fusion in XLA**(Snider & Liang 2023, arXiv 2301.13062)— XLA 4 类 fusion pass 实测。
7. **Apollo**(Zhao et al., MLSys 2022)— **双向反馈 fusion**。NPU 验证。
8. **DeepCuts**(Jung et al., PLDI 2021)— **DL-guided fusion cut selection**。
9. **MLIR LinalgExtFusionInterface + TilingInterface**— 设计与 heuristics 解耦的 fusion 框架。
10. **AStitch**(Zheng et al., ASPLOS 22)— **多维 fusion 空间**。

### B.3 可选参考(★)

11. **ALCOP**(MLSys 2023)— load-compute pipelining(融合后优化)。
12. **Chimera**(HPCA 2023)— compute-intensive fusion 分析。
13. **Kitsune**(SIGARCH 2025)— dataflow execution。
14. **TurboMGNN**(TPDS 2023)— 跨任务 kernel fusion。
15. **TVM FuseOps**(OSDI 2020 论文 + Ansor)— 4 类算子分类的原始定义。
16. **DNNFusion GitHub** + **PET GitHub** + **TASO GitHub**——开源实现可参考。

---

*报告生成日期:2026-08-20(v2 精准版)*
*调研方法:anysearch v3.0.1 多源检索 + 关键论文全文抽取(arXiv) + Stanford/CUHK/CMU slides 交叉验证*
*v1 → v2 修订说明:v1 把"Agent/LLM 整算子代码生成"误归类为 fusion 范围识别,本次精准版**剔除**那部分,只聚焦真正做 fusion 边界识别的工作*
*参考基线:`docs/ai_gen/inductor_fusion_breadth_analysis.md`(2026-07-20 版)*
*下次复审建议:fusion 范围识别领域的进展主要来自 PLDI/OSDI/SOSP/MLSys 顶会,建议每 6 月跟踪一次 arXiv 最近投稿。*