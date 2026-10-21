# TorchInductor 技术调研:PyTorch 2.10→2.14 社区演进、昇腾 PTA 适配 Gap 与搜广推投入方向

> 调研日期:2026-10-21。回答三个问题:
> ① PyTorch 2.10.0(2026-01 发布)至今社区在 TorchInductor 上规划了哪些重点领域、演进趋势是什么;
> ② 昇腾 PTA(v2.13.0-26.2.0 分支)当前适配与社区支持特性的 Gap 在哪里(op 支持/融合范围/测试用例支持度/泛化机制);
> ③ 从搜广推业务视角,follow 社区进展的优先级与重点投入方向。
>
> 方法:PyTorch 主仓(main 2.16.0a0 + tags v2.10.0~v2.14.0)git 量化分析 + PTA 仓(v2.13.0-26.2.0)代码盘点 + 社区公开信源(官方博客/dev-discuss/RFC/issue)调研。文中所有结论以 [S#] 标注来源,完整清单见文末附录。
> 复用前期调研:《inductor_ci_test_coverage_analysis.md》[D1]、《inductor_fusion_breadth_analysis.md》[D2]、《v2130_inductor_test_analysis.md》[D3]、《torchinductor_test_inventory.md》[D4]、《inductor_triton_codegen_extension_usage.md》[D5]。

---

## 0. 摘要(TL;DR)

**社区侧(问题 1)**

1. **投入强度持续上升**:v2.10.0→HEAD(2.16 dev)共 15,999 个提交,其中 `torch/_inductor/` 相关 2,970 个(18.6%),连同 `test/inductor/` 达 36.8%;代码增量逐季放大(净新增行数从 2.10→2.11 区间的 +2.3 万放大到 2.14→HEAD 区间的 +5.5 万),**大特性落地越来越重** [S1]。
2. **六条 GEMM 通道格局形成**:GEMM/模板后端从 v2.10.0 的 "Triton + CUTLASS + CK" 扩张为 **ATen / Triton / CUTLASS(EVT) / CuTeDSL / NVGEMM(QuACK/Universal) / CK(CKTile) / FlyDSL(gfx950)** 多通道竞争,max-autotune 中互博选优 [S1][S7]。
3. **FlexAttention 是单点最大投入**:`kernel/flex` + `kernel/flex_gemm` 在 2.14→HEAD 区间连续两季为第一热点;测试域从 v2.13.0 的 332 方法涨到 main 的 645(接近翻倍) [S1][D3]。
4. **低层控制力成为新战场**:TLX(PyTorch-Triton 3.7 起默认启用)、自动 Warp Specialization、Gluon RFC(#188212)、cuTile RFC(#175311)、Helion(2.14 注册为第三个 Native DSL)——社区在"编译器自动生成"与"专家手写 kernel"之间搭建多层连续谱 [S5][S7][S8][S9]。
5. **后端扩展点正式化(泛化)**:`heuristics/` registry 重构、device_backends hooks、extension_backends 测试目录、Regional AOTI(#199582)——上游正把"厂商 out-of-tree 适配"从 monkey-patch 转向正式接口;华为也在 issue #201537 公开了 6 个上游扩展点 PR(含 cudagraph trees 去耦 #194598、dynamo device interface 反射注册 #195263) [S1][S10][S6]。
6. **发布节奏改为每 2 个月一版**(2.10 起宣布),叠加测试集每版本 ~35% 膨胀,后端跟随成本被系统性放大 [S7][D3]。

**Gap 侧(问题 2)**

1. **架构性依赖 monkey-patch**:PTA 以 56 个 `patch_*` 函数(33 个文件)、21 处对 `torch._inductor.*` 直接属性赋值维持适配,而社区正在把这些接缝正式化为扩展接口——**上游每重构一次,patch 面重做一遍**;华为自己已承认此路线并转向"上游扩展点共建" [S2][S6]。
2. **Op/融合范围**:v2.13.0-26.2.0 分支 fallback 清单 **847 条目/340 概念算子**(NPU_EXTRA 519/198 + TORCH_NATIVE 328/152),即约四成上游 inductor op 概念仍走 ACLNN 预融合 fallback;decomposition 口径下 NPU 可融合算子覆盖率为 GPU 的 50.3%(2026-08 口径实测) [S2][D2]。
3. **测试支持度是最薄弱面**:上游 v2.13.0 inductor 测试 176 文件/5,388 静态方法;PTA 自建 `test/_inductor/` 26 文件/159 用例(静态≈3%),对上游用例 fork 为 0;CI 白名单 126 个上游测试文件中 inductor 仅 2 个,另有 4,068 条 skip 条目 [S2][S3][D3]。
4. **特性级 Gap**:社区 2.11~2.14 的主线特性(FA4/CuTeDSL 后端、NVGEMM、Helion、autoWS、@dynamic_spec、Regional AOTI、strict numerics、compile_to_python)在 PTA 中均无对等实现或仅有部分映射(aclgraph↔cudagraph、DVM↔模板融合) [S2][S3][S7]。
5. **搜广推场景的特别缺口**:PTA `benchmarks/inference/` 收录 DCNv2/DIN/ETA/MMOE 四个推荐模型,但**均未接入 torch.compile**(grep 无命中)——搜广推负载在 NPU 上的编译收益未被官方验证 [S3]。

**优先级侧(问题 3)**

搜广推负载(embedding 密集、变长序列、稀疏 scatter、多表 lookup、动态 shape、训练推理混合)与社区投入方向的交集中,**建议按四层排序**:
- **P0**:device 注入接入社区 opinfo/test_torchinductor 正确性矩阵(一次性继承 5,388 方法分母的看护)+ fallback 清单收敛(每收敛 1 概念算子平均恢复 ~4.7 项矩阵)+ embedding/大规模 pointwise 融合;
- **P1**:图运行时对齐(aclgraph ↔ user_streams/static launcher/多流)、通信-计算重叠(社区 simple_overlap 已默认开启,PTA 已适配 overlap_scheduling 8 函数,应继续)、动态形状(@dynamic_spec 方向);
- **P2**:Regional AOTI 跟进(其 RFC 首要场景就是 torchrec DLRM 稠密块)、strict numerics 精度对齐、FlexAttention(序列特征 attention 化趋势);
- **P3**:TLX/DSL 生态跟踪与人才储备(不急于落地,但搜广推 kernel 定制红利已在社区显现)。

---

## 1. 社区演进全景:v2.10.0 → v2.14.0(2026-01 ~ 2026-09)

### 1.1 节奏与投入强度

| 区间 | torch/_inductor 提交 | test/inductor 提交 | 全仓提交 | inductor 占比 | 净代码增量(_inductor) |
|---|---:|---:|---:|---:|---|
| 2.10→2.11 | 494 | 459 | 2,729 | 18.1%(含测试 34.9%) | +22,767/−6,682 |
| 2.11→2.12 | 630 | 534 | 2,963 | 21.3%(39.3%) | +22,309/−6,166 |
| 2.12→2.13 | 773 | 741 | 3,356 | 23.0%(45.1%) | +29,816/−7,997 |
| 2.13→2.14 | 539 | 554 | 3,026 | 17.8%(36.1%) | +41,885/−11,402 |
| 2.14→HEAD | 560 | 646 | 4,100 | 13.7%(29.4%) | +55,079/−12,341 |

来源:[S1](pytorch 仓 git log 区间统计,HEAD b48b4a49d28,2026-10-21)。提交数在 2.12→2.13 达峰,代码体量在后期爆发,说明单 PR 规模变大(大特性落地)。

**发布节奏**:2.10 release blog 明确 "For 2026, the expected release cadence will be increased to 1 per 2 months"。实际节奏:2.10.0(2026-01-21)→2.11.0(2026-03-23)→2.12.0(2026-05-13)→2.13.0(2026-07-08)→2.14.0(2026-09) [S7]。

### 1.2 重点领域一:多 DSL/GEMM 后端格局(NVGEMM / CuTeDSL / FlyDSL / Helion)

v2.10.0 的 autotune GEMM 后端集合为 `ATen, Triton, CUTLASS, CK, CKTILE, CPP`;HEAD 已扩为 `ATen, Triton, CUTLASS, CUTEDSL, FLYDSL, NVGEMM, CK, CKTILE, CPP`,**净增 3 个一级后端** [S1]:

| 后端 | 引入区间 | 定位 | 证据 |
|---|---|---|---|
| NVGEMM(Universal GEMM) | 2.11 周期(2025-12 首提 b8e27ea1c64) | NVIDIA cutlass.operators Universal GEMM,2.14 起带 epilogue fusion/scaled/NVFP4/grouped-reduction epilogue,与 Triton/ATen 竞争 | [S1][S7] |
| CuTeDSL(Native DSL) | 2.12~2.13(#176546 Blockscaled GEMM 起) | 2.13 确立 "Native DSL registry",首个用户可见落地为 GEMM/RMSNorm(Quack 系 kernel) | [S7] |
| FlyDSL | 2.15 周期(2026-08 首提 b2ba1d6124c) | ROCm gfx950 MXFP8 grouped GEMM,experimental | [S1] |
| Helion | 2.14 注册(#190636) | Python DSL,注册为第三个 Native DSL,**本版不路由任何算子**,官方声明是"后续版本 kernel override 的基座" | [S7][S9] |

**趋势判断**:GEMM 这一搜广推/LLM 共同的性能核心,已从"单一 Triton 模板"演化为"多 DSL 通道 autotune 竞争";`kernel/gemm_epilogue{,_analysis,_codegen,_utils}.py` epilogue 融合框架(2.14,49db9c6333d)把"GEMM + 后处理点融合"做成了一等公民机制 [S1]。

### 1.3 重点领域二:FlexAttention 家族(单点最大投入)

- commit 触达:`kernel/flex` 在 2.12→2.13 区间 88 次;`kernel/flex_gemm` 在 2.13→2.14 / 2.14→HEAD 区间分别 105/112 次,**连续两季第一热点** [S1]。
- 后端矩阵:Hopper/Blackwell 上 FlexAttention 获得 FA4(CuTeDSL 自动生成)后端,1.2~3.2x 加速(2.11);MPS 移植(2.13,约 12.3x sparse);`test_flex_gemm.py` 259 方法在 main 落地,Flex 测试域 v2.14 起 332→645 接近翻倍 [D3][S7]。
- 2.14 补齐确定性反向(#174813)、varlen-M reductions、chunked GLU epilogue [S1][S7]。

### 1.4 重点领域三:低层控制力(TLX / autoWS / Gluon / cuTile)

社区在"自动 codegen"与"专家手写"之间铺了一条连续谱,这是窗口期最鲜明的战略转向:

| 机制 | 状态 | 要点 | 来源 |
|---|---|---|---|
| TLX(Triton Low-Level eXtensions) | 已落地 | PyTorch-Triton 3.7 引入 Plugin Extensions 系统,TLX 经 utlx 包 out-of-tree 加载,官方承诺 "TLX will be enabled by default on all Triton releases going forward";Inductor 侧 `tlx_mode` 三态开关,TLX mm/flex 模板(Blackwell)进入 max-autotune 竞争 | [S5][S8][D5] |
| 自动 Warp Specialization(autoWS) | 进行中 | `warp_specialize=True` 对 handwritten/TorchInductor/Helion kernel 均可用;B200 flash attn fwd 达 stock Triton 1.5-2x;长期规划含 "IR 级手写-自动 kernel 融合(megakernel)"(当前 Inductor 明确不支持与 user-defined Triton kernel 融合) | [S8] |
| Gluon 后端 | RFC OPEN | Triton 之下更低层前端(显式 tile layout/warp spec/TMA/MMA),让 Inductor autotune 专家手写 Gluon kernel;三阶段:gfx950 flex-attn fwd → epilogue+backward → Blackwell+GEMM | [S11] |
| cuTile / Tile IR | RFC OPEN | NVIDIA 官方推动,"Opt-in only: Triton remains the default";Tile IR 定位 post-Blackwell 跨架构抽象;与 Helion 形成依赖关系(Helion cuTile 后端以 Inductor cuTile 支持为前提) | [S12] |

### 1.5 重点领域四:运行时与图(cudagraph / 多流 / 通信重叠 / AOTI)

- **图捕获设备无关化**:2.12 引入 `torch.accelerator.Graph` 设备无关 capture;torch.cond 进 CUDA Graph(NVIDIA),**"Inductor ordering support planned for a future release"**;2.14 torch.while_loop CUDA graph capture [S7]。
- **cudagraph 体系持续强化**:CUDAGraphPolicy 可插拔(2.12,#180163)、static Triton launcher(2.11 起)、多流语义正式看护 `test_user_streams.py`(71 方法,2.13 周期新增)[S1][D3]。
- **通信-计算重叠**:2.14 `simple_overlap` 默认开启(#184240/#184235);`reorder_for_locality_in_training`(#186643);`decompose_shard_dim_alltoall` 等通信 config 族(2.12~2.14)[S1][S7]。
- **AOTInductor 三级跳**:2.14 外部常量零拷贝(`AOTInductorModelContainerCreateWithExternalConstants`,#188643)、单次 codegen 同时产 JIT+AOTI 包(#184735 等)、pinned async copy;**2026-10-02 新 RFC "Regional AOTInductor"(#199582):区域级增量编译,首要场景 torchrec DLRM 稠密块** [S7][S13]。
- **编译管线**:2.13 kernel 编译从线程池改为子进程池(绕开 GIL,#181267 等);2.15 周期出现 inductor lite mode、module-level kernels [S7][S1]。

### 1.6 重点领域五:后端扩展点正式化(泛化支持)

这是对 NPU 类 out-of-tree 后端影响最深的暗线:

1. **代码结构**:`template_heuristics/` → `heuristics/` 重构(2.14,1374acc2890,经历 4 次 revert 才落定),模板后端注册走向通用 registry + lazy import;cutlass 从 `codegen/cuda/` 独立(2.11)[S1]。
2. **测试结构**:main 新增 `test_device_backends.py` / `test_device_backend_hooks.py` / `test_extension_backend.py` / `test_choices_composition.py` 等后端抽象测试(~75+ 方法),`extension_backends/` 目录(2.11 周期)成为外部后端扩展点测试 [S1][D3]。
3. **对称插件口子**:`config.inductor_choices_class` 可被外部替换(torchTLX 是第一个官方实例);`register_backend_for_device` 机制 [D5]。
4. **厂商共建**:IBM Spyre 承诺 "generalizing dataflow accelerator enablement in torch.inductor"(KTIR)[S14];**华为 TorchNPU issue #201537(2026-10-20)公开 6 个上游扩展点 PR**:cudagraph trees 去耦重构(#194598)、storage creation hook(#193770)、dynamo device interface 反射注册(#195263)、backendfeature 收回设备能力(#193355)、Torch Logging C++ 通知(#192513)、fx wrapper 支持 non-Triton fused kernel(#187649/#187938) [S6]。

### 1.7 重点领域六:正确性/数值严格性与动态形状

- **strict numerics 体系**(2.14):`config.numerics` 旋钮 + `strict_pointwise`/`strict_reduction`/`strict_signed_zero`/`strict_output_strides`,`test_strict_numerics.py` 20 方法——对异构硬件精度对齐是硬看护 [S1][D3]。
- **动态形状**:2.14 `@dynamic_spec` 声明式动态形状(#187639 等);`test_unbacked_symints.py` 26→54 方法(unbacked 近乎翻倍,LLM 变长场景)[S7][D3]。
- **精度模拟**:`emulate_precision_casts*` 体系替代旧 `force_same_precision`;FMA-based addcdiv lowering(2.12) [S1][S7]。

### 1.8 测试看护演进:分母每版本 +35%

| | v2.10.0(2026-01) | v2.13.0(2026-07) | main 快照(2026-09-30) |
|---|---:|---:|---:|
| 测试文件 | 143 | 176 | 205 |
| 静态测试方法 | 3,978 | 5,388 | 7,284 |

来源:[D3](v2.13.0 分支 docs/ai_gen/v2130_inductor_test_analysis.md,AST 口径)。
- 四域接近翻倍:缓存与并行编译 +104%、分布式 +105%、融合与调度 +98%、Triton 代码生成 +96%;CUDAGraphs +87% [D3]。
- GPU CI 实跑口径:NVIDIA L4 单卡 23,749 用例(2026-08 实测),其中 opinfo 单文件展开 3,697 项;单卡全量推算约 2.87 万 [D4]。

### 1.9 社区规划机制与"未来"信号

- **半年 roadmap 机制常态化**:Meta 2026 H1(11 份方向文档,含 Compiler/Triton 独立 roadmap)与 2026 H2(合并版公开 PDF) [S15]。
- **Inductor meetup 周记**(ezyang 主持):2.10 排期 autotune/locality 变体、2.11/2.12 多流、FlexAttention 后端可配置化(2.11 兑现)、cuTile 对接讨论、对 Helion 与 Inductor 关系的架构性顾虑 [S16]。
- **规划中/进行中特性汇总**(截至 2026-10-21):

| 领域 | 状态 | 来源 |
|---|---|---|
| Native DSL registry(Triton/CuTeDSL/Helion) | 已落地;Helion 算子路由规划中 | [S7][S9] |
| NVGEMM 扩展(autotune 时间与性能) | 进行中(官方声明继续投入) | [S7] |
| Gluon(gfx950→Blackwell/GEMM) | RFC 三阶段 | [S11] |
| cuTile(Tile IR,post-Blackwell) | RFC opt-in | [S12] |
| autoWS(megakernel/IR 级融合) | 短/长期规划 | [S8] |
| Regional AOTI(torchrec DLRM 首场景) | RFC OPEN(2026-10-02) | [S13] |
| torch.cond/while_loop 的 Inductor ordering | "planned for future release" | [S7] |
| 复数 torch.compile | 实验性(tracking #194061) | [S7] |
| fp8/mxfp8/NVFP4 融合与 autotune | 进行中 | [S7][S17] |
| Blackwell/Rubin(sm_100/107/110)覆盖 | 进行中(Jetson Thor 适配 #201193) | [S7][S17] |
| TLX out-of-tree 自定义后端(Intel/CPU) | 规划中 | [S5] |
| LLM 辅助 kernel 开发(KernelBot 等) | 探索/进行中 | [S18] |

---

## 2. PTA v2.13.0 适配现状与社区 Gap

### 2.1 适配架构盘点

**规模与后端体系** [S2]:
- `torch_npu/_inductor/`:**165 个 .py / 89,757 行**,docs 65 个文件;5 种后端:`default`(Triton-Ascend)/`mlir`(已标废弃)/`dvm`/`ascendc`/`triton_experimental`(物理隔离的实验后端,自有 codegen 7,873 行);
- 三套图优化/融合引擎并存:Triton 管线(fx_passes + scheduler patch)、DVM(`third_party/dvm`,mlir_fusion/graph_fusion)、CATLASS GEMM 模板(codegen/catlass/,950DT);
- torchair(`backend="npu"`)是独立于 inductor 的 GE 静态图路径;`backend="npugraphs"`(aclgraph 管理)内部复用 inductor。

**版本对应**:v2.13.0-26.2.0 = 上游 PyTorch 2.13.0 + PTA 产品版本 26.2.0;同一分支可构建 2.13.0/2.14.0/2.15.0 三条线(`requirements_2.13.txt` pin `torch==2.13.0+cpu`,根 requirements 已指向 2.15 nightly);CANN docker 默认 9.1.0;**官方约束:Inductor-Ascend 面向 Ascend 950DT 系列** [S2]。

**patch 面**:**56 个 `def patch_*` 函数分布于 33 个文件,对 `torch._inductor.*` 直接属性赋值 21 处**,覆盖:codegen.common(注册 `register_backend_for_device("npu", NPUCombinedScheduling, NPUPythonWrapperCodeGen, CppWrapperNpu)`)、lowering(reduction 分发/fallback 注册)、scheduler(融合阈值/MultiTemplateBuffer/图分区签名)、triton_heuristics(cached_autotune 整体替换为 NPUCachingAutotuner 等)、select_algorithm、async_compile、cpp_builder、flex_attention patch、dynamo NpuInterface 等 [S2]。

**fallback 清单(v2.13.0-26.2.0 实测)** [S2]:

| 清单 | 条目 | 去重概念算子 | 构成 |
|---|---:|---:|---|
| NPU_EXTRA_FALLBACK_LIST | 519 | 198 | aten 128 + prims 58 + 其它 12 |
| TORCH_NATIVE_FALLBACK_LIST | 328 | 152 | aten 142 + prims 3 + 其它 7 |
| **FALLBACK_LIST(合计)** | **847** | **340** | 去重后 aten 263 + prims 58 + 其它 19 |
| INDIRECT_MEM_FALLBACK_LIST | 22 | 9 | cat/embedding/gather/index/scatter 系 |

对照历史口径:2026-08 master 分支 NPU_EXTRA_FALLBACK_LIST 为 592 条/221 算子,后收敛至 136 算子/409 条目 [D1][D2];v2.13.0 分支引入 TORCH_NATIVE_FALLBACK_LIST 并重整为 519+328 结构——**清单结构在版本切换中持续重组,口径不直接可比,但"数百个概念算子依赖 ACLNN fallback"的基本面未变** [S2]。

**融合/图优化 pass 资产**:31 个注册制自定义 FX pass(PRE 4 + POST 常开 23 + 默认关闭 4,如 `batch_embedding_fusion_pass`、`cat_slice_cat_fold_pass`),NPU 化的 overlap_scheduling(8 个上游函数)、parallel_scheduler 多流调度、算子模板融合(mm/addmm/bmm/grouped_mm/multi_slice_concat/flex_attention)+ CATLASS GEMM 模板 [S2]。

### 2.2 Op 支持与融合范围 Gap

**(1)融合参与广度**:decomposition 口径实测(2026-08,torch_npu master vs pytorch main):GPU 可融合 173 个 primitive,NPU 可融合 87 个,**覆盖率 50.3%;扣除 34 个永不出现的惰性 prims 后,运行时相关真实差距 52 个算子(55.2%)** [D2]。fallback ≠ 慢(ACLNN kernel 本身可能融合),但该指标量化的"参与 inductor 融合管线的算子面"确实只有 GPU 的一半 [D2]。

**(2)模板/prologue-epilogue 机制差距**:上游 mm 模板 `prologue_loads_all_inputs=True` 允许前驱 Pointwise 吃进 GEMM kernel;CUTLASS/CuteDSL/NVGEMM 模板显式拒绝 prologue;PTA 侧的 CATLASS 模板与 NPUTritonTemplate 是独立实现,prologue/epilogue 能力未对齐上游 `get_allowed_prologue_inps()` 机制的演进速度 [D2][S2]。

**(3)override 清单规模**:NPU 专属 override-lowering 仅 17 条(mm/bmm/addmm/cumsum/mean/max/min/argmax/argmin/var_mean/cat/pow/flex_attention 系)+ 11 条 indirect-mem 条目;对比上游 6 条 GEMM 通道 + epilogue 融合框架 + flex_gemm 13 文件,模板层深度差距显著 [S1][S2]。

**(4)opinfo 可过上界(推算,2026-08 口径)**:以 GPU 可跑面 3,015 项为基准,NPU 对上游 opinfo 矩阵的可过上界 62%~66%,已知必然缺口(fallback 清单命中)15%~21%,fp64 精度风险 17%~18% 需逐例决策;**每收敛 1 个概念算子平均恢复约 4.7 项矩阵覆盖** [D1]。

### 2.3 测试用例支持度 Gap(最薄弱面)

**总量对照(v2.13.0 同代)** [S2][S3][D3]:

| 口径 | 上游 v2.13.0 | PTA v2.13.0-26.2.0 | 比值 |
|---|---:|---:|---:|
| inductor 测试文件 | 176 | 26(test/_inductor) | 14.8% |
| 静态用例(上游 AST 5,388 / PTA grep) | 5,388 | 159 | **约 2.9%~3.0%** |
| 对上游用例的 fork 数 | — | **1**(`test_codecache.py` 大幅裁剪,189 行 vs 上游 5,425 行)+1 个适配型 autotune 测试 | — |

- PTA `test/_inductor/` 为**纯自建体系**(24/26 纯自建),覆盖域:融合(约 90,含 overlap scheduling/DVM MLIR fusion)、动态形状(35)、lowering/codegen(约 30)、autotune(11)、缓存(3)、flex(5);**无 AOTI 专项、无 cpp_wrapper 专项、cudagraph 等价物在 test/npu/(aclgraph 95 用例 + graph_tree 90 用例)** [S3]。
- **注意口径**:torch_npu master(2026-08,对应 2.10 代)自建为 126 文件/547 用例 [D1];v2.13.0-26.2.0 分支实测 26 文件/159 用例 [S3]——版本切换中测试体系重构(部分等价看护移至 test/npu/ 的 aclgraph/graph_tree),两口径不可直接衔接,但均远小于上游分母。
- 域级投入密度(547 分子口径,v2.13.0 分母):全量 10.2%;**P0 算子正确性基线 5.8%(67/1,148)仍是最短板**;FlexAttention 1.5%、Autograd 训练 1.0%、缓存 0.8%、调试工具链 1.9%、分布式 0% [D3]。
- **CI 注入机制已存在但规模极小**:GitCode 流水线把上游 `pytorch/test` 打包为 artifact,以 `PYTORCH_TESTING_DEVICE_ONLY_FOR=privateuse1` 在 NPU 上收集执行;但白名单仅 126 个上游测试文件(其中 **inductor 仅 2 个**:`test_minifier_utils.py`、`test_op_completeness.py`),另有 4,068 条 skip 条目——**"上游测试树 + device 注入"机制已通,inductor 域尚未真正接入** [S3]。
- 外围层对比:test/dynamo fork 上游 74% 文件但用例保留率仅 25%(1,681/6,609);opinfo 核心套件曾 fork 适配 8,239 项(2.10 时代口径)——**外围强、后端弱,inductor 是断层** [S3][D1]。
- 分母膨胀压力:上游 inductor 测试每版本 +35%,v2.13 锁定基线 4,883 应适配方法,v2.10.x 锁定基线 3,522——**接入越晚,一次性面对的基线越大** [D3]。

### 2.4 特性级 Gap 对照表(社区 2.11~2.14 主线特性 vs PTA 现状)

| 社区特性 | 社区状态 | PTA v2.13.0-26.2.0 现状 | Gap 定级 |
|---|---|---|---|
| 多 GEMM 通道 autotune(CUTLASS EVT/CuTeDSL/NVGEMM/FlyDSL) | 已落地 | CATLASS 单通道(950DT)+ NPUTritonTemplate | **结构性差距** |
| epilogue 融合框架(gemm_epilogue_*) | 2.14 落地 | 模板级零散融合(view epilogue 等) | 结构性差距 |
| FlexAttention FA4/CuTeDSL/flex_gemm | 已落地(多后端) | NPU flex 模板(5 文件 7,234 行),仅 Triton 通道 | 大 |
| TLX/autoWS/Gluon/cuTile/Helion | 落地/规划中 | 无对等物(triton-ascend 生态独立演进) | 大(但可暂缓) |
| cudagraph 体系(Policy 插件/static launcher/user_streams 看护) | 已落地 | aclgraph 等价实现 + graph_tree(90 用例),未对齐 static launcher/user_streams 看护面 | 中大 |
| simple_overlap 通信-计算重叠默认开 | 2.14 | 已适配 overlap_scheduling 8 函数 + parallel_scheduler(需显式开启) | 中(方向一致) |
| AOTI(零拷贝常量/单遍 codegen/Regional) | 落地/RFC | 有 aoti_shim + export.py patch_aot_load + MegaCache;官方文档注明 AOTI 暂不支持叠加 Catlass | 中大 |
| @dynamic_spec 声明式动态形状 | 2.14 实验入口 | shape_handling 档位机制(NPUShapeHandling)+ 符号分组 autotune(NPUSymbolicGroupedAutotuner) | 中(自有方案) |
| strict numerics | 2.14 | 无对等(有 npu_compare 精度比对工具) | 中 |
| compile_to_python/load_from_python | 2.14 | 无 | 小 |
| heuristics registry/device_backends hooks 扩展点 | 进行中 | 全靠 monkey-patch;#201537 列出的 6 个扩展点 PR 为对冲动作 | **结构性(见 2.5)** |
| strict 测试矩阵/opinfo 全矩阵看护 | 持续膨胀(+35%/版本) | 自建 159 用例 + 白名单 2 个 inductor 文件 | **最大缺口** |

来源:[S1][S2][S3][S7][D2][D3]。

### 2.5 结构性风险:monkey-patch 面与上游泛化的赛跑

1. **维护成本模型**:上游 2.10→HEAD 期间 `torch/_inductor/` 310→451 个文件,两次大重构(cutlass 迁移、heuristics/ registry,后者 4 次 revert 后落定)都发生在 patch 面覆盖的模块上;PTA 56 个 patch 函数中,scheduler/triton_heuristics/select_algorithm 恰是上游变更最频的 top 文件 [S1][S2]。
2. **社区已给出解法路线图**:上游扩展点(backend registry、choices class、device hooks)+ 华为 #201537 的 6 个上游 PR,本质是把"隐式接缝"变成"显式契约"。**越早上游化,长期维护成本越低**;Spyre 团队同样走"向社区回馈通用化"路线 [S6][S14]。
3. **硬件代际约束**:Inductor-Ascend 官方约束 Ascend 950DT;图优化 pass 仅 950DT 生效;fp64 精度风险(910B4)等——硬件边界决定了部分社区用例结构性不可移植,需要明确的"不做集"(上游厂商专属 601 方法即先例) [S2][D1]。

---

## 3. 搜广推视角:优先级与重点投入方向

### 3.1 搜广推负载特征与社区动向的交集

| 搜广推负载特征 | 社区对应动向(证据) | PTA 现状 |
|---|---|---|
| Embedding 密集(多表 lookup/大规模 sparse) | Regional AOTI RFC **首要场景 = torchrec DLRM 稠密块** [S13];RecSys 生产级 kernel 官方博客(TLX)[S19] | `batch_embedding_fusion_pass` + embedding lowering 开关(TORCHINDUCTOR_ENABLE_EMBEDDING_LOWERING);embedding 仍在 fallback/override 清单内 [S2] |
| 变长序列/动态 shape(序列特征、bucket) | @dynamic_spec、unbacked symints 测试翻倍、Dynamic Shape Recompilations 专场(PTC 2025) [S7][S18] | shape_handling 档位 + 符号分组 autotune;动态形状自建用例最厚(35 个,v2.13 分支) [S2][S3] |
| 稀疏 scatter/index/gather | fallback 收敛目标中 scatter/index_put/cumsum 系项数最集中 [D1] | INDIRECT_MEM 清单 9 算子 + SIMT 模板模式 [S2] |
| 训练推理混合/多卡 | compile-on-one-rank(2.14)、simple_overlap、bucket all_gathers/reduce_scatters [S7] | overlap_scheduling 已适配;分布式自建用例 0 [D3] |
| GEMM 密集(DNN 排序模型/双塔) | 6 条 GEMM 通道 + epilogue 融合框架 [S1] | CATLASS 单通道 + mm/bmm/addmm/grouped_mm 模板 [S2] |
| 低精度量化推理 | fp8/mxfp8/NVFP4 融合与 autotune 进行中 [S17] | quantized 系在 TORCH_NATIVE 清单内(fallback) [S2] |
| **业务模型官方验证** | 社区:torchbench/huggingface/timm 三套性能基准 CI [D1] | **benchmarks/inference 收录 DCNv2/DIN/ETA/MMOE 但均未接 torch.compile**(grep 无命中);torchbench harness 28 模型配后门(dvm 18/mlir 10) [S3] |

**核心判断**:搜广推是社区在 AOTI 区域化(Regional AOTI)与 RecSys kernel(TLX 博客)两个方向上**点名**的场景,而 PTA 侧搜广推模型反而还没有进入 torch.compile 的官方验证闭环——这是业务视角下最值得警觉的落差。

### 3.2 优先级矩阵(建议)

| 优先级 | 方向 | 理由(社区证据 → 业务价值) |
|---|---|---|
| **P0** | ① device 注入接入社区 opinfo/test_torchinductor 正确性矩阵;② fallback 清单按项数密度收敛(scatter/index_put/cumsum/一元超越函数优先);③ embedding/大规模 pointwise 融合深化 | 正确性基线是所有后续投入的地基;自建速度追不上 +35%/版本的分母膨胀,"接入社区用例"是唯一能继承整个矩阵的路径 [D1][D3];每收敛 1 算子恢复 ~4.7 项矩阵 [D1];embedding 是搜广推第一算子面 |
| **P1** | ④ 图运行时对齐:static launcher/user_streams/多流语义跟随 aclgraph;⑤ 通信-计算重叠(simple_overlap 对应能力补齐);⑥ 动态形状(@dynamic_spec 思路 + unbacked 场景) | 社区测试四域翻倍中 CUDAGraphs +87%、user_streams 71 方法单文件最大新增——运行时正确性是当前社区重点 [D3];搜广推训练多卡通信重叠直接对应 overlap 投入 [S7];变长序列是搜广推第一动态场景 |
| **P2** | ⑦ Regional AOTI 跟进(字节级跟踪 #199582,准备 torchrec/DLRM 型负载的增量编译方案);⑧ strict numerics 精度体系对齐;⑨ FlexAttention(序列特征 attention 化) | RFC 首要场景即搜广推(torchrec DLRM) [S13];精度硬看护决定异构硬件可信度 [D3];Flex 测试域 v2.14 翻倍(332→645),按 645 分母预估工作量 [D3] |
| **P3** | ⑩ TLX/Helion/cuTile/Gluon DSL 生态跟踪;⑪ 上游扩展点 PR 持续推进(#201537 清单) | DSL 生态尚未路由主流算子(Helion 2.14 注册不路由),保持跟踪即可;扩展点 PR 是降长期成本的杠杆,与短期业务收益并行 [S6][S9] |

### 3.3 重点投入方向(具体建议)

1. **把"上游测试树 + device 注入"从 2 个 inductor 文件扩到 P0 域**:PTA 已具备完整机制(artifact 下载 + `PYTORCH_TESTING_DEVICE_ONLY_FOR=privateuse1` + 白名单 + skip 列表),扩白名单是运营动作而非开发动作;以 `test_torchinductor.py`(913 方法)+ opinfo 矩阵(3,697 GPU 实跑项)为首攻目标,建立"分层通过率 + skip 列表单调递减"的量化看护 [S3][D1]。
2. **fallback 收敛与业务模型联动**:用 DCNv2/DIN/ETA/MMOE 等业务模型的 profile 反推高频 fallback 算子,按"恢复矩阵项数 × 业务频度"排序收敛;同时把 torch.compile 接入 benchmarks/inference 四模型,建立搜广推性能基线(社区对标物即 torchbench CI) [S3][D1]。
3. **图运行时投入与社区对表**:aclgraph 等价实现保持对上游 cudagraph 体系(Policy 插件/static launcher/user_streams)的语义跟踪,优先落地 #201537 中 cudagraph trees 去耦(#194598)等已提交 PR 的上游合入,减少双轨维护 [S6]。
4. **Regional AOTI 早卡位**:该 RFC 明确 "No changes to the AOTI C++ runtime" 且以 DLRM 稠密块为第一场景;PTA 已有 AOTI 基础(aoti_shim/MegaCache),在 RFC 阶段即参与(华为已有 #201537 的对话先例),比落地后再追成本低 [S13][S2]。
5. **建立社区信号跟踪机制**:半年 roadmap(dev-discuss #3300/#3415)+ inductor meetup 周记 + release-feature-request 标签 issue,按版本节奏(每 2 月)滚动评估 Gap 表(§2.4),把"版本锁定基线"纳入规划(当前 v2.13 基线 4,883 应适配方法) [S15][S16][D3]。

---

## 附录:来源清单

### A. 代码仓(本地实测)

| 编号 | 来源 | 说明 |
|---|---|---|
| S1 | pytorch 仓 `/Users/huangshilei/Documents/cppprojects/pytorch`,main(2.16.0a0,HEAD `b48b4a49d28`,2026-10-21) | git 区间统计(v2.10.0=`449b176841`, v2.11.0=`70d99e998b`, v2.12.0=`0d62256a2b`, v2.13.0=`cf30153c4c1`, v2.14.0=`2b3ec34829`);目录对比 `git ls-tree`;特性识别 `git log --grep`;config 对比 `git diff v2.10.0 HEAD -- torch/_inductor/config.py`(239→314 项);后端集合见 HEAD `torch/_inductor/config.py:691-705` vs v2.10.0 `config.py:520-527` |
| S2 | PTA 仓 `/Users/huangshilei/Documents/pythonprojects/pta`,分支 v2.13.0-26.2.0(HEAD `57172760c9`,2026-10-20) | `torch_npu/_inductor/`(165 py/89,757 行;5 后端 `__init__.py:386-412`;fallback `lowering_fallback_list.py:65-920`;override `lowering_override_list.py:12-45`;patch 入口 `__init__.py:394-412`;CATLASS `codegen/catlass/`;fx_passes `ascend_custom_passes/ascend_graph_pass.py`);版本对应 `requirements_2.13.txt:4`、`torch_npu/_compat/version.py:10`、`README.md:160`、`COMPATIBILITY.md:27-38`;官方约束 `torch_npu/_inductor/docs/overview/overview.md:23-25`;torchair `third_party/torchair`(pinned `5722384b`)、`torch_npu/dynamo/__init__.py:49-58,260-281` |
| S3 | 同上 PTA 仓测试与 CI 资产 | `test/_inductor/`(26 文件/159 用例;git 引入提交 `7f7312eca8` "2025-09-20 [inductor] inductor support master");`test/dynamo/`(58 文件/1,681 用例,15 个 PTA 独有);`test/npu/`(aclgraph 95 + graph_tree 90);CI:`.gitcode/workflows/ut_job_inductor.yml:93-116`、`PR-pipeline_pytorch.yml:367-415`、`_test-collect.yml:49,209-221`、白名单 `.gitcode/config/pr_test_whitelist.yml`(126 文件)、skip `.gitcode/config/pr_skip_list.jsonl`(4,068 条);benchmark:`benchmarks/torchbench/npu_backend_config.json`(28 模型)、`benchmarks/inference/`(DCNv2/DIN/ETA/MMOE,torch.compile grep 无命中)、`benchmarks/llm/` |
| S4 | pytorch 仓 tag v2.13.0(`cf30153c4c1`,2026-07-08 发布) | release CI 全量 1,005 check-runs(见 D3 §6) |

### B. 本地已有分析文档(docs/ai_gen/,均含各自复现命令)

| 编号 | 文档 | 关键引用点 |
|---|---|---|
| D1 | `inductor_ci_test_coverage_analysis.md` | §5.1 分层(P0~P3 应适配 5,741 方法/23,111 用例);§5.2 NPU 基线(fork=0,自建 547);§5.3 opinfo 上界 62~66%/fp64 17~18%/每算子 4.7 项;§5.2.3 同代分母 13.8% |
| D2 | `inductor_fusion_breadth_analysis.md` | GPU 173 vs NPU 87 可融合(50.3%);惰性 prims 34;四层融合漏斗;模板 prologue(`mm.py:87-98` `prologue_loads_all_inputs=True`,CUTLASS 拒绝 prologue `cuda_combined_scheduling.py:172-173`) |
| D3 | `v2130_inductor_test_analysis.md`(本地分支 v2.13.0,commit `649479d6dbe`) | v2.13.0 176 文件/5,388 方法;+35%/版本;四域翻倍;main 缺口(flex_gemm 259、device_backends 系);v2.13 锁定基线 4,883;密度表(全量 10.2%/P0 5.8%);release CI 单点失败 |
| D4 | `torchinductor_test_inventory.md` | L4 实跑 23,749;opinfo 3,697;全量推算 2.87 万;fast/slow 分层 |
| D5 | `inductor_triton_codegen_extension_usage.md` | torchTLX `tlx_mode`(config.py:2060-2132);utlx 插件;`inductor_choices_class` 对称插件口子 |

### C. GitHub(pytorch/pytorch,均 2026-10-21 可访问)

| 编号 | 链接 | 日期/状态 |
|---|---|---|
| S6 | https://github.com/pytorch/pytorch/issues/201537 "[Discussion] TorchNPU: current state, gaps, and community contribution plan" | 2026-10-20 OPEN;列 PR #194598/#193770/#195263/#193355/#192513/#187649/#187938 |
| S11 | https://github.com/pytorch/pytorch/issues/188212 "[RFC] Add Gluon Backend To Inductor Backend" | 2026-06-26 OPEN |
| S12 | https://github.com/pytorch/pytorch/issues/175311 "[RFC][Inductor] PyTorch Inductor cuTile Backend" | 2026-02-19 OPEN |
| S13 | https://github.com/pytorch/pytorch/issues/199582 "[RFC] Regional AOTInductor" | 2026-10-02 OPEN;首要场景 torchrec DLRM |
| S17 | https://github.com/pytorch/pytorch/issues/193710(fp8 pattern matching 阻断);#187095(NVFP4 grouped);#201193(Jetson Thor sm_110);#194061(复数 tracking) | 2026-08~10 |
| S20 | https://github.com/pytorch/pytorch/issues/178917 "[Triton][Triton-Ext] Include TLX" | 2026-03-31 创建,2026-05-26 CLOSED/COMPLETED |

### D. 官方博客(pytorch.org/blog)

| 编号 | 链接 | 日期 |
|---|---|---|
| S7 | /pytorch-2-10-release-blog(2026-01-21);/pytorch-2-11-release-blog(2026-03-23);/pytorch-2-12-release-blog(2026-05-13);/pytorch-2-13-release-blog(2026-07-08);/pytorch-2-14-release-blog(2026-09) | 版本主线特性;2.10 博客含 "release cadence 1 per 2 months" |
| S5 | /triton-plugin-extensions-enabling-tlx-and-custom-compiler-passes-out-of-the-box | 2026-07-15;PyTorch-Triton 3.7 插件系统;TLX 默认启用承诺 |
| S8 | /warp-specialization-in-triton-design-and-roadmap | 2026-01-08;autoWS;megakernel 长期规划 |
| S19 | /in-kernel-broadcast-optimization-co-designing-kernels-for-recsys-inference | RecSys 生产 kernel(TLX);另有 /flexattention-flashattention-4-fast-and-flexible(2026-03-04) |

### E. dev-discuss 论坛(dev-discuss.pytorch.org)

| 编号 | 链接 | 日期 |
|---|---|---|
| S15 | /t/meta-pytorch-team-2026-h1-roadmaps/3300(含 Compiler/Triton roadmap PDF);/t/meta-pytorch-team-2026-h2-roadmaps/3415 | 2026-02-02 / 2026-07-20 |
| S16 | /t/inductor-meetup-notes-2025-10-13/2375 及 2025-09~11 周记系列 | autotune/locality/多流排期;Helion 顾虑;cuTile 讨论 |
| S14 | /t/ibm-spyre-accelerator-pytorch-enabling-status-and-feature-plan-1h-2026/3319(KTIR);/t/intel-gpu-cpu-enabling-status-and-feature-plan-2026-h1-update/3320(SYCL-TLA) | 2026-03 |

### F. 会议与其他

| 编号 | 来源 | 说明 |
|---|---|---|
| S18 | PyTorch Conference 2025 日程 https://pytorchconference.sched.com/(2025-10-22/23,San Francisco) | Inductor GEMM 生成(Meta)、Helion DSL(Jason Ansel)、CuTile/TileIR(NVIDIA)、FlexAttention 系列、Dynamic Shape Recompilations、LLM kernel 开发、mxfp8/mxfp4/nvfp4 等议题 |
| S9 | https://github.com/pytorch/helion | v1.5.1(2026-10-15);2.14 注册为第三 Native DSL(#190636) |
| S10 | pytorch 仓 `test/inductor/extension_backends/`(2.11 周期新增)、`test_device_backends.py`/`test_device_backend_hooks.py`(2.15 周期) | 后端扩展点测试 |

### G. 口径与局限说明

1. **测试口径**:上游静态方法数以 AST 口径为准(v2.13.0=5,388 [D3]);grep `def test_` 口径为 5,489 [S3],差异来自类外函数/辅助函数计数,不影响量级结论。
2. **PTA 自建用例数**:v2.13.0-26.2.0 分支 26 文件/159 用例 [S3] 与 master(2026-08)口径 126 文件/547 用例 [D1] 分属不同分支与测试组织方式(aclgraph/graph_tree 等价看护在 test/npu/),对比时应注明口径;**建议以 device 注入实跑后的真覆盖率取代投入密度作为长期指标** [D1]。
3. **融合覆盖率 50.3%、opinfo 上界 62~66%** 为 2026-08 master 口径的实测/推算,v2.13.0 分支未重测;方向性结论(约半数可融合面差距、fallback 收敛杠杆)不受版本代际影响 [D1][D2]。
4. **opinfo 可过上界是上界不是预测**:真值需 NPU 环境注入实跑校准(192.168.9.145 环境当时缺 CANN runtime/kernel 未完成) [D1]。
5. Meta roadmap PDF 本体托管 Google Drive,本次仅核实条目与链接,正文未解析 [S15]。
