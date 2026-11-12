# 昇腾 950(A5)上 TorchInductor 融合比例与融合性能低于 GPU 的根因分析

> 基线:torch_npu 2.10.0 仓 `torch_npu/_inductor/`(`is_ascend950` 分支,下称 A5 路径);
> 对照:上游 PyTorch GPU Triton 路径。
> 分析日期:2026-11-12。所有结论均标注代码出处(`文件:行号`,基于 pta 仓工作区状态);
> 涉及影响占比的表述为定性估计,定量方法见第 7 节。

---

## 0. 摘要

**融合"比例"(fused kernel 占比)与融合"性能"(融合 kernel 的实际加速)是两个不同的问题,根因集合不同。**

一句话结论:

- **比例差异的第一性根因在软件**:约 750-766 个 op 重载被 fallback 清单强制逐出融合池,每个 fallback 节点还强制上游物化、冻结 layout;GEMM epilogue 融合有两条通道——Triton 模板(triton CV 融合)与 catlass EVG(白名单仅 `{add, relu}`)——默认全部关闭。
- **即使把软件层全部补齐,A5 靠芯片仍有一段追不平的差距**,集中在五条硬件边界:UB 工作集上限(而非寄存器)、cube/vector 分离、无跨核规约合并、向量核无随机 gather/scatter、指令集缺口与大 kernel 上限。
- **性能差异**主要来自:融合 kernel 在 A5 上的并行度/驻留/流水三要素全部弱于 GPU 同构融合;且 A5 的 192MB L2 使"省 DRAM 往返"这一融合核心收益被稀释;默认配置下不做融合前 benchmark,决策质量依赖启发式。

归因矩阵(定性权重,基于代码证据覆盖面):

| 根因层 | 属性 | 对融合比例 | 对融合性能 |
|---|---|---|---|
| L1 fallback 覆盖面(约 750 op) | 纯软件 | ★★★★★(最大头) | ★★(少融合→多 launch/多 DRAM 往返) |
| L2 GEMM epilogue 缺位 | 软件 gating + 芯片分档 | ★★★★(transformer 大头) | ★★★(GEMM 周边重复访存) |
| L3 调度器保守化 patch | 软件(硬件诱发) | ★★★ | ★★ |
| H1 UB 工作集上限 | 硬件上限 | ★★(编译门拒绝/拆分) | ★★★★(tile 驻留小、循环深) |
| H2 cube/vector 分离 | 硬件架构 | ★★★(epilogue 表达受限) | ★★★(跨单元流水开销) |
| H3 无跨核规约合并 | 硬件缺失 | ★★ | ★★★★(规约并行度塌掉) |
| H4 向量核访存模型 | 硬件缺失(A5 部分缓解) | ★★ | ★★★ |
| H5 指令缺口 + 大 kernel 上限 | 硬件约束 | ★★ | ★★ |
| P1 L2 192MB 收益稀释 | 硬件特征 | —(不减次数) | ★★(改变融合净收益) |

---

## 1. 基线与方法

- **NPU 侧**:`torch_npu` 2.10.0,`TORCHINDUCTOR_NPU_BACKEND=default`(triton-ascend 路径,`__init__.py:358-381`);A5 判定 `is_ascend950 = get_soc_version() >= 260`(`config.py:32-36`)。
- **GPU 侧**:上游 Inductor Triton 路径(CUDA/A100-H100 量级为参照)。
- **关键架构前提**:NPU 的融合入口没有被替换——上游 `Scheduler.fuse_nodes → fuse_nodes_once → can_fuse` 主循环原样生效,NPU 通过 4 层 patch 接入(类方法 monkeypatch、`V.choices`、`NPUCombinedScheduling` 后端、可选 trace/DVM/MFusion 路径)。**因此"scheduler 判不判得出融合"不是差异来源;差异产生在节点进入融合池之前(fallback 逐出)与融合之后(编译可行性否决)。**

代码路径对照:

```
GPU:   FX图 → 分解(pointwise链) → Triton lowering(全覆盖) → Scheduler.can_fuse(同源) → Triton编译(几乎必然成功) → 大融合kernel
A5 :   FX图 → 分解(部分删除)   → lowering(766 fallback强制逐出) → Scheduler.can_fuse(同源+保守patch) → 编译可行性门(UB/线性化/spill) → kernel(或拆分/回退)
```

---

## 2. 全景根因链:节点从 FX 图到 kernel 的六道关卡

1. **分解关**:GPU 上 gelu/LN/softmax/erf 等被分解成 pointwise 链进融合池;A5 上 `decomposition.py:178-200` 删除了一批分解(gelu 默认不分解、nll_loss、log_softmax_backward、repeat_interleave 等),op 以整体形态存在,失去可融合性。
2. **lowering 关(fallback 逐出,最大头)**:`lowering_fallback_list.py` 约 750-766 个 op 重载强制转 aclnn 单算子,头注释明说 "will not be allowed to lowering and fusion"。
3. **融合判定关**:与 GPU 同源(非差异来源),叠加 NPU 保守 patch(见 4.3)。
4. **编译可行性关**:UB 溢出(triton-ascend 编译器硬检查)、索引线性化失败、寄存器 spill——任一命中即拒绝融合或拆分已融合组。
5. **模板/epilogue 关**:GEMM 模板固定 tile 后追加 epilogue 可能超 UB,编译验证不过即拒绝(`npu_template_compile_reject` 计数器)。
6. **调度距离关**:A5 长距离水平融合阈值 64→20。

---

## 3. 软件维度

### 3.1 融合池入口:fallback 覆盖面(比例差异最大头)

**清单规模**(`lowering_fallback_list.py`,AST 统计):

| 列表 | 位置 | 条数(重载) | 生效条件 |
|---|---|---|---|
| `NPU_EXTRA_FALLBACK_LIST` | :47-460 | 409 | 默认 |
| `TORCH_NATIVE_FALLBACK_LIST` | :462-799 | 336 | 默认 |
| `INDIRECT_MEM_FALLBACK_LIST` | :803-827 | 23 | 仅 `inductor_indirect_memory_mode` 为空(非 A5 默认);**A5 默认 `simd_simt_mix` 不追加** |

默认合计约 750(A5)/约 766(非 A5)个 op 重载。**其中大量是 GPU 上原生 Triton lower、正常参与融合的算子**:

- 三角/超越函数族(asin/atan2/cosh/expm1/erfc/erfinv/lgamma/fmod/remainder/signbit/square 等)——GPU 在 `torch/_inductor/lowering.py:8654-8932` 用 `register_pointwise_numeric` 全部原生 lower;
- in-place 族(add_/mul_/relu_/sigmoid_ 等,:77-79/:240-242/:331-343);
- scan 族(cummax/cummin/cumprod/logcumsumexp——GPU 的 SCAN 融合);
- pooling(avg_pool3d/max_pool2d_with_indices 及 backward);
- **view/expand 族(aten.view/view.dtype/_unsafe_view/expand_as 等,:191-224/:348-350)**;
- index/scatter 全家桶(:214-290);RNG(rand/randn/randperm——GPU 用 Philox 融合进 kernel);量化 quantized_decomposed(:448-455);基础 prims(:383-445)。

**三重伤害机制**:

1. fallback 节点 = `ExternKernelSchedulerNode`,**不可融合**;调度到它时上游强制 `flush()`,前后融合组不能跨越(上游 `scheduler.py:12686-12692`)。
2. `npu_make_fallback`(`lowering.py:98-119`)**无条件 `add_needs_realized_inputs`**(:103)→ 所有上游生产者强制物化(`FallbackKernel.create → realize_input`,上游 `ir.py:7549`)、layout 冻结、多用户生产者零拷贝内联失效(上游 `graph.py:2222-2228` `realize_hint`)。
3. **view 也 fallback 是最隐蔽的一条**:GPU 上 view 是零成本 IR,pw→view→pw 在 A5 上被切成 2 个 Triton kernel + 1 次 aclnn 调用;transformer reshape 密集场景受系统性伤害。

**NPU 自有覆盖仅 19 个 op**(`lowering_override_list.py:12-31`:mean/max/min/amin/amax/argmax/argmin/var_mean/var/cat/fractional_max_pool3d/pow/prims.convert_element_type/mm/bmm/addmm/flex_attention(+bwd)),且关键锚点缺位:

- **cat**:非 A5 一律 fallback(`lowering.py:1190-1191`);A5 上 ≤8 输入走 `ConcatKernel`——它本身仍是 ExternKernel(独立 copy kernel),仅 `force_pointwise_cat` 时走可融合形态(:1200-1201)。
- **native_layer_norm**:`lowering.py:1284-1294`——A5 + fp16/bf16 且 `(not enable_welford or LayerNormV4 条件)` 时直接 fallback;`enable_welford` 默认关(`config.py:185`)→ **A5 默认配置下 LN 是一个 aclnn fallback 算子,与周围 pointwise 完全不融合**(LN 是 GPU transformer 融合的锚点)。
- **gelu**:默认不分解(`decomposition.py:220-232` 仅 `TORCHINDUCTOR_ENABLE_FAST_GELU=1` 时注册 tanh 分解)。
- Welford/SIMT 模板、间接访存模板(`INDIRECT_MEM_OVERRIDE_LIST` 11 op)仅 `simt_template` 模式生效,A5 默认 `simd_simt_mix` 不启用。

**分解反向操作**(`decomposition.py`):删除 PT 分解使 gelu(默认)/nll_loss/log_softmax_backward/repeat_interleave/max_pool2d_with_indices(A5)/slice_backward(非 A5)保持整体算子;GPU 上它们分解为 mul/exp/sigmoid/reduction 小算子全部融入大 kernel。MLIR/DVM 路径再排除 30 个分解(`ascend_npu_ir/.../config.py:295-325`)。

**归约能力弱于 GPU**:

- backend_features 缺 `REDUCE_TO_SINGLE_ELEMENT`(`codegen/scheduling.py:180-190`)→ 单元素归约走多层拆分 + 中间物化(`ir.py:84-133` `num_splits` patch,GPU 上单个 persistent reduction 完事);
- `mix_order_reduction=False`、`coalesce_tiling_analysis=False`(`config.py:163-164`)关闭 GPU 默认的乱序归约融合与合并访存分析;
- scatter/mutation 节点(`MutationLayoutSHOULDREMOVE` + 强制 `self.realize()`,`lowering.py:737/:883`)形成原地写屏障。

**布局约束插拷贝**:`require_contiguous` 在 bmm(`kernel/bmm.py:1769-1797`)、flex_attention(`kernel/flex_attention.py:1764-1769/:2898`)、searchsorted(`lowering.py:338-341`)处对非连续输入插入独立 copy kernel,打断可融合链。

### 3.2 GEMM epilogue 融合:两条通道并存,默认全部关闭

NPU 的 GEMM 模板 epilogue 有两条通道:

**通道一:Triton 模板(triton CV 融合)**。`kernel/mm.py:78-90` 注释明示 "(for CV / epilogue fusion)":三族 NPU 自研 Triton GEMM 模板——`npu_triton_mm_template`(标准分块)、`npu_persistent_mm_template`(persistent 对角分核)、`npu_resident_b_mm_template`(B 驻留 L0B;mm.py:2193/:2215/:2248 注入 choices)——以 `acc += tl.dot(...)`(:159 等 6 处)表达 GEMM,tile 由 `optimize_tiling` 按 UB 预算生成、program 数封顶 `NUM_CUBE_CORES`(mm.py:1256-1261)。scheduler 检测 mm 输出被 pointwise 消费时,把消费链渲染进模板的 `{{store_output}}` 位置(:297 "inductor generates epilogue suffix");addmm bias 经 `addmm_epilogue`(prefix_args=1,mm.py:1343-1397),broadcast 负载(bias[N])另有去 broadcast 优化(select_algorithm.py:1061-1065)。triton-ascend 把 `tl.dot` 编译为 cube 单元 MMA(L0A/L0B→L0C),pointwise 编译为 vector 单元——**同一 kernel 内 cube+vector 混合执行,中间结果留 L0/UB 不出 HBM**,这就是 "CV(Cube/Vector)融合" 的含义。门控 `use_triton_template`(`utils.py:69-91`):`max_autotune` + TRITON autotune backend + TRITON_TEMPLATES feature——同样漏查 `max_autotune_gemm`(与 catlass 同一门控缺陷)。接受任意可 lowering 的 pointwise 链(无白名单),但追加 epilogue 放大 UB 工作集,受 3.3 #1 编译可行性门约束。

**通道二:catlass EVG**,注册于 mm/addmm/bmm/grouped_mm(`kernel/mm.py:2171/:2473`、`kernel/bmm.py:1832`、`kernel/mm_grouped.py:238`)。

**启用需要同时满足(全部默认不满足)**:

| 条件 | 出处 | 默认 |
|---|---|---|
| `TORCHINDUCTOR_MAX_AUTOTUNE=1` | docs/feature/catlass | 0 |
| `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS` 含 `CATLASS` | 同上,默认 "ATEN,TRITON,CPP" | 不含 |
| `CATLASS_EPILOGUE_FUSION=1` | `config.py:225-227` | 0 |
| catlass 库可导入(`TORCHINDUCTOR_NPU_CATLASS_DIR`) | `config.py:206-213` | 需安装 |

SOC 限制:catlass 仅 Ascend 950DT/950PR(docs/feature/catlass);Triton 模板通道无 SOC 门槛(模板参数考虑 910B CUBE 硬件限制,mm.py:1509-1511),**但两条通道默认都关**。非 A5 `disable_addmm_fusion = not is_ascend950`(`dvm/config.py:31`)连 mm+add(bias) 都拆开;`fx_passes/post_grad.py:8-29` 主动移除社区 mm+mm 融合 pattern。

**即使全部打开,catlass 通道表达面仍远窄于 GPU**:

- fast/on-fly 融合(`codegen/catlass/gemm_template.py:840-898`):算子白名单**仅 `{add, relu}`**(:857),最多一次(:860),**仅 1 个 epilogue 节点**(:842-844),bias 须 1 维连续(:869-874),且模板需 `is_support_relu`(:510)。
- EVG(Epilogue Visitor Graph)融合(`catlass_python_evg.py:59-88`):支持 mul/truediv/add/sub/relu/sigmoid/cast/constant;`ge`/`tanh` 等显式 `NotImplementedError`;`fusion_type!=2` 的模板不支持 bf16(`catlass_scheduling.py:360-368`)。
- 调度侧(`catlass_scheduling.py:265-369`):仅 Pointwise `ComputedBuffer`;**禁链式 epilogue**(:287-289 `assert len(existing_epilogue_nodes)==0`);禁 reduction(:326-328);禁 aliasing/mutation(:323);**水平融合恒 False**(:102-111)。

**对照 GPU**:GPU Triton 模板 epilogue 同样接受任意 pointwise 链(任意 tl 表达式)且支持 prologue fusion;NPU 的 Triton 模板通道表达面与 GPU 同级,**差距不在表达面而在三点:(a) 默认关(max_autotune)——(b) UB 门可能拒绝 epilogue 扩展——(c) catlass 侧白名单窄**。手工兜底(fused_matmul_relu/grouped_matmul/grad-transpose/multi-slice-cat 四个 FX pass、mfusion MLIR)全部默认关、严格 pattern、逐个验证。

### 3.3 调度器保守化 patch(默认生效的主干)

| # | patch | 出处 | 上游行为 | A5 行为 |
|---|---|---|---|---|
| 1 | 固定 tile 模板编译可行性门 | `scheduler.py:406-431/:834-876` | `benchmark_fusion=False` 时模板 epilogue 无条件融合 | 追加 epilogue 后**完整编译一遍**,UB/编译不过即拒绝(计数器 `npu_template_compile_reject`) |
| 2 | 长距离水平融合阈值 | `scheduler.py:223-235`(仅 A5) | proximity > 64 拒绝 | **> 20 拒绝**("avoiding runtime errors in large kernels") |
| 3 | 模板融合白名单收窄 | `codegen/scheduling.py:1148-1157` | 任意 template 直接放行 | 仅 `TritonTemplateBuffer`;CATLASS/dkdv/DVM 走专用窄通道 |
| 4 | composite dK/dV 模板 | `codegen/scheduling.py:1088-1094` | (上游无此类) | 完全禁止融合 |
| 5 | spill 判死阈值 | `codegen/scheduling.py:983-987` | benchmark 中 n_spills > 8 判 inf | **n_spills > 0 即 inf** |
| 6 | nolinear 重分组二次校验 | `codegen/npu_combined_scheduling.py:214-235` | (上游无) | 拆不开即静默判 inf |

激进的改动(Welford 旁路 `choices.py:67-94`、FlexAttention deferred epilogue 只验编译 `scheduler.py:683-712`、atomic_add 放行 :447-448)均为窄场景。

**caveat(版本漂移)**:本分析的上游参考为 2.16.0a0,torch_npu 基于 2.10——个别"更保守"项(reduction-reduction 判定简化、`fuse_nodes_once` 缺 template-overlap 仲裁/峰值内存 guard)可能是 2.10 基线差异而非 NPU 有意收紧,定责前需在 2.10 checkout 复核。上表 1/2/3/5/6 为 torch_npu 明确新增,不受此影响。

### 3.4 默认关闭的融合相关能力清单

`TORCHINDUCTOR_MAX_AUTOTUNE`、`MAX_AUTOTUNE_GEMM_BACKENDS` 含 CATLASS、`CATLASS_EPILOGUE_FUSION`、`TORCHINDUCTOR_ENABLE_MFUSION`、`TORCHINDUCTOR_ENABLE_FUSED_MATMUL_RELU`、`TORCHINDUCTOR_ENABLE_MULTI_SLICE_CONCAT`、`TORCHINDUCTOR_ENABLE_GRAD_MATMUL_TRANSPOSE_OPT`、`TORCHINDUCTOR_ENABLE_GROUPED_MATMUL_FUSION`、`TORCHINDUCTOR_ENABLE_MATMUL_TRITON`、`PERMUTE_CONTINOUS_REDUCTION`、`TORCHINDUCTOR_ENABLE_WELFORD`、`INDUCTOR_DVM_ENABLE_MATMUL_FUSION`、`TORCHINDUCTOR_BENCHMARK_FUSION`、`INDUCTOR_ASCEND_ENABLE_COSTMODEL`。**开箱即用时:GEMM epilogue 融合 0 条路径生效,手工 matmul 融合 0 个 pass 生效,MLIR 子图融合关闭。**

---

## 4. 硬件维度(A5 芯片边界)

### 4.1 H1:融合 kernel 的工作集上限从"寄存器"变成"UB"

这是最底层的硬件根因,其余约束多由它派生。

- **GPU**:逐元素融合的中间值生在**寄存器**(每 SM 256KB 寄存器文件),融合只增加寄存器压力→影响 occupancy 但不挡编译;shared memory 只承担显式 staging。融合的边际成本≈几条指令。
- **A5**:vector 单元数据驻留以 **UB(256KB/核,`config.py:38-40`)为中心**。这份预算同时要容纳:multibuffer 槽位(`buf_slot_num_of_veccore=3`/`crosscore=2`)、workspace(`set_workspace_multibuffer=4`)、DMA staging;tile 预算再按输入指针数均摊(A5 封顶 3 个指针)并除以 dtype 字节(`tile_generator.py:137-142`:`max_numel_threshold = local_mem_size // input_ptr_num // dtype_bytes`,`stop_numel` 再 //8)。SIMT 路径更紧:标量 local memory 128KB、单 tile 驻留 > 8KB 直接拒绝配置(`tile_generator.py:140/:316-318`)。
- **编译器硬检查**:triton-ascend 报 `ub overflow requires X bits while Y bits available`(`triton_heuristics.py:1031-1048` 正则捕获)→ 一次缩 tile 重试(预算 ×available×4÷required×5,留 20% 余量,:1089-1140)→ 全部失败则 `NoTritonConfigsError` 整体回退。编译成功后记录 `required_ub_bits` 供成本模型(:1246-1273)。
- **对融合的作用方式**:垂直融合加深 → 中间值驻留 UB 增多 → 溢出 → 缩 tile(循环加深,融合收益下降)或拒绝;**横向融合(combo)要求所有子 kernel 共享一份 tile,UB 算不出合法共享 tiling 直接放弃整个 combo**(`triton_heuristics.py:3404-3405`)。GEMM 模板固定 tile 后追加 epilogue 同理(软件层的编译可行性门就是这一硬件边界的显式化,`scheduler.py:406-416` 注释自述)。
- **GPU 上不存在对应环节**:融合决策做出后编译几乎必然成功,没有"融合前先证明编译得过"的门槛。

**判别式:凡是"融合决策已做出、编译阶段被否"的,根因落在本条。**

### 4.2 H2:Cube/Vector 分离,GEMM epilogue 跨单元边界

- **GPU**:GEMM 与 epilogue 是同一条指令流、同一个 SM——Triton 里 MMA 写完接着写 pointwise,表达平凡;epilogue 融合接受任意 pointwise 链。
- **A5**:GEMM 在 cube 单元(数据经 L0C),pointwise/reduction 在 vector 单元。epilogue 融合 = 一个 kernel 内调度 cube+vector 两级流水(mix 模式;`tile_mix_cube_loop`/`tile_mix_vector_loop`/`enable_dynamic_cv_pipeline` 等 compile hint 就是流水参数,`config.py:473-533`)。跨单元边界的主要软件补偿正是 3.2 通道一:triton CV 融合把 GEMM 与 pointwise 编进同一 kernel、由 triton-ascend 做 cube+vector 混合流水;catlass 侧则保守得多——白名单仅 `{add, relu}`、EVG 算子集封闭、禁链式,**每多支持一个算子都要人工验证流水正确性**。
- 核数模型也分离:`num_cube_core`/`num_vector_core` 从 device properties 读取且比例固定(`config.py:43-116`),分核限核(`NPU_DEVICE_LIMIT`)按两族分别约束。

**对 transformer 负载,这是影响最大的单条芯片因素(GEMM 密集,epilogue 是融合大头)。**

### 4.3 H3:无跨核规约合并

- **GPU**:split reduction 是常规操作——多 block 部分和 + 原子加/二次归约,softmax/LN 等规约锚点可铺满整机再融合。
- **A5**:`split_tiling.py:260-278` 注释明说 "making a reduction axis a grid split axis gives grid>1 **with no cross-core combine (A5)**, which overwrites the scalar output"——**动态 shape 的 full reduction 只能 grid==1 单核**;静态场景靠多层拆分 + 中间 buffer(与 backend features 缺 `REDUCE_TO_SINGLE_ELEMENT` 叠加)。
- 后果:规约类 kernel 要么单核串行(并行度项塌掉、融合收益公式失效),要么拆成多 kernel+物化(融合组被物理切开)。

### 4.4 H4:向量核访存是块 DMA,无全随机 gather/scatter

- **GPU**:每线程任意寻址,index/embedding 类天然融进 pointwise。
- **A5**:vector 单元访存走块 DMA + UB staging(950 默认开 NDDMA 非连续块搬运,`config.py:364`);检测到间接寻址即降级 `SIMT_ONLY`(`codegen/ir.py:1708-1769/:1848`)或走 `__builtin_index_select` 等特殊指令(`codegen/triton.py:6334-6498`)。索引线性化失败(`can_split_all`,`ir.py:1562-1583`):A5 降级 SIMT(tile 上限缩到 8KB);910 直接抛错、把已融合节点重新分组拆 kernel(`npu_combined_scheduling.py:111-212`)。A5 的 `INDUCTOR_INDIRECT_MEMORY_MODE=simd_simt_mix`(默认)是缓解,模板覆盖仅 11 个间接访存 op 且需 `simt_template` 模式。
- 这是 embedding/gather 类在 NPU 上融合收益骤降的硬件来源(A5 上不再整类 fallback,与非 A5 的关键差异,但融合质量受限于 SIMT/mix 路径)。

### 4.5 H5:指令集缺口与大 kernel 规模上限

- fp32 vector 性能弱:upcast 约定强制 fp32 中间值(`codegen/scheduling.py:574-579`),fp32 tile 占双倍 UB;flex dK/dV 配置过滤要求 `3×BLOCK_N1×head_dim×4B ≤ ub_size×4/5`(`kernel/flex_attention.py:203-233`)。
- 指令不支持:fp64 DSL(`triton.py:298`)、`welford_combine`(`triton.py:5210`)、scan 进 masked(`triton.py:2218`)、split-scan 与 reduction 融合(`codegen/scheduling.py:1096-1101`)——GPU 上分解后可融的 LN/softmax 链在指令层走不通。
- 大 kernel 运行时错误:A5 上大融合 kernel 触发运行时报错,长距离融合阈值被迫从 64 收到 20(`scheduler.py:231`)——融合 kernel 规模有上限,GPU 无此失败模式。
- 旧芯片 int64 索引需补 cast(A5 已支持,`triton.py:707-708`)。

### 4.6 P1:存储层级与融合收益结构

A5 按 192MB L2 建模(`scheduler.py:149` 的 L2 flush 基准 buffer)。融合的核心收益是消除 DRAM 往返;L2 越大,kernel 间中间量越可能命中 L2,**同等融合的净收益被稀释**。这不减少融合次数,但:(a) 使"少融合"的代价变小;(b) 是 NPU 融合 benchmark 必须做 L2 flush 的原因(否则测不出真实差异,`scheduler.py:60-72` 注释详述);(c) 意味着融合比例低在 A5 上的性能代价低于同等情况下的 GPU。

---

## 5. 融合性能维度(比例之外)

### 5.1 同构融合 kernel 的效率差

融合成立后,同一融合组在 A5 上的实现质量受三要素钳制:

1. **并行度**:tile/切分搜索被核数钳制(`tile_generator.py:153/:218-238/:427-455` 的 program threshold/wave 量化 0.72 效率线,`split_tiling.py:228`);规约因 H3 单核;shape 不足以铺满核数时收益打折。
2. **驻留与流水**:H1 使每核驻留 tile 小→循环深、multibuffer 流水深度受 UB 限制(`fasta_autotune.py:1102` `ub_times`);线性化失败降级 SIMT(吞吐骤降)。
3. **跨单元协调**:含 GEMM 的融合走 CV 流水,调度与同步开销为 GPU 单指令流所无。

### 5.2 融合决策质量(成本模型差异)

- **默认不做融合前 benchmark**:上游与 A5 的 `benchmark_fusion` 默认均关;A5 在关闭 benchmark 时对固定 tile 模板 epilogue 仅做编译验证(3.3 #1)——即**大多数 pointwise/reduction 融合不实测收益,赌启发式正确**。
- benchmark 路径启用时,A5 用 profiler 批测 + L2 flush(`scheduler.py:37-219`,批次内 TOTAL_STEP=10,注释写 50 与实现不一致)替代事件计时;成本模型预过滤(`INDUCTOR_ASCEND_ENABLE_COSTMODEL`)默认关。
- 判死标准更严:spill>0 即 inf(上游 >8)、nolinear 拆不开静默 inf——**宁可错杀**,压比例也改变了"哪些融合活下来"的分布。

### 5.3 融合的反向风险

GPU 上融合几乎单调有益(省 launch + 省 DRAM);A5 上大融合 kernel 有运行时错误风险(H5)、UB 溢出重试后缩 tile 可能比拆开更慢、L2 大使收益本就薄——**这解释了为什么 A5 的融合策略系统性偏保守,以及保守并不总是亏**。

---

## 6. 归因矩阵(汇总)

| 机制 | 出处 | 属性 | 比例 | 性能 | 默认状态 |
|---|---|---|---|---|---|
| 约 750 op 强制 fallback + 强制物化 | `lowering_fallback_list.py`、`lowering.py:103` | 软件 | ★★★★★ | ★★ | 生效 |
| view/expand 逐出融合 | `lowering_fallback_list.py:191-224` | 软件 | ★★★★ | ★★ | 生效 |
| LN/cat/gelu 锚点缺位 | `lowering.py:1190/:1284`、`decomposition.py` | 软件 | ★★★★ | ★★★ | 生效(A5 LN 默认 fallback) |
| GEMM epilogue 双通道默认关(catlass 白名单 {add, relu};Triton 模板受 UB 门) | `kernel/mm.py:78-90`、`config.py:225-227`、`gemm_template.py:840-898` | 软件 gating(芯片分档) | ★★★★ | ★★★ | 默认关 |
| mm+mm 移除 / 非 A5 addmm 拆分 | `post_grad.py:8-29`、`dvm/config.py:31` | 软件 | ★★ | ★★ | 生效 |
| 编译可行性门(UB) | `scheduler.py:406-431` | 软件门 × 硬件因 | ★★★ | ★★ | 生效 |
| 长距离阈值 64→20 | `scheduler.py:223-235` | 软件阈值 × 硬件因 | ★★ | ★ | A5 生效 |
| spill>0 判死 / nolinear 静默 inf | `scheduling.py:983-987`、`npu_combined_scheduling.py:229-231` | 软件 | ★★ | ★★ | 生效 |
| UB 工作集上限 + 编译器硬检查 | `config.py:38-40`、`triton_heuristics.py:1031-1140` | 硬件 | ★★ | ★★★★ | 物理边界 |
| 无跨核规约合并 | `split_tiling.py:260-278` | 硬件 | ★★ | ★★★★ | 物理边界 |
| cube/vector 分离 | CV 流水 hint、catlass 白名单 | 硬件 | ★★★ | ★★★ | 架构 |
| 无随机 gather/scatter | `ir.py:1708-1769` | 硬件(A5 缓解) | ★★ | ★★★ | 物理边界 |
| 指令缺口(fp64/welford/scan) | `triton.py:298/:2218/:5210` | 硬件 | ★★ | ★★ | 物理边界 |
| 大 kernel 运行时上限 | `scheduler.py:231` | 硬件 | ★★ | ★ | 物理边界 |
| L2 192MB 收益稀释 | `scheduler.py:149` | 硬件特征 | — | ★★ | 物理特征 |

---

## 7. 定量验证方法论(三桶归因法)

在 GPU 与 A5(950)上跑同一模型,把"未融合节点"按三类归因,①:②③ 的量级就是软件:硬件的直接度量:

1. **桶① 不在融合池**(纯软件):统计 fallback/extern 节点占比——wrapper 输出中 aclnn 调用数 / 总节点数;对照 `NPU_INDUCTOR_FALLBACK_LIST=allfallback` 的边界情形。
2. **桶② scheduler 拒绝**(软件 patch 为主):`TORCH_LOGS="+inductor"` 收集 `WhyNoFuse` 原因分布("Nodes are too far away"、"not TritonTemplateBuffer"、spill 等)。
3. **桶③ 编译拒绝**(硬件诱发):`counters["inductor"]["npu_template_compile_reject"]`、ub overflow 日志、`NoTritonConfigsError`、combo 放弃日志、nolinear 拆分计数。

补充实验矩阵(每项只动一个开关,观察融合组数与 kernel 数):开 `CATLASS_EPILOGUE_FUSION`+max-autotune(GEMM epilogue 贡献)、开 `TORCHINDUCTOR_ENABLE_WELFORD`(LN 锚点)、`ENABLE_INPLACE_BUFFERS=0`(启用 NPU-IR multi-buffer)、从 fallback 清单移除 view 族(view 障碍贡献)、`TORCHINDUCTOR_BENCHMARK_FUSION=1`(决策质量贡献)。

---

## 8. 优化方向(按预期 ROI 排序,均属软件层)

1. **补 lowering 覆盖**:把 fallback 清单中 GPU 证明可融合的纯 pointwise 族(超越函数、in-place、prims)还给 Triton lowering——每移除一批,融合池直接扩大;**优先恢复 view 零成本语义**(对 transformer 收益最大)。
2. **打通 LN/Welford 锚点**:A5 上默认启用 `TORCHINDUCTOR_ENABLE_WELFORD` 并验证 LayerNormV4 fallback 条件,让 LN 重新进融合池。
3. **cat 走 pointwise_cat**:A5 上 `force_pointwise_cat`(≤8 输入)替代 ConcatKernel extern。
4. **放宽确定性判死**:spill 阈值对齐上游(8);nolinear 拆不开时降级 benchmark 而非静默 inf。
5. **打通 GEMM epilogue 双通道**:修 `use_triton_template`/`use_catlass_template` 漏查 `max_autotune_gemm` 的门控缺陷;开 max-autotune 后实测 Triton CV 通道的 UB 拒绝率(3.3 #1 门),若拒绝率高则优化 epilogue 的 tile 预算策略。
6. **默认开启无风险的编译可行性门结果缓存**:模板 epilogue 编译验证结果按 kernel 签名缓存,减少重复编译开销,让 gating 更敢开。
7. **硬件补偿改进**(吃掉 H 层差距):UB 重试策略多轮化、combo 共享 tiling 搜索扩到多 dispatch 轴、间接访存 simt_template 模板扩面、动态规约双 kernel + 部分和方案(模拟 GPU split reduction)。

---

## 附:与 GPU 的结构性差异一句话总结

GPU 的 Inductor 融合是"**宽容判定 + 便宜编译 + 寄存器预算**":能融就融,编译几乎不失败,中间值住寄存器。
A5 的 Inductor 融合是"**同源判定 + 逐出式覆盖面 + UB 预算 + 跨单元表达受限**":大量节点根本进不了池,进了池的要在 UB 里证明放得下,GEMM epilogue 还要跨 cube/vector 边界——三层差距中,软件层(覆盖面与开关)目前贡献大头,硬件层(UB/规约并行/跨单元)构成追不平的底。
