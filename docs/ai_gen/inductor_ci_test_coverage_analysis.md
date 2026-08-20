# PyTorch 社区 Inductor CI 测试用例盘点与昇腾 NPU 适配可行性分析

> 面向技术评审:目标是让昇腾 NPU 接入 PyTorch 社区 inductor CI 用例看护,倒逼研发完成
> torch inductor 特性支持与覆盖。本文回答三个问题:
> ① GPU 上实际执行哪些测试、按特性怎么分布;② "100% 支持 CUDA 执行的用例"是否可行;
> ③ 若工作量太大,适配优先级怎么排。

数据口径:本仓 `main` 分支 `test/inductor/` 静态扫描(AST 统计 188 个测试文件的
测试类/测试方法/参数化条目)+ CI workflow 与 `test.sh` 实际执行链分析。
统计脚本与 TSV 见 `agent_space/count_inductor_tests.py`、`agent_space/group_inductor_tests.py`。

---

## TL;DR

1. 社区 inductor 测试规模:**188 个测试文件、6375 个静态测试方法、2872 条参数化条目**;
   运行时再经 device 实例化(`instantiate_device_type_tests`)、dtype 参数化、opinfo 算子库
   驱动展开,以及 **`--inductor` 模式重跑整个 torch 核心套件**(test_ops/test_torch),实际执行
   用例量级达数万。CUDA 上分 2 个 shard 跑(单 shard 数小时)。
2. **"100% 支持 CUDA 执行的用例"不可行,也不应作为需求提出**:
   - 约 9.4%(601 个方法/21 个文件)是厂商/后端专属(CUTLASS/Pallas/CuTeDSL/NV GEMM/FP8 特定路径/
     MPS/XPU/Halide),在 NPU 上语义不存在,属"不应做"而非"做不到";
   - 数值/精度/非确定性用例(fp64、tf32、atomic 顺序)在异构硬件上社区本身允许差异;
   - 社区范式佐证:ROCm/XPU/MPS 均非 100%,通过 device-specific skip/xfail 列表管理,从未有
     任何后端追求 100% CUDA 用例通过。
3. 合理需求口径是**"设备无关用例集通过率"**;建议按 P0~P3 分层接入,P0 = 算子级正确性基线
   (test_torchinductor + opinfo),与融合广度分析(NPU 覆盖率 50.3%、Scan 类 0/5)形成
   "广度+正确性"双看护。实测(L4 CI,§5.1):**应适配 5,741 个静态方法(90.5%),对应 TD 裁剪后
   23,111 项 GPU 用例;P0+P1 即覆盖 GPU 用例的 55.4%**,P3 与"不做集"(601 方法)明确排除。

---

## 1. 社区 Inductor CI 执行链(GPU)

CI 入口与测试集的映射关系(本仓 `.github/workflows/` + `.ci/pytorch/test.sh`):

| Workflow | Job(config) | 硬件/规模 | 执行入口 | 跑什么 |
|---|---|---|---|---|
| `inductor-unittest.yml` | `inductor` ×2 shards | A10G(g5.4xlarge),CUDA 13.0 与 13.2 双版本 | `test_inductor_shard()` | 见下分解 |
| 同上 | `inductor_distributed` ×1 | 12xlarge 多卡 | `test_inductor_distributed()` | aot_inductor 多卡/流/设备用例 |
| 同上 | `inductor_cpp_wrapper` ×2 | A10G | `test_inductor_cpp_wrapper_shard()` | C++ wrapper codegen 全量 |
| 同上 | `inductor_amx` ×7 / `triton-cpu` / `halide` | CPU(AMX) | 各自入口 | CPU 后端看护(非本议题) |
| `inductor.yml` | `inductor_huggingface/timm/torchbench` | A10G | 性能基准 | 模型级性能回归(非单测) |
| `trunk.yml` | nightly 全量 | — | 同 pull 链 | 夜间全量重跑 |
| 专用 workflow | ROCm mi200/300/350、b200、xpu 等 | 各自硬件 | 专用脚本 | 后端专属看护 |

`test_inductor_shard()`(GPU 单测主入口)分两层:

```
① 全套件 inductor 模式重跑(--inductor 强制所有模型经 inductor 编译):
   test_modules, test_ops, test_ops_gradients, test_torch   ← torch 核心套件整体再加一层
② inductor 单元测试(分片):
   test_torchinductor, test_torchinductor_opinfo,
   test_aot_inductor, test_cpu_select_algorithm
```

即社区对 GPU inductor 的看护是**三层叠加**:inductor 自身单测 + opinfo 全算子矩阵 +
torch 核心套件 inductor 重跑。

---

## 2. 特性 × 用例数 × 测试文件总表

静态口径:188 个测试文件全量归组(闭合校验 6375 方法 = 100%)。
"参数化条目"指 `@parametrize` 列表元素数(运行时展开倍数的一部分);
opinfo 类文件的实际用例由 op 数据库在运行时展开(见 §3)。

| # | 特性域 | 文件数 | 静态方法数 | 占比 | 主要测试文件(方法数) | CI 看护方式 |
|---|---|---|---|---|---|---|
| 1 | 算子级正确性基线 | 17 | 1263 | 19.8% | test_torchinductor(975)、test_torchinductor_opinfo(harness)、test_torchinductor_strided_blocks(48)、test_indexing(85)、test_foreach(46)、test_padding(26)、test_strict_numerics(13)、test_deterministic(9) 等 | inductor shard ①+② 全量 |
| 2 | 融合与调度 | 25 | 558 | 8.7% | test_loop_ordering(90)、test_nested_reduction(92)、test_combo_kernels(91)、test_mix_order_reduction(34)、test_group_batch_fusion(35)、test_inductor_scheduler(23)、test_scatter_optimization(22)、test_multi_kernel(14)、test_segmented_tree(12) 等 | inductor shard ② |
| 3 | FlexAttention 家族 | 5 | 604 | 9.5% | test_flex_gemm(242)、test_flex_attention(220)、test_flex_flash(79)、test_flex_decoding(56) | 专用 job(test_python_smoke/B200)+shard |
| 4 | AOTI 与 C++ 封装 | 17 | 436 | 6.8% | test_aot_inductor(289)、test_gpu_cpp_wrapper(24)、test_aot_inductor_package(24)、test_wrapper_codegen(16)、test_compile_to_python(18) 等 | inductor shard ② + cpp_wrapper shard |
| 5 | 缓存与并行编译 | 13 | 429 | 6.7% | test_codecache(142)、test_caching(101)、test_compile_worker(47)、test_async_compile(42)、test_compile(26) 等 | inductor shard ② |
| 6 | Autotune 与 GEMM 模板(通用) | 23 | 404 | 6.3% | test_max_autotune(137)、test_select_algorithm(46)、test_custom_op_autotune(21)、test_lookup_table(22)、test_pad_mm(21)、test_mmdecomp(14)、test_coordinate_descent_tuner(14) 等 | 专用 job(smoke/shard) |
| 7 | CUDA Graphs 与运行时 | 6 | 341 | 5.3% | test_cudagraph_trees(211)、test_user_streams(82)、test_static_triton_launcher(36)、test_cudacodecache(6) | inductor shard ② |
| 8 | Autograd 与训练 | 10 | 325 | 5.1% | test_compiled_autograd(132)、test_auto_functionalize(45)、test_control_flow(63)、test_inductor_freezing(24)、test_compiled_optimizers(13) 等 | inductor shard ② |
| 9 | Triton 代码生成 | 8 | 268 | 4.2% | test_triton_kernels(150)、test_triton_heuristics(57)、test_codegen_triton(32)、test_triton_helpers(20) | inductor shard ② |
| 10 | 动态形状 | 4 | 123 | 1.9% | test_torchinductor_dynamic_shapes(64)、test_unbacked_symints(53)、test_torchinductor_codegen_dynamic_shapes(5) | inductor shard ② |
| 11 | 分布式 | 6 | 50 | 0.8% | test_distributed_patterns(20)、test_symm_mem_registry(19) 等 | inductor_distributed 专用 job |
| 12 | 调试与工具链 | 33 | 973 | 15.3% | test_cpu_repro(288)、test_cuda_repro(110)、test_ordered_set(163)、test_pattern_matcher(99)、test_utils(48)、test_perf(73) 等 | shard ② + 按需 |
| 13 | 厂商/后端专属 | 21 | 601 | 9.4% | CUDA 系:Pallas(119)、CUTLASS 三件套(91)、NV GEMM(49)、FP8(42)、CuTeDSL(26+24+2)、flydsl(13)、Blackwell autotune(11);其他后端:fxir(66)、CPU select(64)、mkldnn(36)、MPS(37)、CK(25)、XPU(4)、Halide(5) 等 | 各自专用 job/自跳过 |
| — | **合计** | **188** | **6375** | 100% | 参数化条目另计 2872 条 | — |

要点:

- **正确性基线是最大头**(19.8%),其中 test_torchinductor.py 单文件 975 个方法,覆盖
  dtype/stride/广播/非线性/reduction/自动求导等编译正确性;
- **FlexAttention 家族已占 9.5%**,反映社区重心向 LLM 注意力倾斜(flex_flash/flex_decoding
  依赖 TMA/warp-specialization 的部分为 Hopper/Blackwell 分支);
- **厂商专属 9.4%** 中约 351 个方法是 NVIDIA 库/硬件绑定(CUTLASS/Pallas/CuTeDSL/NVGEMM),
  属结构性不可移植,与 NPU 能力无关。

---

## 3. 静态数 vs 实际执行用例数(为什么"6375"远低估实际量级)

运行时展开的三个放大器:

1. **opinfo 算子库驱动**:test_torchinductor_opinfo 基于
   `torch.testing._internal.common_methods_invocations` 的 op_db(本仓单文件直接定义 405 个
   OpInfo,拼接 custom/hop 等 db 后社区总量约 700+),每个 OpInfo 再乘 dtypes、编译模式
   (eager 对照/inductor)、内存格式 → 单文件展开即数千用例;
2. **device 实例化**:34 个文件(约 1700 个方法)使用 `instantiate_device_type_tests`,
   同一模板类生成 CPU/CUDA 两套用例;
3. **`--inductor` 模式重跑**:test_ops/test_torch/test_modules(静态 43+462+24 个方法,自身即
   opinfo 驱动)在 inductor 编译模式下整体再跑一遍。

因此评估工作量时应按"**静态方法数 × 参数化/实例化倍数 + opinfo 矩阵**"理解,实际执行
用例量级为**数万**,这正是社区用 2 shard 分卡的原因。本文用静态口径做相对比较(占比、
优先级),量级结论不受口径影响。

---

## 4. "100% 支持 CUDA 执行的用例"是否可行

**结论:不可行,且不应作为需求提出。** 四类论据:

### 4.1 结构性不可移植(约 5.5% 静态方法,"不应做")

CUTLASS 后端(91)、Pallas(119)、CuTeDSL/FlyDSL(65)、NV universal GEMM(49)、
Blackwell max-autotune(11)、FP8 特定路径(42) —— 这些用例验证的是**NVIDIA 库/硬件
特性的代码生成正确性**,被测对象(CUTLASS 模板、TMA、warp specialization、MXFP8)在
昇腾上不存在等价物。NPU 对应的是自有模板路径(等价物应自建用例看护,而非跑 CUDA 用例)。

### 4.2 硬件语义差异(可部分支持,需逐例判定,约再加 5~10%)

- fp64:部分用例依赖 Triton CUDA fp64 精度行为,昇腾 fp64 支持有限;
- 非确定性/atomic 累加顺序:test_deterministic 及散布的 strict tolerance 用例;
- CUDA Graphs:cudagraph_trees(211)可对标 NPU graph 能力,但 expandable_segments、
  流语义(CUDA stream ↔ ACL stream 映射)需专项适配后才知道可保留比例;
- user_streams(82)依赖多流语义对齐。

### 4.3 社区范式:没有任何后端追求 100%

社区标准做法是 `instantiate_device_type_tests` 框架 + device-specific skip/xfail 列表:

- 用例侧:`@onlyCUDA`/`@skipXPU`/`device_type` 装饰器显式声明设备边界;
- 后端侧:ROCm(`test/rocm_expecttest_inductor`、skip 列表)、XPU、MPS 各自维护期望差异,
  通过率从来不是 100%,也没有被要求 100%。

提"100% CUDA 用例通过"等于要求昇腾实现 CUTLASS/Pallas/NV 库的语义等价物,既不现实,
也不是社区评判后端质量的方式。**正确口径:非 onlyCUDA 用例集的通过率**(社区 HUD 对各
后端就是这样横向比较的)。

### 4.4 与融合广度分析的交叉印证

若 CI 全量通过,隐含前提是 op_db 约 700+ 算子在 NPU 上全部可编译且数值对齐。而当前实测
融合广度:NPU 可融 87 / GPU 173(覆盖率 50.3%),Scan 类 0/5、prod/var/any 缺失 —— 从
50.3% 到"全算子矩阵通过"之间存在数量级的差距,必须分阶段(见 §5)。

---

## 5. 适配优先级建议(倒逼研发的排序)

排序原则:先看护**正确性底线**(不能算错),再看护**性能收益来源**(融合深度/autotune),
最后是**生态与部署特性**;厂商专属集明确排除。

| 优先级 | 特性域 | 涉及测试(方法数) | 倒逼的研发工作 | 理由/验收口径 |
|---|---|---|---|---|
| **P0** | 算子级正确性基线 | test_torchinductor(975)、test_torchinductor_opinfo、strided_blocks(48)、indexing(85)、foreach(46) | 收敛 `NPU_EXTRA_FALLBACK_LIST`(592→0 的可融部分):补 Scan 5 个全缺(cumsum/cumprod/cummax/cummin/logcumsumexp)、prod/var/any,及降采样出的 52 个 aten 缺口 | 算错是底线;opinfo 是社区算子级看护的标准载体,直接对标 GPU;与融合广度 50.3% 形成同一份缺口清单 |
| **P1** | 动态形状 | dynamic_shapes(64)、unbacked_symints(53)、codegen_dynamic_shapes(5) | symint guard、size 变化的重编译/缓存正确性 | LLM 变长序列/batch 抖动是 NPU 主场景,动态形状出错直接编译失败或静默算错 |
| **P1** | 融合与调度 | loop_ordering(90)、nested_reduction(92)、combo_kernels(91)、mix_order(34)、scheduler(23) 等 558 | 融合规则/调度对齐: Reduction 融合、多 kernel 合并、循环重排 | 这是 inductor 性能收益的核心来源(广度之上的"深度");挂掉说明 NPU 融合决策与 GPU 分叉 |
| **P2** | Autotune(通用部分) | max_autotune(137)、select_algorithm(46)、coordinate_descent(14) 等 404 | Triton GEMM tuning、候选模板选择在 NPU 上生效(排除 CUTLASS/NV 库子集) | GEMM 性能竞争力的看护;需要 NPU 侧 benchmarking 基建先就绪 |
| **P2** | CUDA Graphs/运行时 | cudagraph_trees(211)、user_streams(82) | NPU graph 捕获/回放与 inductor cudagraph 模式对齐、流语义映射 | 训练吞吐刚需;依赖昇腾 graph 能力,需先做能力摸底再定可保留用例比例 |
| **P2** | Autograd 与训练 | compiled_autograd(132)、auto_functionalize(45)、freezing(24) | 反图编译、functionalization、参数冻结 | 训练场景全覆盖的前提 |
| **P3** | FlexAttention 家族 | flex_attention(220)、flex_gemm(242)、flex_flash(79)、flex_decoding(56) | NPU 侧 flex 模板后端(自研,对标社区接口) | LLM 长上下文刚需但工程量大(需模板机制+调优),建议单独立项,先保 flex_attention 主路径 |
| **P3** | AOTI/C++ 封装 | aot_inductor(289) 等 436 | AOT 导出、C++ wrapper 在昇腾工具链的编译 | 部署场景,不阻塞训练主线 |
| **P3** | 缓存/并行编译、调试工具链 | codecache(142)、caching(101) 等 ~1400 | 编译缓存/远程缓存/异步编译稳定性 | 影响 CI 效率与研发效率,不影响正确性 |
| 专项 | Triton 代码生成 | codegen_triton(32)、triton_kernels(150)、triton_heuristics(57) 等 268 | Triton 前端语法/启发式/wrapper 在昇腾 Triton 方言上的兼容 | Triton 是 inductor 唯一内核语言,单列避免淹没在 P3 工具链里 |
| 专项 | 分布式(多卡) | distributed_patterns(20)、symm_mem_registry(19) 等 50 | 多卡/集合通信编译,依赖昇腾 HCCL 专项 | 单卡主线外,按多卡版本节奏单独接入 |
| **不做** | 厂商专属 | CUTLASS(91)、Pallas(119)、CuTeDSL(65)、NV GEMM(49)、Blackwell(11)、FP8 特定(42)、MPS/XPU/Halide 等 | — | 改为**昇腾等价自建用例**(自有 GEMM 模板、FP8 方案)看护,接口对齐社区 |

### 5.1 各优先级的实测用例数(L4 CUDA CI, run 32193788776, commit 165426143e)

静态方法数按 §2 的 13 域分组重算(当前 `inductor_test_counts.tsv`,184 个 `test_*.py` 共
6,342,与 §2 的 6,375 有快照差异);CUDA 实跑 = 该 run junit 中实际执行的用例数(含 skip,
经 TD 裁剪,是各层在 GPU 上的**下界**):

| 优先级 | 特性域(§2 编号) | 文件数 | 静态方法 | CUDA 实跑用例 | 其中 skip | 本 run TD 未覆盖文件 |
|---|---|---:|---:|---:|---:|---:|
| **P0** | 域1 | 17 | 1,263 | **6,427** | 831 | 11 |
| **P1** | 域10 动态形状 + 域2 融合调度 | 27 | 655 | **6,675** | 1,901 | 16 |
| **P2** | 域6 Autotune + 域7 Graphs + 域8 Autograd | 38 | 1,069 | **2,279** | 173 | 22 |
| **P3** | 域3 Flex + 域4 AOTI + 域5 缓存 + 域12 工具链 | 67 | 2,438 | **7,249** | 1,776 | 40 |
| 专项 | 域9 Triton 代码生成 + 域11 分布式 | 13 | 316 | 481 | 50 | 8 |
| 不做 | 域13 厂商专属 | 22 | 601 | 557 | 224 | 14 |
| **应适配合计(P0~P3+专项)** | — | **162** | **5,741** | **23,111** | 4,731 | 97 |
| 全量 | — | 184 | 6,342 | 23,668 | 4,955 | 109 |

三级解读:

- **合计应适配多少**:全量 6,342 个静态方法中,扣除"不做"的 601 个厂商专属方法,
  **应适配 5,741 个静态方法(90.5%)**;在本 run TD 裁剪后的 GPU 执行集上对应
  **23,111 个用例**(占 L4 目录内实跑 23,668 的 97.6%)。"不做"层的 557 项实跑即
  CUTLASS/Pallas 等在 NVIDIA 上正常执行的用例,NPU 无需对齐。
- **逐级累计(静态 / CUDA 实跑)**:P0 1,263 / 6,427 → +P1 1,918 / 13,102 →
  +P2 2,987 / 15,381 → +P3 5,425 / 22,630 → +专项 5,741 / 23,111。
  CUDA 口径下 **P0+P1 即覆盖 55.4%**,验证"先正确性底线、再融合收益"的排序。
- **数字"倒挂"说明**:P1 实跑(6,675)高于 P0 静态数、P3 实跑(7,249)最高,源于参数化
  展开——动态形状 5 个方法展开 1,411 项、compile_subprocess/opinfo 单文件展开数千项;
  **评估工作量应以静态方法数为纲,以 CUDA 实跑数为 GPU 看护口径**。
- TD 未覆盖的 97 个应适配文件属本 commit 影响面之外(多为小文件/工具链),全量执行时
  会补齐,分层占比基本不变。

---

## 6. 落地机制建议(怎么"接入社区 CI 用例")

1. **注入 device 而非 fork 用例**:利用社区 `instantiate_device_type_tests` 框架,以
   `device_type=npu` 收集用例(torch_npu 已具备该机制),用例本体零 fork,长期跟随社区演进;
2. **维护 npu skip/xfail 列表**(社区标准做法):skip 只允许两类理由——
   `onlyCUDA 等价标记` 与 `已知缺口(关联 issue)`,禁止无理由 skip,使列表长度本身成为
   缺口收敛的量化指标(列表单调递减 = 研发在收敛);
3. **看护率作为验收指标**:按 §5 分层设定通过率目标(P0 建议起步 ≥60%,6 个月 ≥95%),
   每次版本发布报告分层通过率与 skip 列表变化,向上汇报与向下倒逼均用同一张表;
4. **与融合广度分析联动**:CI 用例失败模式(opinfo fail/skip)与
   `NPU_EXTRA_FALLBACK_LIST` 缺口清单(见《inductor_fusion_breadth_analysis.md》)映射成
   同一份 P0 任务列表,避免两套口径两套优先级。

---

## 7. 结论

- GPU inductor CI 看护 = 188 文件/6375 静态方法(opinfo 与 `--inductor` 重跑后实际数万用例),
  分 13 个特性域,正确性基线(19.8%)、FlexAttention(9.5%)、厂商专属(9.4%)为三大块;
- "100% 支持 CUDA 执行"**不可行也不可提**:约 9.4% 厂商专属属结构性不可移植,另有 5~10%
  硬件语义敏感用例;社区对任何后端都不要求 100%,标准口径是"设备无关用例集通过率";
- 建议按 P0(算子正确性+fallback 收敛)→ P1(动态形状+融合调度)→ P2(autotune/graph/训练)
  → P3(flex/AOTI/工具链)分层接入,应适配合计 **5,741 个静态方法 / TD 裁剪后 23,111 项
  GPU 用例**(§5.1 实测);厂商专属(601 方法)改为昇腾等价自建用例;
- 落地机制:device 注入 + 受控 skip 列表 + 分层通过率汇报,与融合广度缺口清单合并成统一
  的 P0 任务列表,实现对研发的持续倒逼。

---

## 附:统计口径与复现

- 静态扫描:AST 解析 188 个 `test_*.py`,统计测试类/`test_` 方法/`@parametrize` 条目数;
  分组映射经闭合校验(各组方法数之和 = 6375);
- CI 执行链:`.github/workflows/inductor-unittest.yml`(CUDA 单测)、`inductor.yml`(性能基准)、
  `trunk.yml`(nightly)、`.ci/pytorch/test.sh` 的 `test_inductor_shard()` 等 7 个 inductor
  入口函数;
- op 数据库:`torch/testing/_internal/common_methods_invocations.py` 单文件 405 个 OpInfo;
- 脚本:`agent_space/count_inductor_tests.py`(统计)、`agent_space/group_inductor_tests.py`
  (分组与闭合校验)、`agent_space/priority_cases.py`(§5.1 优先级分层实测数);
- GPU 实跑数(§5.1):trunk CI run 32193788776(commit 165426143e,2026-08-18)的
  L4 junit artifact,按 skip 原因/优先级分组解析,见《torchinductor_test_inventory.md》第 7 节;
- 已知局限:静态方法数不含运行时展开(opinfo/dtype/device 实例化),用于相对比较与优先级
  排序,不用于绝对工作量核算;精确用例数需在可运行环境 `pytest --collect-only` 获取。
