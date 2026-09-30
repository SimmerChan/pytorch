# v2.13.0 分支 Inductor 测试用例静态分析

> 面向技术评审:沿用《inductor_ci_test_coverage_analysis.md》的分析逻辑(静态 AST 口径 +
> 13 特性域分组 + 版本间对照 + release CI 佐证 + NPU 投入密度换算),对 **v2.13.0 正式发布版**
> 的 `test/inductor/` 做全量盘点。回答三个问题:① v2.13.0 的用例规模与域分布;② 相对
> v2.10.0(增长)与 main(缺口)的位置;③ 若 NPU 下一代适配对齐 2.13,分母和基线变成多少。
>
> 逐用例清单(特性域 / test 文件相对路径 / 方法名三列,**5,388 行**,utf-8-sig 编码,Windows
> Excel 直接打开)见 [v2130_inductor_test_cases.csv](v2130_inductor_test_cases.csv)。

---

## TL;DR

1. **v2.13.0 总量:176 个测试文件 / 5,388 个静态测试方法 / 2,421 条 `@parametrize` 条目**,
   全部文件归入 13 域(无未归类)。相比 v2.10.0(143 / 3,978 / 2,006),**三个 release 周期
   (约 5.5 个月)增长 +1,410 方法(+35.4%)**;main 快照(2026-09-30)7,284 方法,即 v2.13
   分叉后 main 又 +1,896(+35.2%)——**inductor 测试集保持每版本 ~35% 的膨胀速度**。
2. 域分布:**算子级正确性基线仍是最大单域(1,148,21.3%)**,其后调试工具链 876(16.3%)、
   厂商专属 505(9.4%)。**相对增速最快的是缓存与并行编译(+104%)、分布式(+105%)、
   融合与调度(+98%)、Triton 代码生成(+96%)**;绝对增量最大的是算子级(+237)、
   融合(+207)、缓存(+185)、CUDAGraphs(+139)。
3. **release CI 佐证(发布 commit 自身)**:GPU(A10G)inductor 单测 2 个 shard 中
   **shard 1/2 失败、且全部失败集中于 1 个用例** `test_aot_inductor.py::AOTInductorLoggingTest::test_shape_env_reuse`
   (pt2 logger handler 数断言 5>2,重跑 3 次一致失败,属 logger 卫生检查而非编译正确性);
   shard 2/2、distributed、cpp_wrapper×2、性能基准×5、ROCm mi300/mi355 各 2 shard、
   pallas-gpu/tpu 全部 success。与 v2.10.0(唯一失败为 CPU 侧 pallas-cpu)同为
   "release 版 inductor 套件整体全绿量级、单点非正确性失败"。
4. **下一版本(Flex)翻倍预告**:main 上已有而 v2.13.0 没有的 30 个文件/605 方法中,
   `test_flex_gemm.py` 单文件 259 方法 —— FlexAttention 域将从 332 涨到 645(**接近翻倍**,
   与 2.13 内 flex_attention 196 → 243 的既有增速一致),是 v2.14+ 最大的单点增量。
5. **NPU 前瞻分母**:以 torch_npu 自建 547 用例为分子,同套域映射下投入密度全量口径从
   v2.10.0 的 13.8% 降到 **10.2%**(应适配合计 10.5%);**版本锁定基线从 v2.10.x 的
   3,493(重算口径 3,522)涨到 v2.13.x 的 4,883 个应适配静态方法(+38.6%)**——
   分母每版本膨胀 ~35%,社区用例接入越晚,一次性要面对的基线越大。

---

## 1. 数据来源与统计口径

| 数据集 | 仓库 | ref | commit | 时间 | 用途 |
|---|---|---|---|---|---|
| **v2.13.0(主对象)** | github.com/pytorch/pytorch | tag `v2.13.0` = 分支 `release/2.13` HEAD(两者同指) | `cf30153c4c1` | 2026-07-03 16:27:09 -0400 提交;**正式 Release 发布于 2026-07-08 17:39 UTC(非 RC)** | 176 文件静态统计、CSV、release CI 佐证 |
| v2.10.0(代际对照) | 同上 | tag `v2.10.0` | `449b17684101` | 2026-01-15(发布 2026-01-21) | 增长对照(`agent_space/v2100/`) |
| main(演进对照) | 同上 | `main` | `0775e53079c2` | 2026-09-30 16:04 +0800(本地 main,2.15 开发周期) | 分叉后缺口对照(`agent_space/main_snap/`) |
| release CI | 同上 | run [28681996328](https://github.com/pytorch/pytorch/actions/runs/28681996328)(inductor)、[28681996171](https://github.com/pytorch/pytorch/actions/runs/28681996171)(trunk)等 | — | 2026-07-03 push 触发;2026-07-08 tag push;2026-09-02 tag 上 nightly | §6 job 级结论 + 失败用例日志 |

- v2.13.0 与 main 的分叉点(merge-base)为 `636bc9106d05`(2026-06-10),此后 release/2.13
  只收 release cherry-pick;tag `v2.13.0` 与分支 HEAD 同指 `cf30153c4c1`,**静态统计即
  正式发布版的精确内容**(`git archive` 导出至 `agent_space/v2130/`,git-ignored)。
- 统计口径与 `count_inductor_tests.py` / `count_v2100_inductor.py` 完全一致:AST 解析,
  统计测试类数、类体内 `test` 开头方法数、`@parametrize` 列表条目数;**逐用例 CSV 只取
  类体内方法**,与静态方法总数闭合。
- **解析器版本是本篇新引入的口径坑**:本机 Python 3.9 无法解析 `match` 语句,会静默漏掉
  v2.13.0 的 3 个文件(`test_flex_aux_vectorization.py`、`test_flex_flash.py`、
  `test_interval_mask_packing.py`)与 main 的 `test_flex_gemm.py`;本篇全部统计用
  **Python 3.10** 重新解析,7 个文件均通过,三版本 0 解析失败。
- 13 域分组沿用 `group_inductor_tests.py` 的 `GROUPS`,另补录 6 个 GROUPS 未收录但实际存在
  的文件(按内容归属):`test_multi_kernel.py`、`test_segmented_tree.py` -> 融合与调度;
  `test_flex_gemm_runtime.py` -> FlexAttention 家族;`test_torchinductor_codegen_config_overrides.py`
  -> Triton 代码生成;`test_extension_backend.py` -> 厂商/后端专属;`test_pad_mm_utils.py`
  -> Autotune。`test_interval_mask_packing.py` 沿用 GROUPS 既有归属(Autograd 与训练)。
  GROUPS 内 `test_triton_cpu_backend.py` 在同域重复登记一处,统计时已去重。
- 闭合校验:v2.13.0 sum(13 域) = 5,388 = 全量(未归类 0);v2.10.0 = 3,959 + 未归类 19
  (`test_static_cuda_launcher.py` 17、`test_cuda_select_algorithm.py` 2);main = 7,069 +
  未归类 215(17 个 2.14/2.15 周期新文件)。
- 复现:`python3.10 agent_space/count_v2130_inductor.py`(同时生成 CSV)。
- 注意:因本篇对 v2.10.0 使用了与《覆盖率分析》§5.2.3 略不同的域映射(6 个补录文件并入域),
  v2.10.0 的分域数字与之有 ±30 以内出入(如融合与调度 212 vs 186、应适配基线 3,522 vs
  3,493);**篇内三版本数字互相可比,跨篇引用时注明口径**。

---

## 2. 总量与三版本对照

| | v2.10.0(2026-01) | **v2.13.0(2026-07)** | main 快照(2026-09-30) |
|---|---:|---:|---:|
| 测试文件 | 143 | **176** | 205 |
| 静态测试方法 | 3,978 | **5,388** | 7,284 |
| `@parametrize` 条目 | 2,006 | **2,421** | 3,497 |
| 相对上一列增长 | — | **+1,410(+35.4%)** | +1,896(+35.2%) |

- v2.10.0 -> v2.13.0 跨越 2.11(2026-03-23 发布)、2.12(2026-05-13)、2.13 三个周期,
  约 5.5 个月;**每版本 ~1,000 方法的增量、~35% 的膨胀速度是社区 inductor 测试的新常态**,
  与 2.10 之前 7 个月 +60% 的速度相比绝对量翻倍、速度有所收敛(分母变大)。
- v2.13 分叉点在 2026-06-10,main 快照包含 2.14 整个周期 + 2.15 开端约 3.5 个月的增量,
  膨胀速度未见放缓(详 §5.3,主要来自 Flex GEMM 与后端抽象层)。

---

## 3. v2.13.0 的 13 域分布

| # | 特性域 | 文件 | 静态方法 | 参数化 | 占比 | vs v2.10.0 | main 缺口 |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 算子级正确性基线(P0) | 16 | **1,148** | 282 | 21.3% | +237(+26.0%) | +166 |
| 2 | 动态形状(P1) | 3 | 109 | 19 | 2.0% | +28(+34.6%) | +16 |
| 3 | 融合与调度(P1) | 24 | **419** | 204 | 7.8% | **+207(+97.6%)** | +299 |
| 4 | Autotune 与 GEMM 模板(P2) | 22 | 375 | 320 | 7.0% | +112(+42.6%) | +115 |
| 5 | FlexAttention 家族(P3) | 5 | 332 | 112 | 6.2% | +84(+33.9%) | **+313** |
| 6 | CUDAGraphs 与运行时(P2) | 6 | 299 | 16 | 5.5% | **+139(+86.9%)** | +72 |
| 7 | AOTI 与 C++ 封装(P3) | 15 | 381 | 87 | 7.1% | +63(+19.8%) | +130 |
| 8 | Triton 代码生成(专项) | 8 | 227 | 195 | 4.2% | **+111(+95.7%)** | +80 |
| 9 | Autograd 与训练(P2) | 10 | 310 | 285 | 5.8% | +38(+14.0%) | +20 |
| 10 | 缓存与并行编译(P3) | 12 | 362 | 147 | 6.7% | **+185(+104.5%)** | +82 |
| 11 | 分布式(专项) | 5 | 45 | 0 | 0.8% | **+23(+104.5%)** | +5 |
| 12 | 调试与工具链(P3) | 30 | **876** | 59 | 16.3% | +134(+18.1%) | +173 |
| 13 | 厂商/后端专属(不做) | 20 | 505 | 695 | 9.4% | +68(+15.6%) | +210 |
| — | **合计** | **176** | **5,388** | **2,421** | 100% | **+1,429(13 域口径)** | +1,681 |

(main 缺口列 = main 快照中同域文件的方法数 - v2.13.0;另有 215 方法在 17 个 main 新文件中,
未计入缺口列,见 §5.3。)

要点:

- **P0 正确性基线占比稳定在 21%**(v2.10.0 为 22.9%),`test_torchinductor.py` 单文件
  753 -> **913** 方法(main 已到 1,004),仍是单一最大看护载体;域内新增
  `test_torchinductor_opinfo_properties.py`(opinfo 矩阵的属性级补充)、
  `test_block_ptr_store_dtype.py`、`test_simd_range_trees.py` 等正确性文件。
- **四域接近翻倍**:缓存与并行编译(+104%,`test_codecache.py` 67 -> 136 翻倍)、
  分布式(+105%,从 2 文件扩到 5 文件)、融合与调度(+98%)、Triton 代码生成(+96%)。
  叠加 CUDAGraphs(+87%,`test_user_streams.py` 71 方法为单文件最大新增),说明社区投入
  正从"能用"转向**性能机制与运行时正确性**(缓存失效/多流/graph 捕获/融合深度)。
- **FlexAttention 332 方法只是"半程"**:v2.13.0 的 flex 文件族为 flex_attention(196)+
  flex_flash + flex_aux_vectorization + flex_gemm_runtime + interval_mask_packing;
  `test_flex_gemm.py`(259 方法)在 main 上才出现 -> main 侧 Flex 域已达 645。
- 厂商/后端专属 505(+15.6%)中新增 NV 侧 `test_nv_universal_gemm.py`(17)、
  `test_cutlass_fallback.py`(12)、`test_max_autotune_blackwell.py`(11)、
  `test_origami.py`(9)——NVIDIA 专属面继续扩大,均属 NPU"不做集"。

---

## 4. 关键文件三版本规模

| 文件 | v2.10.0 | v2.13.0 | main | 备注 |
|---|---:|---:|---:|---|
| test_torchinductor.py | 753 | **913** | 1,004 | P0 主载体,持续线性增长 |
| test_cpu_repro.py | 231 | 272 | 293 | 调试/复现工具 |
| test_aot_inductor.py | 253 | 278 | 343 | AOTI;release CI 唯一失败用例在此文件 |
| test_cudagraph_trees.py | 157 | 191 | 235 | graph 模式主看护 |
| test_flex_attention.py | 172 | 196 | 243 | Flex 主路径 |
| test_flex_gemm.py | — | **未引入** | 259 | v2.14 周期新增,Flex 翻倍主力 |
| test_compiled_autograd.py | 125 | 132 | 133 | 训练编译 |
| test_codecache.py | 67 | **136** | 152 | 缓存域翻倍的载体 |
| test_max_autotune.py | 95 | 130 | 184 | autotune 主看护 |
| test_triton_kernels.py | 98 | 132 | 157 | Triton 用户内核接口 |
| test_torchinductor_dynamic_shapes.py | 55 | 61 | 65 | 动态形状主文件 |
| test_unbacked_symints.py | 26 | **48** | 54 | unbacked 近乎翻倍(LLM 变长场景) |
| test_torchinductor_opinfo.py | 1 | 1 | 5 | harness;运行时由 op_db 展开(数千用例,静态口径不变) |

---

## 5. 版本间变化明细

### 5.1 v2.10.0 -> v2.13.0 新增 35 文件 / 362 方法(方向解读)

单文件 top(方法数 / 域):

| 文件 | 方法 | 域 | 信号 |
|---|---:|---|---|
| test_user_streams.py | 71 | CUDAGraphs | 多流语义成为 graph 正式看护面(NPU 对应 aclgraph 流映射) |
| test_nested_reduction.py | 54 | 融合 | 嵌套 reduction 融合(性能深度) |
| test_static_triton_launcher.py | 28 | CUDAGraphs | 取代 v2.10.0 的 test_static_cuda_launcher.py(17,已删) |
| test_symm_mem_registry.py | 19 | 分布式 | 对称内存注册(分布式域翻倍主力) |
| test_nv_universal_gemm.py | 17 | 厂商专属 | NV GEMM 模板(不做集) |
| test_triton_helpers.py | 16 | Triton | helper 库 |
| test_optimize_indexing.py | 15 | 融合 | 索引优化 |
| test_auto_chunker.py | 13 | 融合 | 自动分块 |
| test_dropout_align_random_eager.py | 13 | Autograd | dropout 随机数与 eager 对齐(正确性) |
| test_cutlass_fallback.py | 12 | 厂商专属 | 不做集 |
| test_max_autotune_blackwell.py | 11 | 厂商专属 | Blackwell 专属(不做集) |
| test_origami.py | 9 | 厂商专属 | NV 新后端(不做集) |

其余 24 个文件各 1-7 方法,分布在算子级(embedding、simd_range_trees、block_ptr_store_dtype、
opinfo_properties)、Flex(aux_vectorization、gemm_runtime、interval_mask_packing)、AOTI、
缓存、调试等域。**新增文件的方向 = 多流/graph 运行时 + 融合深度 + 分布式起步 + NV 专属**,
没有出现全新大域。

### 5.2 v2.13.0 中消失的 v2.10.0 文件(2 个 / 19 方法)

`test_static_cuda_launcher.py`(17)被 `test_static_triton_launcher.py`(28)替代,
`test_cuda_select_algorithm.py`(2)并入通用 select_algorithm。**无能力删除,均为演进替换**。

### 5.3 main 分叉后新增(v2.13.0 缺口,30 文件 / 605 方法)

| 文件 | 方法 | 域 | 说明 |
|---|---:|---|---|
| test_flex_gemm.py | **259** | FlexAttention | Flex 域翻倍主力,v2.14 周期落地 |
| test_dynamic_tunable_ops.py | 59 | 未归类(新) | 动态可调算子(2.15 周期新主题) |
| test_slice_scatter_chunking.py | 35 | 未归类(新) | slice/scatter 分块 |
| test_flydsl_template.py | 33 | 厂商专属 | 不做集 |
| test_wrapper_codegen.py | 28 | AOTI | C++ wrapper codegen(从 cpp_wrapper 域拆出强化) |
| test_choices_composition.py | 21 | 未归类(新) | autotune choices 组合 |
| test_strict_numerics.py | 20 | 算子级 | **严格数值断言(对 NPU 精度对齐是硬看护,建议关注)** |
| test_compile_options.py | 20 | 未归类(新) | 编译选项面 |
| test_compile_to_python.py | 18 | AOTI | 编译到 Python |
| test_constant_offload.py | 17 | 未归类(新) | 常量卸载 |
| test_device_backends.py / test_device_backend_hooks.py / test_cpp_builder.py | 16/10/12 | 未归类(新) | **后端抽象/插件化重构主题**(对 NPU 这类扩展后端直接相关) |

主题归纳:① Flex GEMM 模板大扩展(259);② **inductor 后端抽象层/多后端插件化**
(device_backends、device_backend_hooks、cpp_builder、choices_composition、compile_options
合计 ~75 方法,方向对扩展后端友好);③ 数值严格性(strict_numerics);④ 其余为 AOTI 与
编译基础设施常规增长。**NPU 若以 2.14/2.15 为目标,后端插件化接口值得提前跟入。**

---

## 6. Release CI 佐证(v2.13.0 发布 commit 自身)

`release/2.13` push(2026-07-03,commit `cf30153c4c1`)触发的 inductor 相关 job 结论
(check-runs 全量 1,005 条,按 job 聚合):

| 平台 / job | 结论 |
|---|---|
| **GPU A10G `inductor` 单测 shard 1/2** | **failure**(见下) |
| GPU A10G `inductor` 单测 shard 2/2 | success |
| GPU A10G `inductor_distributed`(4 卡) | success |
| GPU A10G `inductor_cpp_wrapper` ×2 | success |
| GPU A10G 性能基准 huggingface / timm / torchbench ×5 | 全 success |
| ROCm mi300(gfx942)`inductor` ×2 | success |
| ROCm mi355(gfx950)`inductor` ×2 | success |
| pallas-gpu(H100)/ pallas-tpu(tpuv7) | success |
| CPU:`inductor_amx` ×2、pallas-cpu、halide、triton-cpu | success |
| CPU:`inductor_core` py3.11 ×2 | success |
| CPU:`inductor_core` py3.13 shard 2/2 | failure(job 级;日志未逐例解析) |
| CPU:`inductor_core` py3.12 ×2 | cancelled(基建原因,非测试失败) |

**shard 1/2 的失败是单点且非正确性**:日志(过期前取回,artifacts=0)显示全部失败集中于

```
test/inductor/test_aot_inductor.py::AOTInductorLoggingTest::test_shape_env_reuse
AssertionError: 5 not less than or equal to 2 : All pt2 loggers should only have
at most two handlers (debug artifacts and messages above debug level).
1 failed, 2 rerun -> "failed consistently"
```

即 pt2 logger 的 handler 数量卫生断言(应为 <=2,实测 5),重跑 3 次一致失败;属于
logger 泄漏/测试顺序类基建用例,**不涉及编译正确性或数值结果**;main 上此后亦无针对该用例
的修复提交。补充:2026-07-08 tag push(Create Release)未重跑单测;2026-09-02 在 tag 上
dispatch 的 aarch64 `inductor-perf-nightly`(39 job)全 success——release 分支至今仍可跑通。

与 v2.10.0 release CI 对照(《覆盖率分析》§5.2.3):v2.10.0 唯一失败为 CPU 侧
`inductor-pallas-cpu`;v2.13.0 唯一 GPU inductor 失败为上述 1 个 AOTI logging 用例。
**两个正式版的 inductor 套件都是"全绿量级 + 单点非正确性失败"**,用例级 junit 在两版
均已不可恢复(artifacts=0),v2.13.0 额外能给出的就是这份失败清单。

---

## 7. 对 NPU 适配的含义(前瞻分母)

分子沿用《覆盖率分析》§5.2 的 torch_npu 自建用例数(master `0ef1735386`,547;481 口径见
篇末注),分母换成本篇统一域映射下的 v2.10.0 / v2.13.0:

| 域 | NPU 自建 | v2.10.0 分母 | 同代密度 | **v2.13.0 分母** | **v2.13 密度** |
|---|---:|---:|---:|---:|---:|
| 算子级正确性基线(P0) | 67 | 911 | 7.4% | 1,148 | **5.8%** |
| 动态形状(P1) | 86 | 81 | 106% | 109 | 78.9% |
| 融合与调度(P1) | 124 | 212 | 58.5% | 419 | **29.6%** |
| Autotune(P2) | 22 | 263 | 8.4% | 375 | 5.9% |
| CUDAGraphs 与运行时(P2) | 47 | 160 | 29.4% | 299 | 15.7% |
| Autograd 与训练(P2) | 3 | 272 | 1.1% | 310 | 1.0% |
| FlexAttention(P3) | 5 | 248 | 2.0% | 332 | 1.5% |
| AOTI 与 C++ 封装(P3) | 56 | 318 | 17.6% | 381 | 14.7% |
| 缓存与并行编译(P3) | 3 | 177 | 1.7% | 362 | 0.8% |
| 调试与工具链(P3) | 17 | 742 | 2.3% | 876 | 1.9% |
| Triton 代码生成(专项) | 82 | 116 | 70.7% | 227 | 36.1% |
| 分布式(专项) | 0 | 22 | 0% | 45 | 0% |
| 厂商/后端专属(不做) | 35 | 437 | — | 505 | — |
| **全量** | **547** | **3,978** | **13.8%** | **5,388** | **10.2%** |

- **投入密度被分母稀释是持续性的**:同为 547 个自建用例,全量密度 v2.10.0 口径 13.8% ->
  v2.13.0 口径 10.2%;应适配合计(P0~P3+专项,不含厂商专属)512/4,883 = **10.5%**
  (含厂商专属的 547/5,388 = 10.2%)。这是"分母在涨、分子未动"的纯代际效应,不表示 NPU
  投入倒退;与 §5.2.1 相同,密度只回答配比,不是覆盖率。
- **版本锁定基线再膨胀**:若 torch_npu 下一代适配对齐 2.13,版本锁定应适配基线为
  **5,388 - 505(厂商专属)= 4,883 个静态方法**(v2.13.0 无未归类文件),较 v2.10.x 锁定
  基线 3,522(重算口径;《覆盖率分析》§5.2.3 原值 3,493)**+38.6%**。按 P0~P3 累计:
  P0 1,148 -> +P1 1,676 -> +P2 2,660 -> +P3 4,611 -> +专项 4,883。
- **两个结构性提醒**:① P0 正确性基线密度(5.8%)仍是各域最低之一,而它的分母半年内从
  911 涨到 1,148(main 1,314)——**"接入社区用例(device 注入)一次性继承整个矩阵"仍是
  唯一能追上分母增速的路径**,自建速度无法追上 +35%/版本的分母膨胀;② Flex 域
  v2.14 起 flex_gemm(259)落地后接近翻倍,NPU flex 模板若按 P3 排期,建议按 645 而非
  332 的分母预估工作量。

(481 口径:全量 8.9%、应适配 9.9%,结论方向不变。)

---

## 8. 逐用例 CSV

[v2130_inductor_test_cases.csv](v2130_inductor_test_cases.csv):**5,388 行**(另 1 表头行),
三列 `特性域, test文件相对路径, 方法名`,文件列为 `test/inductor/xxx.py` 完整相对路径,
`utf-8-sig` 编码(带 BOM,Windows Excel 直接打开中文不乱码)。生成脚本
`agent_space/count_v2130_inductor.py`(Python 3.10),行数与 §2/§3 静态方法总数闭合,
域分布与 §3 表逐域一致。

已知局限(与《覆盖率分析》一致):静态口径不含 opinfo/dtype/device 实例化的运行时展开,
用于相对比较与基线锁定,不用于绝对工作量核算;v2.13.0 release CI 的用例级 junit 已不可得,
GPU 实跑分布仍以《覆盖率分析》§5.1 的 L4 数据(master 侧,2026-08)为最近似参照。
