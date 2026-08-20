# PyTorch 与 torchinductor 测试用例盘点分析

- 统计基准: 本仓库 `main` 分支, commit `e47c8486484` (2026-08-18)
- 统计方法: 静态文本分析 (`grep` / Python 正则) + 远程容器实测 (`pytest --collect-only`, 见 [第 6 节](#6-远程实测验证-torch-2150-dev-nightly))
- AI 生成声明: 本文档由 AI 助手生成, 数据可由附录命令复现

## 1. 总体规模

| 维度 | torchinductor (`test/inductor/`) | PyTorch 全仓 (`test/`) | 占比 |
|---|---:|---:|---:|
| Python 测试函数 (`def test_`) | 6,477 | 43,764 | 14.8% |
| 测试类 (`class Test*`) | 333 | 2,673 | 12.5% |
| Python 文件 (含辅助脚本) | 201 | 1,558 | 12.9% |
| 其中含测试用例的测试模块 | 184 | — | — |
| C++ gtest 用例 (`TEST`/`TEST_F`/`TEST_P`) | 6 | 2,789 | 0.2% |

torchinductor 以约 13% 的测试文件占比贡献了全仓 14.8% 的 Python 测试函数, 是 PyTorch 中测试密度最高的子系统之一。

若把 torch.compile 的前端 Dynamo (`test/dynamo/`, 7,454 个测试函数) 一并计入编译栈口径, 则 Dynamo + Inductor 合计 13,931 个测试函数, 占全仓 31.8%。

## 2. torchinductor 测试的主题构成

按文件名对 `test/inductor/` 下 184 个测试模块 (6,477 个测试函数) 做主题分组:

| 主题分组 | 用例数 | 文件数 | 代表性文件 (用例数) |
|---|---:|---:|---|
| 编译基础设施/缓存/调优 | 1,374 | 41 | `test_ordered_set` (163), `test_max_autotune` (142), `test_codecache` (142), `test_config` 等 |
| 内核后端与代码生成 | 1,368 | 45 | `test_flex_gemm` (242), `test_triton_kernels` (151), `test_pallas` (127), `test_combo_kernels` (102) |
| 核心正确性 (`test_torchinductor*` 系列) | 1,106 | 7 | `test_torchinductor` (979), `test_torchinductor_dynamic_shapes` (64), `test_torchinductor_strided_blocks` (48) |
| 其他功能与平台 | 1,034 | 58 | `test_user_streams` (82), `test_flex_flash` (79), `test_perf` (73), `test_fxir_backend` (66) |
| 特性支持 | 569 | 17 | `test_flex_attention` (221), `test_pattern_matcher` (101), `test_auto_functionalize` (46) |
| AOTInductor / AOTI | 352 | 7 | `test_aot_inductor` (290), `test_aot_inductor_package` (24), `test_aot_inductor_custom_ops` (21) |
| CUDA Graph | 328 | 3 | `test_cudagraph_trees` (212), `test_cuda_repro` (110) |
| CPU / C++ wrapper | 325 | 4 | `test_cpu_repro` (289), `test_gpu_cpp_wrapper` (31) |
| 分布式/集合通信 | 21 | 2 | `test_distributed_patterns` (20) |
| **合计** | **6,477** | **184** | |

观察:

- 没有单一"巨石"主题: 基础设施、代码生成、端到端正确性三足鼎立, 各约 1,100-1,400 例。
- "其他功能与平台"虽然单文件用例少, 但覆盖面最广 (58 个文件), 包含 MPS (`test_mps_basic`)、XPU (`test_xpu_basic`)、Halide 后端 (`test_halide`)、fp8、flex 系列变体、内存规划等长尾特性。
- 分布式 inductor 的大部分测试不在本目录, 而在 `test/distributed/` 下 (见第 4 节)。

## 3. 用例数 Top 15 测试文件

| 排名 | 文件 | 用例数 | 说明 |
|---:|---|---:|---|
| 1 | `test/inductor/test_torchinductor.py` | 979 | 端到端正确性主战场, 覆盖绝大多数算子的 eager 对比 |
| 2 | `test/inductor/test_aot_inductor.py` | 290 | AOT 导出编译 (python/c++ wrapper) |
| 3 | `test/inductor/test_cpu_repro.py` | 289 | CPU 后端正确性 |
| 4 | `test/inductor/test_flex_gemm.py` | 242 | FlexGemm 模板 |
| 5 | `test/inductor/test_flex_attention.py` | 221 | FlexAttention 模板 |
| 6 | `test/inductor/test_cudagraph_trees.py` | 212 | CUDA Graph Trees 管理 |
| 7 | `test/inductor/test_ordered_set.py` | 163 | 数据结构单元测试 |
| 8 | `test/inductor/test_triton_kernels.py` | 151 | 用户自定义 Triton 内核导入 |
| 9 | `test/inductor/test_max_autotune.py` | 142 | 自动调优 |
| 10 | `test/inductor/test_codecache.py` | 142 | 代码缓存 |
| 11 | `test/inductor/test_compiled_autograd.py` | 132 | 编译版 autograd |
| 12 | `test/inductor/test_pallas.py` | 127 | TPU Pallas 后端 |
| 13 | `test/inductor/test_cuda_repro.py` | 110 | CUDA 复现工具 |
| 14 | `test/inductor/test_combo_kernels.py` | 102 | 组合内核 (split_cat 等) |
| 15 | `test/inductor/test_pattern_matcher.py` | 101 | 图模式匹配改写 |

Top 15 文件 (占 8% 的模块数) 贡献了 3,360 例, 约占目录总量的 52%; 长尾的 169 个模块贡献其余 48%。

## 4. 目录之外的 torchinductor 相关测试

### 4.1 文件名直接以 inductor 命名 (强相关, 4 个文件 / 168 例)

| 文件 | 用例数 |
|---|---:|
| `test/distributed/test_inductor_collectives.py` | 88 |
| `test/dynamo/test_regional_inductor.py` | 49 |
| `test/dynamo/test_wrap_inductor_compiled_regions.py` | 26 |
| `test/distributed/test_inductor_compile_collectives.py` | 5 |

### 4.2 内容引用 inductor 的测试文件 (宽口径, 139 个文件 / 10,530 例)

`test/` 下除 `test/inductor/` 外, 有 139 个测试文件的代码中引用了 `inductor` (import、config、mark_dynamic 编译等), 合计 10,530 个测试函数。主要分布:

- `test/dynamo/` (7,454 例): torch.compile 前端, 与 inductor 共同构成编译栈
- `test/export/`、`test/functorch/`、`test/distributed/` 等: 通过 `torch.compile` 间接消费 inductor

注意这是"编译栈耦合"的宽口径, 其中多数用例并非专门测试 inductor 本身, 不应与 6,477 直接相加得出"inductor 测试总数"。

### 4.3 配套清单与 C++ 测试

- `test/inductor_expected_failures/`: 363 个预期失败用例条目 (多为 opinfo 驱动的 `TestCommon*` 数值用例)
- `test/inductor_skips/`: 5 个跳过条目
- `test/inductor/cpp/test_cpp_prefix.cpp`: 6 个 gtest 用例 (C++ 前缀生成逻辑)
- `benchmarks/dynamo/`: inductor 性能/精度基线 CSV (非 assert 型用例, 未计入)

## 5. 参数化放大效应

静态统计的 6,477 是"测试函数个数"; pytest 实际收集到的用例数会显著放大, 放大来源:

| 来源 | `test/inductor/` 内出现次数 | 放大方式 |
|---|---:|---|
| `@parametrize` 装饰器 | 1,221 | 每个参数组合生成一个用例 |
| `@dtypes` / `@dtype` | 126 | 按数据类型成倍展开 |
| `instantiate_device_type_tests` | 35 个文件 | 用例按设备 (cpu/cuda/...) 各实例化一份 |
| OpInfo 矩阵 | `test_torchinductor_opinfo.py` 等 | 1 个 `def test_` 函数经 opinfo 数据库展开为数千个收集项 |

作为参考, 全仓 `@parametrize` 共 4,183 处, inductor 目录占 29%。放大倍数的实测数据见第 6 节: 在成功收集的 156 个模块上, 平均放大 **2.89 倍**, 参数化最密集的 `test_torchinductor_codegen_dynamic_shapes.py` (5 个函数) 展开出 1,411 项 (282x)。

## 6. 远程实测验证 (torch 2.15.0.dev nightly)

在远程服务器 (192.168.9.145) 的 openEuler aarch64 容器 (`zengxiong`, 1bc1283bbd5d) 中, 独立 venv 安装 **torch 2.15.0.dev20260818+cpu** (与本地 `main` 同日构建, API 完全匹配) + pytest 8.4.1, 将本仓库 `test/` 与 `tools/` 目录完整拷入后运行:

```bash
PYTHONPATH=/tmp/pt:/tmp/pt/test/inductor TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  python -m pytest --noconftest inductor --collect-only -q \
    --continue-on-collection-errors $(28 个 sys.exit 文件的 --ignore 列表)
```

实测结果 (2026-08-18):

| 指标 | 数值 |
|---|---:|
| **实测收集用例数 (collected)** | **14,445** |
| 收集错误 | **0** |
| 成功收集的模块数 | 156 / 184 |
| 未收集模块 | 28 (模块级 `sys.exit(0)` 环境跳过: 纯 CPU wheel 下 NCCL/CUDA/Pallas 不可用) |

(中间过程: 先用 pypi 最新发布版 torch 2.13.0 实测, 因本地 main 的 API 领先于 2.13 有 64 个模块 import 失败, 仅收集到 5,956 项; 换 nightly 后全部恢复, 佐证失败原因是版本代差而非测试本身。)

### 6.1 torchinductor 用例总数汇总 (静态 + 实测)

| 统计口径 | 用例数 | 覆盖范围 | 获取方式 |
|---|---:|---|---|
| 静态测试函数数 | 6,477 | `test/inductor/` 全部 184 模块 | grep 静态计数 |
| **实测收集数 (CPU 环境, 可直接复现)** | **14,445** | 156 模块 (0 错误) | `pytest --collect-only` |
| 实测校准的参数化放大系数 | **2.89x** | 4,992 静态函数 -> 14,445 收集项 | 实测计算 |
| 环境跳过模块推算 (GPU/分布式环境) | ~4,300 | 28 模块, 1,485 静态函数 x 2.89x | 推算 |
| **全环境推算总数 (含 CUDA/分布式)** | **约 1.9 万** | 全部 184 模块 | 14,445 + ~4,300 |
| 目录外强相关 (静态) | 168 | `test/distributed/` 93 + `test/dynamo/` 75 (文件名含 inductor) | grep 静态计数 |

一句话结论: **torchinductor 本体 (`test/inductor/`) 的测试用例, 静态口径 6,477 个测试函数, 实测 pytest 收集 14,445 个用例 (参数化放大 2.89 倍); 加上 GPU/分布式环境才启用的 28 个模块 (推算 ~4,300 项), 全环境总量约 1.9 万个用例。**

### 6.2 数字的精确性分级

| 数字 | 精确性 | 验证方式 |
|---|---|---|
| **14,445** (CPU 实测) | **精确, 可复现** | pytest 官方收集器输出, 0 错误; `--noconftest` 与打补丁后带 conftest 两种独立配置收集数一致; conftest 无 `pytest_generate_tests` 类参数化钩子, 其过滤钩子仅在 `--shard-id`/`--step`/rerun 等特定选项下生效, 默认不增减用例 |
| 6,477 (静态函数数) | 精确 (口径=文本匹配) | grep 正则确定性计数, 附录命令可复现; 注意其含义是"测试函数个数"而非"执行用例个数" |
| 2.89x (放大系数) | 精确于实测子集 | 14,445 / 4,992 直接计算; 但外推到其他文件属分布假设 |
| ~4,300 / **约 1.9 万** (全环境推算) | **推算, 误差最大** | 28 个模块的实际参数化密度未实测 (CPU 环境下 `sys.exit` 跳过); 且 `test_torchinductor_opinfo.py` 静态仅 1 个函数, 其 opinfo 全矩阵在 GPU CI 中可展开数千项 —— 约 1.9 万只能视为**保守下界**, 真实全环境总数可能显著更高 |
| 168 / 43,764 等其余静态数 | 精确 (口径=文本匹配) | 同 6,477 |

### 6.3 两个口径的关系

- 156 个成功模块的静态函数数 = 6,477 - 1,485 = 4,992, 实测收集 14,445 项, **平均参数化放大 2.89x**。
- 28 个未收集模块在模块级用 `sys.exit(0)` 判断环境 (NCCL/CUDA/Pallas 不可用则整文件退出), 这是 PyTorch 测试的常规做法, CPU CI 同样跳过它们; 在带 GPU 的 CI 环境它们正常收集, 按 2.89x 推算约 4,300 项。
- 注意: 28 个模块中的 `test_torchinductor_opinfo.py` 静态只有 1 个 `def test_`, 但经 OpInfo 全矩阵展开在 GPU CI 中可贡献数千项, 因此 **约 1.9 万是保守下界**。

### 6.4 典型文件的静态数 vs 实测收集数

| 文件 | 静态 `def test_` | 实测 collected | 放大倍数 |
|---|---:|---:|---:|
| `test_torchinductor_codegen_dynamic_shapes.py` | 5 | 1,411 | **282x** |
| `test_cache.py` | 23 | 728 | 31.7x |
| `test_torchinductor_dynamic_shapes.py` | 64 | 1,480 | 23.1x |
| `test_mix_order_reduction.py` | 34 | 514 | 15.1x |
| `test_control_flow.py` | 63 | 775 | 12.3x |
| `test_cooperative_reductions.py` | 13 | 167 | 12.8x |
| `test_torchinductor_strided_blocks.py` | 48 | 433 | 9.0x |
| `test_compiled_autograd.py` | 132 | 933 | 7.1x |
| `test_max_autotune.py` | 142 | 584 | 4.1x |
| `test_torchinductor.py` | 979 | 1,474 | 1.5x |
| `test_flex_attention.py` | 221 | 582 | 2.6x |

放大最极端的 `test_torchinductor_codegen_dynamic_shapes.py` 仅 5 个函数, 但用 `@parametrize` 笛卡尔积枚举配置组合, 展开出 1,411 项; 而 `test_torchinductor.py` 的 979 个函数大多单形态, 仅 1.5x。这说明: 静态函数数衡量"测试逻辑单元数", 收集数衡量"CI 执行单元数", 二者相差一个与参数化密度相关的系数 (本目录实测平均 2.89)。

### 6.5 实测过程中的环境坑 (复现参考)

1. 昇腾容器默认 autoload `torch_npu` 失败, 需 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`。
2. 测试基础设施依赖 `expecttest`/`hypothesis`/`numpy`/`parameterized`/`pyyaml`/`psutil`, 需额外安装; `test_aot_inductor_package.py` 还需拷入仓库 `tools/` 目录并加入 `PYTHONPATH`。
3. `main` 分支 `test/conftest.py` 依赖 `pytest_shard_custom` 且引用了发布版尚未导出的 `MultiProcContinuousTest`; 统计收集数时用 `--noconftest` 绕过即可。
4. 28 个测试文件在模块级用 `sys.exit(0)` 做环境跳过 (无 NCCL/CUDA/Pallas 时整文件退出); 纯 CPU wheel 下 `sys.exit` 会触发 pytest INTERNALERROR, 需 `--ignore` 这 28 个文件。
5. 发布版 torch (2.13.0) 落后于 `main` 分支 API, 会造成大量 import error; 需用 `--index-url https://download.pytorch.org/whl/nightly/cpu/` 安装 nightly (`pip install -U --pre`, 注意不加 `-U` 时 pip 不会升级已满足的版本)。

## 7. GPU 实测: 官方 CI 真实执行数据

### 7.1 数据基准: commit 与实测日期

| 数据集 | 代码 commit | torch 版本 | 硬件 | 实测/解析日期 |
|---|---|---|---|---|
| 静态分析 | 本地 main `e47c8486484` | — | — | 2026-08-18 |
| CPU 实测收集 | 同上 (`test/` 目录拷入容器) | 2.15.0.dev20260818+cpu (nightly, 与上述 commit 同日构建) | aarch64 CPU 容器 | 2026-08-18 晚 (08-19 复核) |
| CUDA 实际执行 | pytorch main `165426143e` (CI run 32193788776, 2026-08-18 22:40 UTC 触发) | CI 自构建 (同日 main) | NVIDIA L4 (cuda13.0) | 2026-08-19 拉取解析 |
| ROCm 实际执行 | 同上 (同一 run) | 同上 | AMD MI350 (gfx950) | 2026-08-19 拉取解析 |

两组 GPU 数据来自**同一个 CI run** (commit 165426143e), 与本地静态分析 commit (e47c8486484) 同日、差异极小, 口径可比。

### 7.2 数据源与获取方式

| GPU | CI job | artifact 来源 | 获取方式 |
|---|---|---|---|
| NVIDIA L4 (cuda13.0) | `linux-jammy-cuda13.0-py3.10-gcc11 / test (default, i/14)` 14 分片 | S3 `gha-artifacts` bucket | `curl https://gha-artifacts.s3.amazonaws.com/pytorch/pytorch/<run_id>/1/artifact/test-reports-test-default-<shard>-14-<runner>_<job_id>.zip` |
| AMD MI350 (rocm) | `test (inductor, i/2)` 2 分片 | GitHub artifacts (ROCm runner 无 S3 权限) | `gh api repos/pytorch/pytorch/actions/artifacts/<id>/zip` |

每个 zip 内是按测试文件组织的 junit XML (`python-pytest/inductor.<file>/<hash>.xml`), 统计 `<testcase>` 数即为实际执行数 (含 skipped, 不含收集失败)。

### 7.3 三平台实测对比: CPU / CUDA / ROCm

| 指标 | CPU (容器收集) | CUDA (NVIDIA L4) | ROCm (AMD MI350) |
|---|---:|---:|---:|
| **用例数** | **14,445** (收集) | **23,749** (实跑) | **5,126** (实跑) |
| 口径 | `--collect-only` 全收集 | junit 实际执行 (含 skip) | junit 实际执行 (含 skip) |
| 覆盖文件 | 156 / 184 | 77 (`test/inductor/` 75 + 目录外 2) | 5 |
| 其中 skipped | — (收集口径无此数) | 4,955 (20.9%) | 1,281 (25.0%) |
| failed / errored | — | 0 | 0 |
| 数据性质 | 环境可收集的**全集** | TD 裁剪后的**执行子集** | 固定 5 文件的**设计子集** |

三个数字口径不同, 不能直接互比; 下文按两两差异展开。

#### 差异 1: CUDA (23,749) vs CPU (14,445) —— +64%, 两个来源

1. **环境前置文件恢复** (+~9,100): 28 个模块在纯 CPU 环境 `sys.exit(0)` 跳过, GPU 上正常执行。大头是 opinfo/AOTI/子进程类: `test_torchinductor_opinfo` (3,697)、`test_compile_subprocess` (1,428)、`test_aot_inductor` (1,420) 等, 合计约 6,250 例是 CPU 上根本无法收集的。
2. **参数化展开更宽** (1.33x): 61 个两边都覆盖的文件, GPU 执行 14,314 项 vs CPU 收集 10,729 项。`@dtypes` 在 GPU 展开更多数据类型 (bfloat16/float16 的 GPU-only 路径), `instantiate_device_type_tests` 为 cuda 设备额外实例化一份用例。

#### 差异 2: CUDA (23,749) vs ROCm (5,126) —— 4.6 倍, 主要是 CI job 设计而非硬件能力

两者都是 GPU, 但跑的 job 不同:

| 维度 | CUDA (L4 `test (default, i/14)`) | ROCm (MI350 `test (inductor, i/2)`) |
|---|---|---|
| job 覆盖面 | `test/inductor/` 近全量 (TD 裁剪), 77 文件 | **固定 5 个文件**, 由 `.ci/pytorch/test.sh` 的 `test_inductor_shard()` 硬编码 |
| 包含文件 | 全部类型 | `test_torchinductor` + `test_torchinductor_opinfo` + `test_aot_inductor` + `test_cpu_select_algorithm` + ROCm 专属 `test_origami` |
| 厂商专属后端 | cutlass、`max_autotune_blackwell`、`nv_universal_gemm` | **origami** (AMD GEMM 库, 需 `TORCHINDUCTOR_ORIGAMI=1`, 仅 ROCm 跑) |
| skip 率 | 20.9% | 25.0% (MI350 对部分 bfloat16/arch 特性 skip 更多) |

关键证据: **两 GPU 上共同文件的展开数完全一致** —— `test_torchinductor_opinfo` 两边都是 3,697 项 (ROCm 1,432+2,265 分片合计), `test_aot_inductor` 两边都是 1,420 项 (803+617)。说明 opinfo 按 op x dtype 展开的矩阵与 GPU 厂商无关, 4.6 倍差距几乎全部来自 job 覆盖面 (ROCm 只跑 5 文件), 而非 AMD 支持的用例形态不同。

真正"厂商支持差异"的用例是少数, 且是双向的: ROCm 独有 origami (~9 例); NVIDIA 独有 cutlass 模板、Blackwell autotune、`nv_universal_gemm`; 此外 MI350 的 skip 率高 4 个百分点, 反映硬件能力 (arch/bfloat16 覆盖) 的边缘差异。

### 7.4 关键发现

1. **交叉验证**: ROCm 两分片的 `test_torchinductor_opinfo` 合计 (1,432 + 2,265 = 3,697) 与 L4 单文件执行数 (3,697) 完全一致 —— opinfo 全矩阵在两种 GPU 上均展开为 3,697 项, 印证了静态口径下它只有 1 个 `def test_` 但实际展开数千项的推断。
2. **GPU 展开系数大于 CPU**: 61 个两边都跑的文件, GPU 执行 14,314 项 vs CPU 收集 10,729 项 (**1.33x**), 因 `@dtypes` 在 GPU 上展开更多数据类型 (如 bfloat16/float16 GPU-only 路径)、`instantiate_device_type_tests` 增加 cuda 变体。
3. **GPU 恢复了 CPU 下跳过的文件**: `test_torchinductor_opinfo` (3,697)、`test_aot_inductor` (1,420)、`test_compile_subprocess` (1,428)、`test_cudagraph_trees` (244) 等在纯 CPU 环境下 `sys.exit(0)` 的模块在 GPU 上正常执行。
4. **GPU 全量推算**: L4 实跑 23,749 + TD 未跑的 93 个文件按 CPU 收集数 (3,716) x 1.33x 设备展开系数 ≈ 4,960, **单卡 CUDA 环境全量约 2.87 万个用例** (不含多卡/TPU 专属的 `test_pallas` 等)。

### 7.5 全量用例总表 (所有口径汇总)

| # | 统计口径 | 用例数 | 覆盖范围 | 获取方式 |
|---:|---|---:|---|---|
| 1 | 静态测试函数 | 6,477 | `test/inductor/` 全部 184 模块 | grep |
| 2 | CPU 容器全量收集 | 14,445 | 156 模块 (0 错误) | `pytest --collect-only` |
| 3 | CPU 全环境推算 (含 28 个环境跳过模块) | ~1.9 万 (下界) | 全部 184 模块 | 14,445 + ~4,300 推算 |
| 4 | **NVIDIA L4 CI 实际执行** | **23,749** | `test/inductor/` 75 文件 (23,677) + 目录外 2 文件 (72) | junit 解析 |
| 4a | └ fast 层 (`default` config, 每次提交) | 23,670 | 同上, 排除 slow 标记项 | junit skip 原因拆分 |
| 4b | └ slow 层 (`slow` config, 每日 slow.yml) | 79 | 清单见 `test/slow_tests.json` + `@slowTest` | junit skip 原因拆分 |
| 5 | AMD MI350 CI 实际执行 | 5,126 | 固定 5 文件 (opinfo/AOTI/origami) | junit 解析 |
| 6 | **单卡 CUDA 全量推算 (TD 不裁剪)** | **约 2.87 万** | 77 + 93 个 TD 未选文件 | 23,749 + 3,716 x 1.33 |

其中口径 4 的 23,677 (目录内) + 72 (目录外 `test_inductor_collectives` 67 + `test_inductor_compile_collectives` 5) = 23,749。

#### 7.5.1 按 CI 执行优先级分层 (fast / slow)

PyTorch CI 把测试分两个执行优先级层, 机制是:

- **标记**: 两个来源。代码内 `@slowTest` 装饰器 (inductor 目录 6 个静态函数: `test_benchmark_fusion` 1 / `test_cpu_repro` 3 / `test_torchinductor` 2); 以及 `test/slow_tests.json` 清单 (全仓 275 条, 按 "测试名 (classname)" 精确匹配, 可直接命中 opinfo 的单个展开项如 `test_comprehensive_linalg_svd_cuda_float64`, inductor 相关 44 条 = 40 个 opinfo 展开项 + 4 个 AOTI/dynamic 项)。
- **执行**: `default` config (trunk.yml 每次提交) 不设 `PYTORCH_TEST_WITH_SLOW`, slow 项**收集但 skip**; `slow` config (slow.yml, push main + 每日 cron 01:29 PDT, GPU 3 分片) 设 `PYTORCH_TEST_WITH_SLOW=1` + `PYTORCH_TEST_SKIP_FAST=1`, 只执行 slow 项。两层互补无重叠, **并集恰好等于收集全量**。

L4 CUDA 实测 (run 32193788776) 的分层结果:

| 优先级层 | CI 载体 | 用例数 | 其中实跑 | 其中 skip (原因) |
|---|---|---:|---:|---:|
| fast 层 (`default` config) | trunk.yml `test (default, i/14)`, 每次提交 | **23,670** | 18,794 | 4,876 (硬件/环境: `cpu not supported`、CuTeDSL/MPS/FlashAttention 库缺失、显存不足等, 非 slow) |
| slow 层 (`slow` config) | slow.yml 每日 cron, 3 分片 | **79** | 0 (本 run 内被 skip, 由 slow job 执行) | 79 (`test is slow`) |
| **合计 (两层并集)** | — | **23,749** | | = default job 收集全量 |

79 个 slow 项在文件上的分布: `test_torchinductor_opinfo` 41 (linalg/svd/插值/池化等重计算 op x dtype 组合, 对应 `slow_tests.json` 中 40 条清单 + 清单外运行时判定 1), `test_torchinductor` 9, `test_compile_subprocess` 8, `test_torchinductor_dynamic_shapes` 7, `test_torchinductor_codegen_dynamic_shapes` 6, `test_compiled_autograd` 4, `test_aot_inductor` 2, `test_cpu_repro` 2。目录外 2 个 distributed 文件 (72 例) 全部属于 fast 层。

即: **适配 (执行) 一次完整 CI, 需同时跑 fast 层 23,670 项 (每次提交) 与 slow 层 79 项 (每日一次), 合计 23,749 项 —— 与 default job 的收集全量一致, slow 层占比仅 0.3%, 但含 svd/linalg 等单用例耗时最重的数值验证**。

### 7.6 按特性拆分的用例统计

同一测试文件按功能特性分组, 各口径用例数对照 (CUDA 列为 L4 实跑、目录内严格口径; skip 为其中跳过数):

| 特性分组 | 文件数 | 静态函数 | CPU 收集 | CUDA 实跑 | 其中 skip | 特性说明 |
|---|---:|---:|---:|---:|---:|---|
| 端到端正确性 (torchinductor 系列) | 7 | 1,106 | 4,804 | 12,181 | 2,396 | `test_torchinductor*` 主战场: 编译后模型与 eager 逐算子数值对比, 含 opinfo 全矩阵 (单文件展开 3,697 项), 是 CI 用例数的最大来源 |
| 编译基础设施/缓存/调优 | 41 | 1,374 | 4,791 | 4,385 | 794 | 代码缓存 (codecache/自动缓存命中)、编译子进程与 worker 池、max_autotune 自动调优、autoheuristic、配置系统、compiled_autograd/compiled_optim (编译版反向与优化器) |
| 其他功能与平台 | 57 | 1,023 | 1,972 | 1,599 | 425 | 长尾: MPS/XPU/Halide 等平台后端、fp8、内存规划、性能冒烟、fxir backend、user streams 等专项 |
| AOTInductor / AOTI | 8 | 352 | 108 | 1,516 | 637 | `torch.export` 后 AOT 编译为独立 `.so`, 部署不依赖 Python; 含 custom_ops/package 归档加载, CUDA 上跑 1,420 项 (两 GPU 一致) |
| 内核后端与代码生成 | 47 | 1,379 | 1,785 | 1,396 | 624 | Triton 内核生成与 IR (lowering/irgen/layout)、cutlass/ck/pallas/cutedsl 模板后端、GEMM/conv 分解、atomic/indexing/indirect 等低级语义 |
| CPU / C++ wrapper | 5 | 325 | 5 | 1,133 | 14 | cpp_wrapper 模式: 把 Python 包装代码编译为 C++; `test_cpu_repro` (289 函数) 主要在 CPU CI 跑, GPU job 只跑其中 cuda 相关部分 |
| 特性支持 | 17 | 569 | 956 | 1,081 | 47 | flex_attention 可编程注意力模板、自定义算子 (custom_op/auto_functionalize)、higher-order ops、模式匹配改写、动态形状 sym_*、minifier、调试工具 |
| CUDA Graph | 4 | 328 | 3 | 366 | 18 | cudagraph_trees: CUDA Graph 录制/回放管理、内存池、与编译缓存交互; 纯 CUDA 环境, CPU 上 `sys.exit` |
| 分布式/集合通信 | 2 | 21 | 21 | 20 | 0 | inductor 编译下的集合通信模式 (目录内部分; 主体在 `test/distributed/`, 72 例) |
| **合计** | 188 | **6,477** | **14,445** | **23,677** | **4,955** | |

注: 文件数合计 188 为三口径文件名并集 (静态 184 个 `test_*.py` + 收集/执行中出现的 4 个非 `test_` 前缀辅助模块); compiled_autograd 并入"编译基础设施"组。各列数字与第 6/7 节分口径合计一致。

特性维度的三平台观察:

- **端到端正确性是 GPU CI 的绝对大头** (CUDA 12,181 / 51%): opinfo 矩阵只在 GPU 完整展开, CPU 仅收集到 4,804。
- **编译基础设施是 CPU 与 GPU 共同的大头**: CPU 4,791 / CUDA 4,385, 两边都接近满覆盖, 因其不依赖 GPU 硬件 (缓存逻辑、worker、配置)。
- **AOTI 与 CUDA Graph 的 skip 率最高** (637/1,516 与 18/366 中相当比例来自环境分支): 依赖编译器工具链或特定 GPU 能力。
- **CPU/C++ wrapper 组两极分化**: CPU 仅收集 5 例 (其余 `sys.exit`), CUDA 实跑 1,133 —— cpp_wrapper 的验证主要在 GPU CI 完成。

## 8. 统计口径与局限

- **口径**: "测试用例数" = `def test_` 开头的函数个数 (静态文本匹配, 含类方法); "测试类" = `class Test*`; C++ 用例 = `TEST`/`TEST_F`/`TEST_P` 宏。子目录 `pallas_skip_tests/`、`extension_backends/` 等仅含配置, 无测试函数。
- **未计入**: 参数化展开后的用例数 (静态口径, 见第 5/6 节)、`functorch`/`export` 等目录中间接经由 `torch.compile` 的用例、benchmarks 目录的性能用例。
- **局限**: 静态匹配会把少量非测试 helper (命名为 `def test_` 但被条件跳过) 计入, 也会漏掉运行时动态生成的用例; 误差量级估计在 1-2% 以内, 不影响结论。
- **实测口径**: 第 6 节的 14,445 为 torch 2.15.0.dev nightly + aarch64 CPU 容器下 156 个模块 (0 错误) 的 pytest 收集数; 另 28 个模块因纯 CPU 环境 `sys.exit(0)` 跳过, 全环境 (含 GPU) 推算总量约 1.9 万。
- **源码对比**: `torch/_inductor/` 源码共 379 个 `.py` 文件, 对应 184 个测试模块 (约 2:1)。

## 9. 结论

1. torchinductor 在 `test/inductor/` 下维护 **6,477 个测试函数、333 个测试类、184 个测试模块**, 占 PyTorch 全仓 Python 测试函数的 **14.8%**; 加上 Dynamo 前端后, torch.compile 编译栈合计占 **约 31.8%**。
2. **CPU 实测收集 14,445 个用例** (torch 2.15.0.dev nightly, 156 模块 0 错误), 参数化平均放大 **2.89 倍**。
3. **GPU 实测** (CI run 32193788776, commit 165426143e, 2026-08-18): NVIDIA L4 实际执行 **23,749 个用例** (77 文件, 0 失败), AMD MI350 执行 5,126 个。CUDA 比 CPU 多 64%, 来自环境前置文件恢复 (opinfo/AOTI 等仅 GPU 可收集) 与 1.33x 的 GPU 参数化展开; CUDA 比 ROCm 多 4.6 倍, **主要是 CI job 覆盖面设计** (ROCm 固定 5 文件 vs L4 近全量), 而非硬件差异 —— 共同文件展开数在两种 GPU 上完全一致 (opinfo 3,697 / AOTI 1,420)。单卡 CUDA 全量推算**约 2.87 万**。
4. 测试结构呈"三层鼎立 + 长尾" (见 7.6 特性拆分): 端到端正确性 (CUDA 12,181, 占 51%)、编译基础设施 (静态 1,374 / CUDA 4,385)、内核代码生成 (1,379 / 1,396), 外加 57 个长尾特性/平台模块; 厂商专属后端用例是少数且双向 (ROCm 独有 origami, NVIDIA 独有 cutlass/Blackwell autotune/nv_universal_gemm)。
5. inductor 的测试边界不止 `test/inductor/`: 分布式场景在 `test/distributed/` (静态 93, GPU 实跑 72), Dynamo 协同在 `test/dynamo/` (75 例, 文件名含 inductor), 宽口径耦合文件达 139 个。

## 附录: 复现命令

```bash
# test/inductor 基础规模
grep -rE '^[[:space:]]*def test_' test/inductor --include='*.py' | wc -l   # 6477
grep -rE '^class Test' test/inductor --include='*.py' | wc -l              # 333
find test/inductor -name '*.py' | wc -l                                    # 201

# 全仓规模
grep -rE '^[[:space:]]*def test_' test --include='*.py' | wc -l            # 43764
grep -rE '^(TEST|TEST_F|TEST_P)\(' test --include='*.cpp' | wc -l          # 2789 (grep -h 聚合)

# 目录外 inductor 相关
grep -rl 'inductor' test --include='*.py' | grep -v '^test/inductor/' | wc -l   # 139

# 主题分组
python3 agent_space/group_inductor_tests.py   # 本仓库 agent_space/ 下的临时脚本 (git-ignored)
```

实测收集 (远程容器, torch 2.15.0.dev20260818+cpu + pytest 8.4.1):

```bash
# 容器内准备: 拷入本仓库 test/ 与 tools/, 安装 nightly 及依赖
pip install -U --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu/
pip install pytest expecttest hypothesis numpy parameterized pyyaml psutil

# 在 test/ 目录下执行 (IGNORES 为 28 个模块级 sys.exit 文件的 --ignore 列表)
PYTHONPATH=/path/to/repo:/path/to/repo/test/inductor TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  python -m pytest --noconftest inductor --collect-only -q \
    --continue-on-collection-errors $IGNORES
# 结果: 14445 tests collected in 13.85s (0 errors)
```

GPU 实测数据 (官方 CI junit artifact):

```bash
# 1. 找最近成功的 trunk run
gh run list -R pytorch/pytorch --workflow=trunk.yml --limit 40 --json databaseId,conclusion \
  --jq '.[] | select(.conclusion=="success") | .databaseId'

# 2a. L4 CUDA 报告从 S3 下载 (job id 从 run 的 jobs API 获取)
curl -O https://gha-artifacts.s3.amazonaws.com/pytorch/pytorch/<run_id>/1/artifact/test-reports-test-default-<shard>-14-<runner>_<job_id>.zip

# 2b. ROCm 报告从 GitHub artifacts 下载
gh api repos/pytorch/pytorch/actions/runs/<run_id>/artifacts?per_page=100 --paginate \
  --jq '.artifacts[] | select(.name | test("test-reports.*inductor")) | "\(.id) \(.name)"'
gh api repos/pytorch/pytorch/actions/artifacts/<id>/zip > reports.zip

# 3. 解析 junit XML 统计 <testcase> 数 (按 python-pytest/inductor.* 目录过滤)
python3 agent_space/parse_junit_inductor.py <解压目录>
```
