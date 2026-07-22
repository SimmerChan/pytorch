#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decomposition 口径实测:GPU 与 NPU 各自实际能融合多少算子,差距多少。

方法:
  1. 从 core_aten_decompositions() + inductor decomposition 拿到【分解表】。
  2. survivor 判定:aten op 只要【任一】overload 不在分解表就算存活(直达 lowering)。
  3. 未被分解的 op(survivor) + prims(registration) = 到达 lowering 的【primitive 宇宙 U】。
  4. 从 GPU lowering.py 抽取【GPU 可融合】集合(pointwise+reduction+view+scan+inplace)。
  5. 从 torch_npu fallback 列表抽取【NPU 不融合】集合。
  6. 逐 op 判定:GPU 可融? NPU 可融?(= GPU可融 且 不在 NPU fallback)
  7. 出三组数:|U|, GPU 可融, NPU 可融, 差距, 覆盖率。

运行:
  cd <pytorch 仓根>
  TORCH_NPU_FALLBACK_LIST=/path/to/torch_npu/_inductor/lowering_fallback_list.py \
      python docs/ai_gen/scripts/measure_decomp_coverage.py

依赖:torch(本仓)、torch_npu 的 lowering_fallback_list.py(单独的仓,用环境变量指定)。
"""

import os
import re
import sys

import torch

# 本脚本位于 docs/ai_gen/scripts/,向上 4 级到 pytorch 仓根。
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
LOWER = os.path.join(_REPO_ROOT, "torch/_inductor/lowering.py")

# torch_npu 是独立仓(不在本仓内),用环境变量指向其 fallback 列表文件。
FB = os.environ.get("TORCH_NPU_FALLBACK_LIST")


# ---------------- 1. GPU 可融合集合(概念算子) ----------------
def build_gpu_fusible():
    """构建 GPU 可融合算子集(产生 fusible IR:Pointwise/Reduction/Scan/View/In-place)。

    方法学(为什么不直接用 `lowerings` 字典):
      - `lowerings` 是"inductor 能 lower"的全集(1768 条,含 template/scatter/sort/fallback),
        `op in lowerings` 会把不可融的也算进去 -> 严重高估。它不是"可融"的注册表。
      - lowering fn 全部被同一个 `wrapped` 包装、无 IR 类型属性、闭包特征一致,
        无法按 fn 静态分类 IR 类型;import 时注册 + reload 重定义,纯插桩也走不通。
      - 因此用【注册辅助函数】作为 IR 类型的静态代理:
          register_pointwise -> Pointwise IR;make_reduction -> Reduction IR。
      - pointwise 用 regex(register_pointwise(aten.X) 绑定实际 op 名,准确)。
      - reduction/view/scan/inplace 用显式完整集(小而稳定,逐一枚举,含 amax/amin/mean/var)。
    """
    src = open(LOWER).read()
    pw = set(re.findall(r"register_pointwise(?:_numeric(?:_ldf64)?)?\(\s*aten\.(\w+)", src))
    pw |= set(re.findall(r"register_pointwise(?:_numeric(?:_ldf64)?)?\(\s*prims\.(\w+)", src))
    red = set(re.findall(r'make_reduction\("(\w+)"', src))
    # make_reduction 的内部名(max/min)与实际注册的 op 名(aten.amax/amin/mean/var)不一致,
    # 这些通过 register_lowering(aten.X)(make_reduction("...")) 注册,补上实际 op 名
    red |= {"amax", "amin", "mean", "var"}
    views = set("""view _unsafe_view reshape permute squeeze squeeze_copy unsqueeze expand
        expand_as broadcast_tensors broadcast_in_dim slice select select_scatter
        slice_scatter split split_with_sizes unbind unfold diagonal diagonal_copy
        diagonal_scatter cat alias detach detach_ lift lift_fresh view_of repeat
        as_strided as_strided_copy glu t transpose flatten""".split())
    scan = set("cumsum cumprod cummax cummin logcumsumexp".split())
    inplace = set("""add_ sub_ mul_ div_ relu_ sigmoid_ neg_ abs_ sqrt_ rsqrt_ exp_ log_
        tanh_ bitwise_and_ bitwise_or_ bitwise_xor_ bitwise_not_ clamp_ floor_ ceil_
        round_ trunc_ lerp_""".split())
    fusible = set()
    for op in pw:
        fusible |= {("aten", op), ("prims", op)}
    for op in red:
        fusible.add(("aten", op))
    for op in views | scan | inplace:
        fusible.add(("aten", op))

    def cat(ns, op):
        if op in pw:
            return "Pointwise"
        if op in red:
            return "Reduction"
        if op in views:
            return "View/Shape"
        if op in scan:
            return "Scan"
        if op in inplace:
            return "In-place"
        return "Other"
    return fusible, cat


# ---------------- 2. NPU fallback 集合(概念算子) ----------------
def build_npu_fallback():
    if not FB or not os.path.isfile(FB):
        sys.exit(
            "ERROR: set TORCH_NPU_FALLBACK_LIST to torch_npu/_inductor/lowering_fallback_list.py\n"
            "  例:TORCH_NPU_FALLBACK_LIST=/path/to/torch_npu/_inductor/lowering_fallback_list.py "
            f"python {__file__}"
        )
    src = open(FB).read()
    out = set()
    for ns, full in re.findall(
        r"\b(aten|prims|_c10d_functional|quantized_decomposed|rngprims|inductor|"
        r"fsdp|_dtensor|_inductor_test)\.([A-Za-z_]\w*(?:\.\w+)*)", src):
        out.add((ns, full.split(".")[0]))
    return out


# ---------------- 3. 分解表 + survivor,构造到达 lowering 的 primitive 宇宙 ----------------
def build_universe(gpu_fusible):
    """U = aten survivor(overload 级判定)+ prims(registration)。

    survivor 判定(关键修正):一个 op 只要【任一】overload 不在分解表,就算存活——
    因为 mean/max/min/add/cat/cumsum 等只是罕见 overload(如 names_dim)被分解,
    主 overload(default/dim)仍存活并进 lowering。早期版本用"op 名是否在分解表"
    粗判,会误把这些排除(Reduction 漏为 3、Scan 漏为 0)。
    """
    from torch._inductor.compile_fx import select_decomp_table
    full = select_decomp_table()                                  # inductor 实际分解表(1160 overload)
    aten = torch.ops.aten

    def survives(name):
        pkt = getattr(aten, name, None)
        if pkt is None or not pkt.overloads():
            return False
        return any(getattr(pkt, ov) not in full for ov in pkt.overloads())

    aten_pkts = set(n for n in dir(aten) if not n.startswith("_")
                    and hasattr(getattr(aten, n, None), "__call__"))
    survivors = {("aten", n) for n in aten_pkts if survives(n)}
    prims_reg = {("prims", o) for ns, o in gpu_fusible if ns == "prims"}
    U = survivors | prims_reg
    fully_decomposed = sum(1 for n in aten_pkts if not survives(n))
    stats = dict(
        aten_packets=len(aten_pkts),
        fully_decomposed=fully_decomposed,
        survivors=len(survivors),
        prims=len(prims_reg),
        U=len(U),
    )
    return U, stats


def main():
    gpu_fusible, cat = build_gpu_fusible()
    npu_fb = build_npu_fallback()
    U, stats = build_universe(gpu_fusible)

    gpu_fuse_U = U & gpu_fusible
    npu_fuse_U = {(ns, op) for (ns, op) in gpu_fuse_U if (ns, op) not in npu_fb}
    npu_loss_U = gpu_fuse_U - npu_fuse_U

    print("=" * 66)
    print(" Decomposition 口径实测(GPU vs NPU 可融合算子数)")
    print("=" * 66)
    print(f"aten packet 总数              : {stats['aten_packets']}")
    print(f"全部 overload 被分解的算子    : {stats['fully_decomposed']}  (其余 op 主 overload 存活,进 lowering)")
    print(f"survivor(直达 lowering)       : {stats['survivors']}")
    print(f"+ prims(registration)         : {stats['prims']}")
    print(f"宇宙 |U|(lowering 真实处理集)  : {stats['U']}")
    print(f"  其中【GPU 可融合】          : {len(gpu_fuse_U)}")
    print(f"  其中【NPU 可融合】          : {len(npu_fuse_U)}")
    print(f"  GPU 可融但 NPU fallback(差距): {len(npu_loss_U)}")
    print()
    if gpu_fuse_U:
        cov = len(npu_fuse_U) / len(gpu_fuse_U) * 100
        print(f"  NPU 覆盖率(相对 GPU)       : {cov:.1f}%")
        print(f"  覆盖率差距                 : {100 - cov:.1f} 个百分点")
    print()

    # 三组算子完整清单(分类 markdown 表)
    print_op_tables(gpu_fuse_U, npu_fuse_U, npu_loss_U, cat)


def print_op_tables(gpu_U, npu_U, loss, cat):
    """输出三张分类表:① NPU 可融合(GPU+NPU 都可融)② NPU 不可融合(差距)③ 校验汇总。"""
    def by_cat(s):
        d = {}
        for ns, op in sorted(s):
            d.setdefault(cat(ns, op), {}).setdefault(ns, []).append(op)
        return d

    ORDER = ["Pointwise", "Reduction", "View/Shape", "Scan", "In-place", "Other"]

    def render(title, s):
        d = by_cat(s)
        print(f"\n### {title} (共 {len(s)} 个)")
        print("| 类别 | 数量 | 算子清单 |")
        print("|---|---|---|")
        for k in ORDER:
            if k not in d:
                continue
            parts = []
            for ns in sorted(d[k]):
                ops = ", ".join(f"{ns}.{o}" for o in sorted(d[k][ns]))
                parts.append(ops)
            print(f"| {k} | {sum(len(v) for v in d[k].values())} | " + "<br>".join(parts) + " |")

    render("① NPU 可融合 (GPU+NPU 都可融)", npu_U)
    render("② NPU 相对 GPU 不可融合 (差距)", loss)
    # 校验:GPU 可融合 = NPU 可融合 + 差距
    assert gpu_U == npu_U | loss, "集合不一致!"
    print(f"\n校验:GPU 可融合({len(gpu_U)}) = NPU 可融合({len(npu_U)}) + 差距({len(loss)}) ✓")


if __name__ == "__main__":
    main()
