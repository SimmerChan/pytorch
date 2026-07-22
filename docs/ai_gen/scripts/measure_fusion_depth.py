#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图级融合深度测量(D 口径):GPU(CUDA) vs NPU(Ascend)对比脚本。

用途:
    在【真实硬件】上编译一组典型子图,量化 inductor 实际把多少 op 融进多少个 kernel,
    以及多少 op 走了 extern/fallback。这是判断 "GPU 与 NPU 融合范围差距到底有多大
    实际影响" 的终极口径(静态列表口径给不了)。

核心指标(全部来自 torch._inductor.metrics,真实 API):
    ops_in        : ir_nodes_pre_fusion     进入 scheduler 的 IR 节点数(融合前)
    kernels       : generated_kernel_count  codegen 生成的融合 compute kernel 数
    extern_calls  : 解析 output_code 得到的 extern 库调用(mm/conv/aclnn...)
    fusion_depth  : ops_in / kernels         平均每个 kernel 融了多少 op(越大=融合越深)
    npu_fallback  : (仅 NPU)compile 后图里命中 torch_npu FALLBACK_LIST 的 op 数

运行:
    # 在 GPU 机器
    python measure_fusion_depth.py --device cuda
    # 在 NPU 机器
    python measure_fusion_depth.py --device npu
    # 不传 --device 则自动探测(cuda -> npu -> cpu)
    python measure_fusion_depth.py

    两台机器各跑一次,把两张表对照即可看出 GPU/NPU 融合深度差距。
"""

import argparse
import io
import logging
import re
import sys

import torch
import torch._dynamo
from torch._inductor import metrics


# ----------------------------- 子图定义 -----------------------------
# 每个子图覆盖一类典型融合模式;args 工厂按 device 产出输入。
def g_pointwise(x):
    # 纯逐元素链:理论应融成 1 个 kernel
    y = torch.relu(x)
    y = y * 2.0 + 1.0
    y = torch.sigmoid(y)
    y = torch.tanh(y) * 0.5
    return y


def g_reduction(x):
    # reduction + epilogue pointwise
    y = x.float()
    s = y.sum(dim=-1, keepdim=True)
    m = y.mean(dim=-1, keepdim=True)
    return (y - m) / (s + 1e-6)


def g_mlp(x, w1, b1, w2, b2):
    # MLP block:Linear -> GELU -> Linear (matmul extern + pointwise 融合)
    h = torch.nn.functional.linear(x, w1, b1)
    h = torch.nn.functional.gelu(h)
    return torch.nn.functional.linear(h, w2, b2)


def g_softmax_layernorm(x, w, b):
    # softmax + layernorm:reduction + 归一化 pointwise
    attn = (x @ w) / (x.size(-1) ** 0.5)
    attn = torch.softmax(attn, dim=-1)
    ln = torch.nn.functional.layer_norm(attn, [attn.size(-1)], w, b)
    return ln


def g_attention(q, k, v):
    # scaled dot-product attention(非 fused SDPA):两段 matmul + softmax
    d = q.size(-1) ** 0.5
    scores = (q @ k.transpose(-1, -2)) / d
    probs = torch.softmax(scores, dim=-1)
    return probs @ v


def make_args(name, device):
    dtype = torch.float32
    if name == "pointwise":
        return (torch.randn(2048, 2048, device=device, dtype=dtype),)
    if name == "reduction":
        return (torch.randn(2048, 2048, device=device, dtype=dtype),)
    if name == "mlp":
        x = torch.randn(512, 512, device=device, dtype=dtype)
        w1 = torch.randn(2048, 512, device=device, dtype=dtype)
        b1 = torch.randn(2048, device=device, dtype=dtype)
        w2 = torch.randn(512, 2048, device=device, dtype=dtype)
        b2 = torch.randn(512, device=device, dtype=dtype)
        return (x, w1, b1, w2, b2)
    if name == "softmax_layernorm":
        x = torch.randn(64, 64, 64, device=device, dtype=dtype)
        w = torch.randn(64, device=device, dtype=dtype)
        b = torch.randn(64, device=device, dtype=dtype)
        return (x, w, b)
    if name == "attention":
        q = torch.randn(8, 64, 64, 64, device=device, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        return (q, k, v)
    raise ValueError(name)


SUBGRAPHS = [
    ("pointwise", g_pointwise),
    ("reduction", g_reduction),
    ("mlp", g_mlp),
    ("softmax_layernorm", g_softmax_layernorm),
    ("attention", g_attention),
]


# ----------------------------- 设备探测 -----------------------------
def detect_device(req: str):
    if req == "auto":
        if torch.cuda.is_available():
            return "cuda", "GPU (CUDA)"
        try:
            import torch_npu  # noqa: F401
            if torch.npu.is_available():
                return "npu", "NPU (Ascend)"
        except ImportError:
            pass
        return "cpu", "CPU (baseline only)"
    # 显式指定
    if req == "cuda" and not torch.cuda.is_available():
        sys.exit("ERROR: --device cuda 但 CUDA 不可用")
    if req == "npu":
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            sys.exit("ERROR: --device npu 但 torch_npu 未安装")
        if not torch.npu.is_available():
            sys.exit("ERROR: --device npu 但 NPU 不可用")
    label = {"cuda": "GPU (CUDA)", "npu": "NPU (Ascend)", "cpu": "CPU"}[req]
    return req, label


# ----------------------------- output_code 捕获 -----------------------------
def attach_code_capture():
    """挂一个 handler 到 output_code_log,把生成的 wrapper 代码缓存到 StringIO。"""
    buf = io.StringIO()
    try:
        from torch._inductor.codecache import output_code_log
        h = logging.StreamHandler(buf)
        output_code_log.addHandler(h)
        output_code_log.setLevel(logging.DEBUG)
    except Exception as e:  # 捕获不到也能跑(只是少了 extern 统计)
        print(f"[warn] output_code 捕获不可用,extern 统计将缺省: {e}", file=sys.stderr)
        return buf, False
    return buf, True


def parse_extern_calls(code: str):
    """从生成的 wrapper 代码里数 extern 库调用(mm/conv/aclnn...)。"""
    externs = re.findall(r"extern_kernels\.(\w+)", code)
    # NPU 的 aclnn fallback 通常形如 torch.ops.npu.aclnnXxx / aclearrnn
    aclnn = re.findall(r"(?:torch\.ops\.npu\.(aclnn\w+)|aclnn\w+)", code)
    aclnn = [a for a in aclnn if a]
    return sorted(set(externs)), sorted(set(aclnn))


# ----------------------------- NPU fallback 交叉引用(可选)---------------------
def load_npu_fallback_names():
    """尽力从 torch_npu 加载 FALLBACK_LIST 概念算子名集合。"""
    try:
        from torch_npu._inductor.lowering_fallback_list import FALLBACK_LIST
    except Exception:
        return None
    names = set()
    for op in FALLBACK_LIST:
        # op 可能是 OpOverloadPacket / OpOverload / 高阶算子对象
        try:
            names.add(op.__name__)
        except AttributeError:
            try:
                names.add(op.name)
            except AttributeError:
                continue
    return names


def count_fallback_in_compiled(compiled_fn, args, fb_names):
    """trace compiled graph 的 node.target,统计命中 fallback 的 op 数。"""
    if not fb_names:
        return None
    # 通过 guard 重新进入拿不到 graph;用 make_graph_signature 不通用。
    # 退而求其次:用 torch.export 拿结构化图(部分算子会被 decompose,仅作粗略统计)。
    try:
        import torch.export  # noqa: F401
        ep = torch.export.export(compiled_fn, args=tuple(a.clone() for a in args))
    except Exception as e:
        return f"(export 失败: {type(e).__name__})"
    hits = set()
    for node in ep.graph.nodes:
        if node.op != "call_function":
            continue
        target = node.target
        tname = getattr(target, "__name__", None) or getattr(target, "name", None)
        if tname and tname in fb_names:
            hits.add(tname)
    return sorted(hits) if hits else []


# ----------------------------- 单次测量 -----------------------------
def measure_one(name, fn, args, device, buf, fb_names):
    metrics.reset()
    buf.seek(0); buf.truncate(0)
    torch._dynamo.reset()

    compiled = torch.compile(
        fn,
        mode="max-autotune-no-cudagraphs",
        fullgraph=True,
    )
    # warmup + 正式跑
    _ = compiled(*[a.clone() if torch.is_tensor(a) else a for a in args])
    _ = compiled(*[a.clone() if torch.is_tensor(a) else a for a in args])

    if device in ("cuda",):
        torch.cuda.synchronize()
    elif device == "npu":
        torch.npu.synchronize()

    ops_in = metrics.ir_nodes_pre_fusion
    kernels = metrics.generated_kernel_count
    # ir_nodes_pre_fusion 是【post-lowering 合并后】的 IR 节点数:一条纯 pointwise 链
    # 会在 lowering 阶段被合并成 1 个 Pointwise IR 节点,所以纯 pointwise 子图 ops_in 常为 1。
    # 它衡量的是"fusion pass 看到的待融合单位",对 reduction/extern 边界多的图更有区分度。
    depth = (ops_in / kernels) if kernels else None  # kernels=0 → 无 compute kernel(全 extern)
    externs, aclnn = parse_extern_calls(buf.getvalue())
    extern_repr = ",".join(externs + aclnn) if (externs or aclnn) else "-"

    fb_hits = count_fallback_in_compiled(compiled, args, fb_names) if device == "npu" else None
    if isinstance(fb_hits, list):
        fb_repr = ",".join(fb_hits) if fb_hits else "-"
        fb_n = len(fb_hits)
    elif fb_hits is None:
        fb_repr = "n/a"; fb_n = "-"
    else:
        fb_repr = fb_hits; fb_n = "?"

    return dict(name=name, ops_in=ops_in, kernels=kernels, depth=depth,
                extern=extern_repr, fb_n=fb_n, fb_repr=fb_repr)


# ----------------------------- 输出 -----------------------------
def print_table(device_label, rows):
    print(f"\n{'='*78}")
    print(f" 后端: {device_label}")
    print(f"{'='*78}")
    hdr = f"{'subgraph':<20}{'ops_in':>8}{'kernels':>9}{'depth':>8}  {'extern/fallback_calls':<24}{'npu_fb':>6}"
    print(hdr)
    print("-" * 78)
    for r in rows:
        ext = r["extern"][:24]
        depth_str = f"{r['depth']:.2f}" if isinstance(r["depth"], (int, float)) else "n/a"
        print(f"{r['name']:<20}{r['ops_in']:>8}{r['kernels']:>9}{depth_str:>8}  "
              f"{ext:<24}{str(r['fb_n']):>6}")
    print("-" * 78)
    tot_ops = sum(r["ops_in"] for r in rows if isinstance(r["ops_in"], int))
    tot_k = sum(r["kernels"] for r in rows if isinstance(r["kernels"], int))
    tot_depth = (tot_ops / tot_k) if tot_k else 0
    print(f"{'合计':<20}{tot_ops:>8}{tot_k:>9}{tot_depth:>8.2f}")
    # extern 汇总
    all_extern = sorted({e for r in rows for e in (r["extern"].split(",") if r["extern"] != "-" else [])})
    if all_extern:
        print(f"extern 库调用汇总: {all_extern}")
    npu_fbs = sorted({f for r in rows for f in (r["fb_repr"].split(",") if r["fb_repr"] not in ("-","n/a") else [])})
    if npu_fbs:
        print(f"NPU fallback op 汇总: {npu_fbs}")
    print()


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "npu", "cpu"])
    ap.add_argument("--mode", default="max-autotune-no-cudagraphs")
    args = ap.parse_args()

    device, label = detect_device(args.device)
    print(f"[*] 探测到设备: {label}  (device='{device}')")
    if device == "cpu":
        print("[!] CPU 仅作 baseline,无 cuda/npu 时融合行为不代表真实硬件差距。")

    buf, _ = attach_code_capture()
    fb_names = load_npu_fallback_names() if device == "npu" else None
    if device == "npu":
        print(f"[*] 加载 NPU FALLBACK_LIST 概念算子: {'成功' if fb_names else '失败(跳过交叉引用)'}")

    rows = []
    for name, fn in SUBGRAPHS:
        try:
            targs = make_args(name, device)
            r = measure_one(name, fn, targs, device, buf, fb_names)
            rows.append(r)
            print(f"  [ok] {name}: ops_in={r['ops_in']} kernels={r['kernels']} depth={r['depth']:.2f}")
        except Exception as e:
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            rows.append(dict(name=name, ops_in="-", kernels="-", depth=0,
                             extern="-", fb_n="-", fb_repr="-"))

    print_table(label, [r for r in rows if isinstance(r["ops_in"], int)])

    # 提示对照
    print("说明:")
    print("  ops_in   = 进入 scheduler 的 IR 节点数(融合前的待融合单位)")
    print("             注意:纯 pointwise 链在 lowering 阶段已合并成 1 个 IR 节点,")
    print("             所以纯 pointwise 子图 ops_in 常为 1;reduction/extern 边界越多越有区分度。")
    print("  kernels  = inductor 生成的融合 compute kernel 数(越少越好)")
    print("  depth    = ops_in / kernels,平均每个 kernel 融了几个 op(越大=融合越深);")
    print("             kernels=0(全 extern,如纯 attention)显示 n/a")
    print("  extern   = 调用外部库的算子(mm/conv/aclnn 等,不参与 inductor 融合)")
    print("  npu_fb   = (仅NPU) compile 后图里命中 FALLBACK_LIST 的 op 数")
    print()
    print("对照方法:GPU 和 NPU 各跑一次,比较每个子图的 kernels / depth / npu_fb。")
    print("         kernels 更多、depth 更小、npu_fb 更大 => NPU 融合更弱。")


if __name__ == "__main__":
    main()
