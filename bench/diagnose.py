"""Why is tl.dot slow on sm75? Compile the forward kernel and a bare fp16 matmul kernel, look for
tensor-core instructions (mma) in the PTX, and time a few variants at seq 4096."""

import json
import re

import torch
import triton
import triton.language as tl

from tattn import attention
from tattn.kernels import _fwd_kernel


@triton.jit
def mm(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(A + rm[:, None] * K + (k + rk)[None, :])
        b = tl.load(B + (k + rk)[:, None] * N + rn[None, :])
        acc += tl.dot(a, b)
    tl.store(C + rm[:, None] * N + rn[None, :], acc.to(tl.float16))


def ptx_stats(ptx: str) -> dict:
    return {
        "mma": len(re.findall(r"\bmma\.sync", ptx)),
        "fma_f32": len(re.findall(r"\bfma\.rn\.f32", ptx)),
        "ldmatrix": len(re.findall(r"\bldmatrix", ptx)),
    }


out: dict = {"gpu": torch.cuda.get_device_name(), "triton": triton.__version__}
M = N = K = 4096
a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(K, N, device="cuda", dtype=torch.float16)
c = torch.empty(M, N, device="cuda", dtype=torch.float16)
for bm, bn, bk, w in ((64, 64, 32, 4), (128, 128, 32, 8)):
    k = mm[(M // bm, N // bn)](a, b, c, M, N, K, BM=bm, BN=bn, BK=bk, num_warps=w)
    ms = triton.testing.do_bench(
        lambda: mm[(M // bm, N // bn)](a, b, c, M, N, K, BM=bm, BN=bn, BK=bk, num_warps=w)
    )
    out[f"triton_mm_{bm}x{bn}x{bk}"] = {"tflops": 2 * M * N * K / ms / 1e9, **ptx_stats(k.asm["ptx"])}
ms = triton.testing.do_bench(lambda: a @ b)
out["torch_mm_tflops"] = 2 * M * N * K / ms / 1e9

q = torch.randn(4, 16, 4096, 64, device="cuda", dtype=torch.float16)
attention(q, q, q)
# The autotuner keeps compiled kernels per device; grab the PTX of the best config.
kern = _fwd_kernel.fn
try:
    best = _fwd_kernel.best_config
    out["fwd_best_config"] = str(best)
except AttributeError:
    pass
caches = [v for v in kern.device_caches.values()] if hasattr(kern, "device_caches") else []
ptxs = []
for dc in caches:
    for ck in dc[0].values():
        if hasattr(ck, "asm") and "ptx" in ck.asm:
            ptxs.append(ptx_stats(ck.asm["ptx"]))
out["fwd_ptx"] = ptxs[:4]
from pathlib import Path

Path("results").mkdir(exist_ok=True)
print(json.dumps(out, indent=2))
json.dump(out, open(f"results/diagnose-triton-{triton.__version__}.json", "w"), indent=2)
