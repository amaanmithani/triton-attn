"""Does this Triton release emit tensor-core code on this GPU? Compiles (1) a bare fp16 matmul
kernel and (2) this repo's attention forward kernel with a fixed config, counts `mma.sync` /
`ldmatrix` / fp32 FMA instructions in the PTX, and times the matmul. Compile failures are recorded
too. Writes results/diagnose-triton-<version>.json."""

import json
import re
import traceback
from pathlib import Path

import torch
import triton
import triton.language as tl


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
        "ldmatrix": len(re.findall(r"\bldmatrix", ptx)),
        "fma_f32": len(re.findall(r"\bfma\.rn\.f32", ptx)),
    }


out: dict = {"gpu": torch.cuda.get_device_name(), "triton": triton.__version__, "torch": torch.__version__}
M = N = K = 4096
a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(K, N, device="cuda", dtype=torch.float16)
c = torch.empty(M, N, device="cuda", dtype=torch.float16)
for bm, bn, bk, w in ((64, 64, 32, 4), (128, 128, 32, 8)):
    name = f"triton_mm_{bm}x{bn}x{bk}"
    try:
        k = mm[(M // bm, N // bn)](a, b, c, M, N, K, BM=bm, BN=bn, BK=bk, num_warps=w)
        ms = triton.testing.do_bench(
            lambda bm=bm, bn=bn, bk=bk, w=w: mm[(M // bm, N // bn)](
                a, b, c, M, N, K, BM=bm, BN=bn, BK=bk, num_warps=w
            ),
            return_mode="median",
        )
        out[name] = {"tflops": 2 * M * N * K / ms / 1e9, **ptx_stats(k.asm["ptx"])}
    except Exception as e:  # compile failures are part of the finding
        out[name] = {"error": f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"}
out["torch_mm_tflops"] = 2 * M * N * K / triton.testing.do_bench(lambda: a @ b, return_mode="median") / 1e9

# The attention forward kernel itself, fixed config (bypassing the autotuner).
try:
    from tattn.kernels import _fwd_kernel

    q = torch.randn(1, 2, 1024, 64, device="cuda", dtype=torch.float16)
    o = torch.empty_like(q)
    m = torch.empty(1, 2, 1024, device="cuda", dtype=torch.float32)
    kk = _fwd_kernel.fn[(1024 // 128, 2)](
        q, q, q, o, m, 0.125, *q.stride(), *q.stride(), *q.stride(), *o.stride(), 2, 1024, 1024,
        HEAD_DIM=64, CAUSAL=False, DOT_PREC="tf32", BLOCK_M=128, BLOCK_N=64, num_warps=4, num_stages=1,
    )  # fmt: skip
    out["attention_fwd_128x64"] = ptx_stats(kk.asm["ptx"])
except Exception as e:
    out["attention_fwd_128x64"] = {"error": f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"}
    out["attention_fwd_traceback_tail"] = traceback.format_exc().splitlines()[-3:]

Path("results").mkdir(exist_ok=True)
print(json.dumps(out, indent=2))
Path(f"results/diagnose-triton-{triton.__version__}.json").write_text(json.dumps(out, indent=2) + "\n")
