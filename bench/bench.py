"""T4 benchmark: tattn vs PyTorch attention, forward and forward+backward.

For each sequence length (fixed tokens per batch, head dim 64, fp16), measures median time
(triton.testing.do_bench), achieved TFLOP/s and peak extra memory for:
  - tattn (this repo's Triton kernels, autotuned)
  - SDPA with each backend PyTorch will run on this GPU (flash / efficient / math), individually
  - naive materialised attention (the definition; runs out of memory at long sequences)
plus max abs error of each against a float32 reference, and the autotuner's chosen configs.
Writes results/<gpu>.json. Needs a CUDA GPU.
"""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import triton
from torch.nn.attention import SDPBackend, sdpa_kernel

from tattn import attention
from tattn.kernels import _bwd_dkdv_kernel, _bwd_dq_kernel, _fwd_kernel
from tattn.reference import naive_attention


def flops(b: int, h: int, n: int, d: int, causal: bool, mode: str) -> float:
    f = 4.0 * b * h * n * n * d  # QK^T and PV, 2 flops per MAC
    if causal:
        f /= 2
    return f * (3.5 if mode == "fwd+bwd" else 1.0)  # backward ~2.5x forward


def measure(fn: Callable[[], Any]) -> dict[str, Any]:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    try:
        fn()  # warm-up (and autotune)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        fn()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base
        ms = triton.testing.do_bench(fn, warmup=25, rep=100)
        return {"ms": ms, "peak_extra_mib": peak / 2**20}
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"oom": True}
    except RuntimeError as e:  # backend not available for this GPU / shape
        torch.cuda.empty_cache()
        return {"error": str(e).splitlines()[0][:200]}


def impls() -> dict[str, Callable[..., torch.Tensor]]:
    def sdpa(backend: SDPBackend) -> Callable[..., torch.Tensor]:
        def run(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
            with sdpa_kernel(backend):
                return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

        return run

    return {
        "tattn": lambda q, k, v, causal: attention(q, k, v, causal=causal),
        "sdpa_flash": sdpa(SDPBackend.FLASH_ATTENTION),
        "sdpa_efficient": sdpa(SDPBackend.EFFICIENT_ATTENTION),
        "sdpa_math": sdpa(SDPBackend.MATH),
        "naive": lambda q, k, v, causal: naive_attention(q, k, v, causal=causal),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="512,1024,2048,4096,8192,16384")
    ap.add_argument("--tokens", type=int, default=16384, help="batch * seq held constant")
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for n in [int(x) for x in a.seqs.split(",")]:
        b = max(1, a.tokens // n)
        for causal in (False, True):
            g = torch.Generator(device=dev).manual_seed(n)
            q, k, v = (
                torch.randn(b, a.heads, n, a.dim, device=dev, dtype=torch.float16, generator=g)
                for _ in range(3)
            )
            do = torch.randn_like(q)
            for name, f in impls().items():
                for mode in ("fwd", "fwd+bwd"):
                    if mode == "fwd":

                        def fn(f: Callable[..., torch.Tensor] = f, q=q, k=k, v=v, causal=causal) -> None:  # type: ignore[no-untyped-def]
                            f(q, k, v, causal)
                    else:
                        qg, kg, vg = (t.detach().clone().requires_grad_() for t in (q, k, v))

                        def fn(  # type: ignore[no-untyped-def]
                            f: Callable[..., torch.Tensor] = f, qg=qg, kg=kg, vg=vg, do=do, causal=causal
                        ) -> None:
                            qg.grad = kg.grad = vg.grad = None
                            f(qg, kg, vg, causal).backward(do)

                    r = measure(fn)
                    if "ms" in r:
                        r["tflops"] = flops(b, a.heads, n, a.dim, causal, mode) / (r["ms"] * 1e-3) / 1e12
                    rows.append({"impl": name, "seq": n, "batch": b, "causal": causal, "mode": mode, **r})
                    print(json.dumps(rows[-1]), flush=True)
            # Accuracy against float32 at lengths where the float32 reference fits.
            if n <= 4096:
                ref = naive_attention(q.float(), k.float(), v.float(), causal=causal)
                for name, f in impls().items():
                    try:
                        out = f(q, k, v, causal).float()
                        errors.append(
                            {"impl": name, "seq": n, "causal": causal,
                             "max_abs_err": (out - ref).abs().max().item(),
                             "mean_abs_err": (out - ref).abs().mean().item()}
                        )  # fmt: skip
                    except (RuntimeError, torch.OutOfMemoryError) as e:
                        errors.append(
                            {"impl": name, "seq": n, "causal": causal, "error": str(e).splitlines()[0][:120]}
                        )
                del ref
            # Gradient accuracy for tattn vs float32 autograd at a moderate length.
            if n == 1024:
                qs, ks, vs = (t[:1].detach().clone() for t in (q, k, v))
                dos = do[:1]
                qg, kg, vg = (t.clone().requires_grad_() for t in (qs, ks, vs))
                attention(qg, kg, vg, causal=causal).backward(dos)
                qr, kr, vr = (t.float().requires_grad_() for t in (qs, ks, vs))
                naive_attention(qr, kr, vr, causal=causal).backward(dos.float())
                for label, got, want in (
                    ("dq", qg.grad, qr.grad),
                    ("dk", kg.grad, kr.grad),
                    ("dv", vg.grad, vr.grad),
                ):
                    errors.append(
                        {"impl": "tattn", "seq": n, "causal": causal, "grad": label,
                         "max_abs_err": (got.float() - want).abs().max().item()}
                    )  # fmt: skip
            del q, k, v, do
    tuned = {
        kname: {str(key): str(cfg) for key, cfg in kern.cache.items()}
        for kname, kern in (("fwd", _fwd_kernel), ("bwd_dkdv", _bwd_dkdv_kernel), ("bwd_dq", _bwd_dq_kernel))
    }
    result = {
        "env": {
            "gpu": props.name,
            "sm": f"{props.major}.{props.minor}",
            "memory_gib": round(props.total_memory / 2**30, 1),
            "torch": torch.__version__,
            "triton": triton.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
        },
        "config": {
            "tokens": a.tokens,
            "heads": a.heads,
            "dim": a.dim,
            "dtype": "float16",
            "timer": "triton.testing.do_bench median, warmup 25 ms, rep 100 ms",
            "flops": "4*B*H*N^2*D (halved when causal); fwd+bwd = 3.5x fwd",
        },  # fmt: skip
        "rows": rows,
        "accuracy": errors,
        "autotune": tuned,
    }
    out = Path(a.out or f"results/{props.name.replace(' ', '_').lower()}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
