"""Every tile configuration against the definition, bypassing the autotuner.

Under TRITON_INTERPRET=1 (CI) this covers BLOCK_M != BLOCK_N, including the causal lower bound
rounding in the dK/dV kernel. On a GPU it covers every configuration the autotuner may pick, so a
config that is only ever chosen by timing is still checked for correctness."""

import itertools

import pytest
import torch

from tattn import triton_available
from tattn.reference import naive_attention

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pytestmark = pytest.mark.skipif(not triton_available(DEV), reason="needs Triton (CUDA or TRITON_INTERPRET=1)")


def configs() -> list[dict[str, int]]:
    if DEV.type == "cpu":
        return [
            {"BLOCK_M": m, "BLOCK_N": n, "num_warps": 1, "num_stages": 1}
            for m, n in ((16, 16), (16, 32), (32, 16))
        ]
    from tattn import kernels

    seen = {}
    for c in kernels._fwd_configs() + kernels._bwd_configs():
        key = (c.kwargs["BLOCK_M"], c.kwargs["BLOCK_N"], c.num_warps, c.num_stages)
        seen[key] = {"BLOCK_M": key[0], "BLOCK_N": key[1], "num_warps": key[2], "num_stages": key[3]}
    return list(seen.values())


CASES = list(itertools.product([1, 37, 70], [False, True]))


@pytest.mark.parametrize(
    "config", configs(), ids=lambda c: "{BLOCK_M}x{BLOCK_N}w{num_warps}s{num_stages}".format(**c)
)
def test_config_forward_and_backward(config: dict[str, int]) -> None:
    from tattn.api import _backward, _forward

    for n, causal in CASES:
        g = torch.Generator().manual_seed(n)
        q, k, v, do = (torch.randn(1, 2, n, 32, generator=g).to(DEV, torch.float16) for _ in range(4))
        scale = 32**-0.5
        o, m = _forward(q, k, v, causal, scale, config=config)
        ref = naive_attention(q.float(), k.float(), v.float(), causal=causal)
        torch.testing.assert_close(o.float(), ref, atol=2e-2, rtol=2e-2)
        dq, dk, dv = _backward(q, k, v, o, m, do, causal, scale, config=config)
        qr, kr, vr = (t.float().requires_grad_() for t in (q, k, v))
        naive_attention(qr, kr, vr, causal=causal).backward(do.float())
        for got, want in zip((dq, dk, dv), (qr.grad, kr.grad, vr.grad), strict=True):
            torch.testing.assert_close(got.float(), want, atol=3e-2, rtol=3e-2)
