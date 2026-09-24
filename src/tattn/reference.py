"""Reference implementations in plain PyTorch.

`naive_attention` materialises the full score matrix (the definition). `tiled_forward` and
`tiled_backward` run the same block-by-block algorithm as the Triton kernels (online softmax in
base 2, saved log-sum-exp, FlashAttention-2 backward), so the algorithm is testable and readable
on machines without Triton.
"""

from __future__ import annotations

import math

import torch

LOG2E = 1.4426950408889634


def naive_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False, sm_scale: float | None = None
) -> torch.Tensor:
    scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    s = (q.float() @ k.float().transpose(-1, -2)) * scale
    if causal:
        n = q.shape[-2]
        s = s.masked_fill(torch.ones(n, n, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
    return (torch.softmax(s, dim=-1) @ v.float()).to(q.dtype)


def tiled_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
    block_m: int = 16,
    block_n: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (O, M) where M = rowmax + log2(rowsum) of the base-2 scaled scores."""
    *batch, n, d = q.shape
    scale = (sm_scale if sm_scale is not None else 1.0 / math.sqrt(d)) * LOG2E
    qf, kf, vf = q.float(), k.float(), v.float()
    out = torch.empty(*batch, n, d, dtype=torch.float32, device=q.device)
    lse = torch.empty(*batch, n, dtype=torch.float32, device=q.device)
    for sm in range(0, n, block_m):
        rows = torch.arange(sm, min(sm + block_m, n), device=q.device)
        qb = qf[..., rows, :]
        m_i = torch.full((*batch, len(rows)), float("-inf"), device=q.device)
        l_i = torch.zeros(*batch, len(rows), device=q.device)
        acc = torch.zeros(*batch, len(rows), d, device=q.device)
        hi = min(sm + block_m, n) if causal else n
        for sn in range(0, hi, block_n):
            cols = torch.arange(sn, min(sn + block_n, n), device=q.device)
            s = (qb @ kf[..., cols, :].transpose(-1, -2)) * scale
            if causal:
                s = s.masked_fill(rows[:, None] < cols[None, :], float("-inf"))
            m_new = torch.maximum(m_i, s.amax(-1))
            p = torch.exp2(s - m_new[..., None])
            alpha = torch.exp2(m_i - m_new)
            l_i = l_i * alpha + p.sum(-1)
            acc = acc * alpha[..., None] + p @ vf[..., cols, :]
            m_i = m_new
        out[..., rows, :] = acc / l_i[..., None]
        lse[..., rows] = m_i + torch.log2(l_i)
    return out.to(q.dtype), lse


def tiled_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
    block_m: int = 16,
    block_n: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FlashAttention-2 backward: dK/dV per key block, dQ per query block; P is rebuilt from lse."""
    *_, n, d = q.shape
    sm = sm_scale if sm_scale is not None else 1.0 / math.sqrt(d)
    scale = sm * LOG2E
    qf, kf, vf, dof = q.float(), k.float(), v.float(), do.float()
    delta = (dof * o.float()).sum(-1)
    dq = torch.zeros_like(qf)
    dk = torch.zeros_like(kf)
    dv = torch.zeros_like(vf)
    for sn in range(0, n, block_n):
        cols = torch.arange(sn, min(sn + block_n, n), device=q.device)
        lo = (sn // block_m) * block_m if causal else 0
        for sm_ in range(lo, n, block_m):
            rows = torch.arange(sm_, min(sm_ + block_m, n), device=q.device)
            s = (qf[..., rows, :] @ kf[..., cols, :].transpose(-1, -2)) * scale
            p = torch.exp2(s - lse[..., rows, None])
            if causal:
                p = p.masked_fill(rows[:, None] < cols[None, :], 0.0)
            dv[..., cols, :] += p.transpose(-1, -2) @ dof[..., rows, :]
            dp = dof[..., rows, :] @ vf[..., cols, :].transpose(-1, -2)
            ds = p * (dp - delta[..., rows, None])
            dk[..., cols, :] += ds.transpose(-1, -2) @ qf[..., rows, :] * sm
            dq[..., rows, :] += ds @ kf[..., cols, :] * sm
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)
