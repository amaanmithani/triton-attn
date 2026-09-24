"""Triton kernels: FlashAttention-2 style forward and backward.

Layout: q, k, v, o, do are [Z, H, N, D] with arbitrary strides; D is a power of two.
fp32 inputs use IEEE fp32 dots (DOT_PREC="ieee"): sm75 has no TF32, and Triton's default of TF32
fails to compile there. fp16 inputs use tensor cores regardless of DOT_PREC.

Softmax runs in base 2: scores are scaled by sm_scale * log2(e) and exponentiated with exp2,
and the forward saves M = rowmax + log2(rowsum) so the backward can rebuild P = exp2(S - M)
without storing it.
"""

import os

import triton
import triton.language as tl

LOG2E = 1.4426950408889634
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"


def _fwd_configs() -> list:
    if INTERPRET:  # the CPU interpreter is slow: one small config, no tuning
        return [triton.Config({"BLOCK_M": 16, "BLOCK_N": 16}, num_warps=1, num_stages=1)]
    # sm75: 64 KB shared memory per block, no cp.async, so few stages.
    return [
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w, num_stages=s)
        for bm in (32, 64, 128)
        for bn in (32, 64)
        for w in (4, 8)
        for s in (1, 2)
    ]


def _bwd_configs() -> list:
    if INTERPRET:
        return [triton.Config({"BLOCK_M": 16, "BLOCK_N": 16}, num_warps=1, num_stages=1)]
    return [
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w, num_stages=1)
        for bm, bn in ((32, 32), (32, 64), (64, 32), (64, 64))
        for w in (4, 8)
    ]


def _prune_smem(configs, named_args, **kwargs):
    """Drop tiles that can't fit a T4's shared memory for this head size."""
    d = named_args.get("HEAD_DIM", kwargs.get("HEAD_DIM", 64))
    q = named_args.get("Q")
    size = q.element_size() if hasattr(q, "element_size") else 2  # fp32 tiles take twice the space

    def need(c) -> int:
        return (c.kwargs["BLOCK_M"] + 2 * c.kwargs["BLOCK_N"]) * d * size

    keep = [c for c in configs if need(c) <= 48 * 1024]
    return keep or [min(configs, key=need)]  # nothing fits the budget: try the smallest tile


@triton.autotune(
    configs=_fwd_configs(),
    key=["N_BUCKET", "HEAD_DIM", "CAUSAL"],
    prune_configs_by={"early_config_prune": _prune_smem},
)
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    O,
    M,
    sm_scale,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_oz,
    stride_oh,
    stride_om,
    stride_od,
    H,
    N_CTX,
    N_BUCKET,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
    DOT_PREC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = (off_hz // H).to(tl.int64)  # 64-bit offsets: large batches overflow int32
    off_h = (off_hz % H).to(tl.int64)
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
    O += off_z * stride_oz + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    row_ok = offs_m < N_CTX

    q = tl.load(
        Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd, mask=row_ok[:, None], other=0.0
    )
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.4426950408889634

    # Causal: key blocks entirely above the diagonal contribute nothing; stop before them.
    hi = N_CTX
    if CAUSAL:
        hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX)
    for start_n in range(0, hi, BLOCK_N):
        cols = start_n + offs_n
        col_ok = cols < N_CTX
        k = tl.load(
            K + cols[None, :] * stride_kn + offs_d[:, None] * stride_kd, mask=col_ok[None, :], other=0.0
        )
        s = tl.dot(q, k, input_precision=DOT_PREC) * qk_scale
        keep = col_ok[None, :]
        if CAUSAL:
            keep = keep & (offs_m[:, None] >= cols[None, :])
        s = tl.where(keep, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.math.exp2(s - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(
            V + cols[:, None] * stride_vn + offs_d[None, :] * stride_vd, mask=col_ok[:, None], other=0.0
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision=DOT_PREC)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(M + off_hz.to(tl.int64) * N_CTX + offs_m, m_i + tl.math.log2(l_i), mask=row_ok)
    tl.store(
        O + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
        acc.to(O.dtype.element_ty),
        mask=row_ok[:, None],
    )


@triton.autotune(
    configs=_bwd_configs(),
    key=["N_BUCKET", "HEAD_DIM", "CAUSAL"],
    prune_configs_by={"early_config_prune": _prune_smem},
)
@triton.jit
def _bwd_dkdv_kernel(
    Q,
    K,
    V,
    DO,
    DK,
    DV,
    M,
    DELTA,
    sm_scale,
    stride_z,
    stride_h,
    stride_n,
    stride_d,
    H,
    N_CTX,
    N_BUCKET,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
    DOT_PREC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One program per key block: dV = P^T dO, dK = dS^T Q * sm_scale, looping over query blocks.
    All of q, k, v, do, dk, dv share one contiguous layout (the wrapper guarantees it)."""
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = (off_hz // H).to(tl.int64)
    off_h = (off_hz % H).to(tl.int64)
    base = off_z * stride_z + off_h * stride_h
    Q += base
    K += base
    V += base
    DO += base
    DK += base
    DV += base
    M += off_hz.to(tl.int64) * N_CTX
    DELTA += off_hz.to(tl.int64) * N_CTX

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    col_ok = offs_n < N_CTX
    k = tl.load(K + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d, mask=col_ok[:, None], other=0.0)
    v = tl.load(V + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d, mask=col_ok[:, None], other=0.0)
    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.4426950408889634

    # Causal: query rows before this key block never attend to it.
    lo = 0
    if CAUSAL:
        lo = (start_n * BLOCK_N // BLOCK_M) * BLOCK_M
    for start_m in range(lo, N_CTX, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        row_ok = offs_m < N_CTX
        q = tl.load(
            Q + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d, mask=row_ok[:, None], other=0.0
        )
        do = tl.load(
            DO + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d, mask=row_ok[:, None], other=0.0
        )
        m = tl.load(M + offs_m, mask=row_ok, other=0.0)
        delta = tl.load(DELTA + offs_m, mask=row_ok, other=0.0)
        s_t = tl.dot(k, tl.trans(q), input_precision=DOT_PREC) * qk_scale  # [BLOCK_N, BLOCK_M]
        keep = col_ok[:, None] & row_ok[None, :]
        if CAUSAL:
            keep = keep & (offs_m[None, :] >= offs_n[:, None])
        p_t = tl.where(keep, tl.math.exp2(s_t - m[None, :]), 0.0)
        dv += tl.dot(p_t.to(do.dtype), do, input_precision=DOT_PREC)
        dp_t = tl.dot(v, tl.trans(do), input_precision=DOT_PREC)
        ds_t = p_t * (dp_t - delta[None, :])
        dk += tl.dot(ds_t.to(q.dtype), q, input_precision=DOT_PREC)

    tl.store(
        DK + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d,
        (dk * sm_scale).to(DK.dtype.element_ty),
        mask=col_ok[:, None],
    )
    tl.store(
        DV + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d,
        dv.to(DV.dtype.element_ty),
        mask=col_ok[:, None],
    )


@triton.autotune(
    configs=_bwd_configs(),
    key=["N_BUCKET", "HEAD_DIM", "CAUSAL"],
    prune_configs_by={"early_config_prune": _prune_smem},
)
@triton.jit
def _bwd_dq_kernel(
    Q,
    K,
    V,
    DO,
    DQ,
    M,
    DELTA,
    sm_scale,
    stride_z,
    stride_h,
    stride_n,
    stride_d,
    H,
    N_CTX,
    N_BUCKET,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
    DOT_PREC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One program per query block: dQ = dS K * sm_scale, looping over key blocks."""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = (off_hz // H).to(tl.int64)
    off_h = (off_hz % H).to(tl.int64)
    base = off_z * stride_z + off_h * stride_h
    Q += base
    K += base
    V += base
    DO += base
    DQ += base
    M += off_hz.to(tl.int64) * N_CTX
    DELTA += off_hz.to(tl.int64) * N_CTX

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    row_ok = offs_m < N_CTX
    q = tl.load(Q + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d, mask=row_ok[:, None], other=0.0)
    do = tl.load(
        DO + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d, mask=row_ok[:, None], other=0.0
    )
    m = tl.load(M + offs_m, mask=row_ok, other=0.0)
    delta = tl.load(DELTA + offs_m, mask=row_ok, other=0.0)
    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.4426950408889634

    hi = N_CTX
    if CAUSAL:
        hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX)
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        col_ok = offs_n < N_CTX
        k = tl.load(
            K + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d, mask=col_ok[:, None], other=0.0
        )
        v = tl.load(
            V + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d, mask=col_ok[:, None], other=0.0
        )
        s = tl.dot(q, tl.trans(k), input_precision=DOT_PREC) * qk_scale
        keep = row_ok[:, None] & col_ok[None, :]
        if CAUSAL:
            keep = keep & (offs_m[:, None] >= offs_n[None, :])
        p = tl.where(keep, tl.math.exp2(s - m[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v), input_precision=DOT_PREC)
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(k.dtype), k, input_precision=DOT_PREC)

    tl.store(
        DQ + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d,
        (dq * sm_scale).to(DQ.dtype.element_ty),
        mask=row_ok[:, None],
    )


@triton.jit
def _bwd_preprocess(O, DO, DELTA, stride_z, stride_h, stride_n, stride_d, H, N_CTX,
                    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr):  # fmt: skip
    """DELTA[z, h, i] = sum_d O[z, h, i, d] * dO[z, h, i, d] in fp32, without fp32 copies of O and dO."""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    base = (off_hz // H).to(tl.int64) * stride_z + (off_hz % H).to(tl.int64) * stride_h
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    ok = offs_m < N_CTX
    ptr = offs_m[:, None] * stride_n + offs_d[None, :] * stride_d
    o = tl.load(O + base + ptr, mask=ok[:, None], other=0.0).to(tl.float32)
    do = tl.load(DO + base + ptr, mask=ok[:, None], other=0.0).to(tl.float32)
    tl.store(DELTA + off_hz.to(tl.int64) * N_CTX + offs_m, tl.sum(o * do, axis=1), mask=ok)
