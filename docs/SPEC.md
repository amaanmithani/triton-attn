# triton-attn spec

## Goal
A readable, tested FlashAttention-2 style attention in Triton, forward and backward, tuned for the
NVIDIA T4 (sm75, the free Kaggle GPU), with honest measurements against PyTorch's own attention.

## In scope
- Fused forward: tiled QK^T, online softmax (running max and sum, base-2 exponent), PV accumulation in
  fp32, causal and non-causal, fp16 inputs (fp32 also accepted), head dims 16/32/64/128, any sequence
  length (masked tails). Saves the per-row log-sum-exp for the backward pass.
- Backward: FlashAttention-2 split into a dK/dV kernel (one program per key block, loops over queries)
  and a dQ kernel (one program per query block), so no atomics. D = rowsum(dO * O) is precomputed.
- `torch.autograd.Function` wrapper, `attention(q, k, v, causal=False, sm_scale=None)`.
- Autotuned block sizes and warps for sm75 (64 KB shared memory, no bf16, no async copies).
- A pure-PyTorch tiled implementation of the same algorithm (forward and backward), used to test the
  algorithm on machines without Triton and as readable documentation.

## Out of scope
Dropout, attention bias/ALiBi, variable-length (packed) batches, GQA/MQA, bf16 (not on sm75),
FP8, Hopper features (TMA, warp specialisation), the KV-cache decode path.

## Success metrics
- Correctness: forward and gradients match a float32 reference; max abs error reported per shape.
- Performance on a T4, per sequence length (1k–16k), for forward and forward+backward:
  achieved TFLOP/s and peak memory, next to `torch.nn.functional.scaled_dot_product_attention`
  (whatever backend PyTorch picks on sm75) and the naive materialised-scores version.
  No target is promised in advance; the numbers are reported as measured.
- Every README number is produced by a committed script from committed JSON.
- Coverage >= 85 % (Triton kernels exercised under the Triton CPU interpreter in CI).
