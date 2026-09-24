# triton-attn

FlashAttention-2 style attention in [Triton](https://github.com/triton-lang/triton), forward and backward, tuned
for the NVIDIA T4 (the free Kaggle GPU) and measured against PyTorch's own attention backends.

```python
from tattn import attention

out = attention(q, k, v, causal=True)  # q, k, v: [batch, heads, seq, dim], fp16 or fp32; autograd works
```

On a CUDA GPU the Triton kernels run; under `TRITON_INTERPRET=1` they run on the CPU (that's how CI tests them); on
anything else the same algorithm runs as a tiled PyTorch reference.

## What's inside

- **Forward** (`src/tattn/kernels.py`): one program per (query block, batch×head). It streams key/value blocks
  through shared memory with an online softmax: a running row max and sum in fp32, exponentials in base 2
  (`exp2`, scores pre-scaled by `log2(e)`), and a rescale of the accumulator when the max moves. Causal masking skips
  key blocks above the diagonal entirely. It saves one float per row, `M = max + log2(sum)`, for the backward.
- **Backward**, split the FlashAttention-2 way into two kernels so nothing needs atomics:
  - dK/dV: one program per key block, looping over query blocks.
  - dQ: one program per query block, looping over key blocks.
  - Both rebuild the probabilities from `M` instead of storing the N×N matrix. `D = rowsum(dO ∘ O)` is computed once
    up front.
- **Autotuning for sm75**: block sizes, warps and stages. Configurations that can't fit a T4's shared memory for the
  head size are pruned before tuning.
- **A tiled PyTorch reference** (`src/tattn/reference.py`) of the same algorithm. It's readable, and it makes the
  algorithm testable without Triton.

## Correctness

Tests compare the output and all three gradients with the textbook definition (materialised scores, float32
autograd) across:

- fp16 and fp32;
- causal and non-causal;
- head dims 16/32/64;
- sequence lengths that aren't multiples of the block size (1, 17, 33, 70);
- non-contiguous inputs.

CI runs them against the real Triton kernels under the Triton CPU interpreter. The Kaggle job runs them again on the
T4 before benchmarking, and keeps the log.

<!-- accuracy:start -->
| impl | worst max abs error (fwd, seq ≤ 4096) | mean abs error |
|---|---|---|
| tattn (this repo) | 1.21e-03 | 2.60e-05 |
| SDPA mem-efficient | 1.21e-03 | 2.60e-05 |
| SDPA math | 9.76e-04 | 1.82e-05 |
| naive | 9.76e-04 | 1.82e-05 |

tattn gradients vs float32 autograd at seq 1,024, max abs error: dq 2.5e-04 (full), dk 3.5e-04 (full), dv 2.3e-04 (full), dq 1.2e-03 (causal), dk 1.5e-03 (causal), dv 1.5e-03 (causal).
<!-- accuracy:end -->

## Performance on a T4

<!-- perf:start -->
**Summary:** against PyTorch's fastest attention on this GPU (SDPA memory-efficient; the flash backend needs sm80), tattn is 1.45–1.98× faster forward and 1.42–1.80× faster forward+backward across all lengths, causal and not, with the same memory. The forward reaches roughly the fp16 matmul rate cuBLAS gets on this power-limited card, which is higher than I expected; output correctness at every benchmarked length up to 4,096 is checked in the same run (below), so it is not skipping work.

Tesla T4 (sm 7.5, 14.6 GiB), torch 2.10.0+cu128, Triton 3.2.0, CUDA 12.8. fp16, head dim 64, 16 heads, batch × seq = 16,384 tokens. Timer: triton.testing.do_bench median, warmup 25 ms, rep 100 ms. FLOPs: 4*B*H*N^2*D (halved when causal); fwd+bwd = 3.5x fwd.

**Forward, non-causal: TFLOP/s**

| seq (batch) | tattn (this repo) | SDPA mem-efficient | SDPA flash | SDPA math | naive |
|---|---|---|---|---|---|
| 512 (32) | 19.9 | 10.0 | n/a | 1.0 | 1.2 |
| 1,024 (16) | 20.5 | 11.3 | n/a | 1.1 | 1.4 |
| 2,048 (8) | 21.5 | 11.8 | n/a | 1.1 | 1.4 |
| 4,096 (4) | 22.4 | 11.8 | n/a | 1.1 | 1.4 |
| 8,192 (2) | 21.9 | 12.0 | n/a | OOM | OOM |
| 16,384 (1) | 22.2 | 12.0 | n/a | OOM | OOM |

**Forward, causal: TFLOP/s**

| seq (batch) | tattn (this repo) | SDPA mem-efficient | SDPA flash | SDPA math | naive |
|---|---|---|---|---|---|
| 512 (32) | 13.2 | 9.0 | n/a | 0.5 | 0.5 |
| 1,024 (16) | 15.8 | 10.4 | n/a | 0.5 | 0.6 |
| 2,048 (8) | 17.9 | 11.0 | n/a | 0.5 | 0.6 |
| 4,096 (4) | 19.1 | 11.5 | n/a | 0.5 | 0.5 |
| 8,192 (2) | 19.9 | 11.8 | n/a | OOM | OOM |
| 16,384 (1) | 20.1 | 11.9 | n/a | OOM | OOM |

**Forward + backward, non-causal: TFLOP/s**

| seq (batch) | tattn (this repo) | SDPA mem-efficient | SDPA flash | SDPA math | naive |
|---|---|---|---|---|---|
| 512 (32) | 10.3 | 7.1 | n/a | 1.5 | 1.5 |
| 1,024 (16) | 13.0 | 8.4 | n/a | 1.6 | 1.7 |
| 2,048 (8) | 14.8 | 9.4 | n/a | 1.7 | 1.8 |
| 4,096 (4) | 15.6 | 9.5 | n/a | OOM | OOM |
| 8,192 (2) | 16.2 | 9.5 | n/a | OOM | OOM |
| 16,384 (1) | 16.6 | 9.3 | n/a | OOM | OOM |

**Forward + backward, causal: TFLOP/s**

| seq (batch) | tattn (this repo) | SDPA mem-efficient | SDPA flash | SDPA math | naive |
|---|---|---|---|---|---|
| 512 (32) | 7.4 | 5.2 | n/a | 0.8 | 0.7 |
| 1,024 (16) | 10.0 | 6.8 | n/a | 0.9 | 0.7 |
| 2,048 (8) | 12.5 | 7.9 | n/a | 0.8 | 0.8 |
| 4,096 (4) | 13.9 | 8.4 | n/a | OOM | OOM |
| 8,192 (2) | 15.1 | 8.8 | n/a | OOM | OOM |
| 16,384 (1) | 15.6 | 8.7 | n/a | OOM | OOM |

**Peak extra memory, forward + backward, non-causal (MiB)**

| seq (batch) | tattn (this repo) | SDPA mem-efficient | SDPA flash | SDPA math | naive |
|---|---|---|---|---|---|
| 512 (32) | 129 | 130 | n/a | 2,176 | 2,144 |
| 1,024 (16) | 129 | 130 | n/a | 4,224 | 4,192 |
| 2,048 (8) | 129 | 130 | n/a | 8,320 | 8,288 |
| 4,096 (4) | 129 | 130 | n/a | OOM | OOM |
| 8,192 (2) | 129 | 130 | n/a | OOM | OOM |
| 16,384 (1) | 129 | 130 | n/a | OOM | OOM |

- SDPA flash did not run on this GPU: `No available kernel. Aborting execution.`
<!-- perf:end -->

### Triton version matters on Turing

The first T4 run used Kaggle's default Triton (3.6.0). The kernels were correct but ran at a tenth of the speed.
The compiled PTX had no tensor-core instructions: every `tl.dot` had become fp32 FMAs. A bare fp16 matmul kernel
behaved the same way, so the problem was in the compiler, not these kernels. The sweep below found the last release
that still emits `mma.sync` for sm75, and every number above is measured with it (Triton 3.2.0, pinned in the Kaggle
job).

<!-- versions:start -->
| Triton | fp16 matmul 4096³, 64×64 tile (TFLOP/s) | 128×128 tile | `mma.sync` instructions in PTX |
|---|---|---|---|
| 2.3.1 | 8.0 | 12.6 | 32 |
| 3.0.0 | 7.9 | 13.9 | 32 |
| 3.1.0 | 8.2 | 14.2 | 32 |
| 3.2.0 | 8.1 | 14.0 | 32 |
| 3.6.0 | 1.4 | 0.1 | 0 |

cuBLAS (`torch.matmul`) on the same GPU: 22.4 TFLOP/s. Triton 3.3.1 failed to compile the kernel (`Unsupported rounding mode for conversion`), so it has no row. Source: `bench/diagnose.py`, `results/diagnose-triton-*.json`. The matmul kernel in the sweep is deliberately untuned (it exists to show whether tensor-core code is emitted), so its TFLOP/s are not a reference for what Triton can reach.
<!-- versions:end -->

<!-- regression:start -->
Same kernels, same GPU, Triton 3.6.0: forward at 16k tokens, non-causal, 1.3 TFLOP/s, against 22.2 with Triton 3.2.0 (`results/t4-triton-3.6.0.json`).
<!-- regression:end -->

Reproduce: `python kaggle/build.py bench 3.2.0 && uvx kaggle kernels push -p kaggle/kernel` (or `python kaggle/build.py diagnose` for the version sweep) (Kaggle account with a phone-verified
GPU quota), then `uvx kaggle kernels output amaanmithani/triton-attn-bench -p results/` and
`uv run python bench/report.py`.

## Limits

- Forward and backward for dense attention only: no dropout, bias/ALiBi, GQA/MQA, variable-length batches or KV-cache
  decoding.
- Needs Triton ≤ 3.2 for tensor cores on sm75 (see above). Newer Triton runs correctly there, but about 10× slower.
- Head dims 16/32/64/128 (powers of two). Tuned and measured only on sm75. The autotuner will pick configurations on
  other GPUs, but nothing here is measured there.
- `D = rowsum(dO ∘ O)` is a separate PyTorch op rather than a fused preprocessing kernel.
- FLOP counts are the usual convention (4·B·H·N²·D forward, halved for causal, backward counted as 2.5× forward), so
  TFLOP/s figures are comparable across implementations but aren't hardware counters.

## Development

```sh
uv sync && uv run pytest              # reference path on macOS; on Linux set TRITON_INTERPRET=1 to test the kernels
uv run ruff check . && uv run mypy
```
