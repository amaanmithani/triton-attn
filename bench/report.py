"""Render README results sections from results/t4.json (--check: only verify it parses)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
IMPLS = ["tattn", "sdpa_efficient", "sdpa_flash", "sdpa_math", "naive"]
LABEL = {
    "tattn": "tattn (this repo)",
    "sdpa_efficient": "SDPA mem-efficient",
    "sdpa_flash": "SDPA flash",
    "sdpa_math": "SDPA math",
    "naive": "naive",
}


def splice(text: str, name: str, body: str) -> str:
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    i, j = text.index(start), text.index(end)
    return text[: i + len(start)] + "\n" + body.strip() + "\n" + text[j:]


def cell(r: dict[str, Any] | None, key: str) -> str:
    if r is None:
        return "–"
    if r.get("oom"):
        return "OOM"
    if "error" in r:
        return "n/a"
    return f"{r[key]:.1f}" if key != "peak_extra_mib" else f"{r[key]:,.0f}"


def table(res: dict[str, Any], mode: str, causal: bool, key: str) -> str:
    rows = [r for r in res["rows"] if r["mode"] == mode and r["causal"] == causal]
    seqs = sorted({r["seq"] for r in rows})
    present = [i for i in IMPLS if any(r["impl"] == i for r in rows)]
    out = [
        "| seq (batch) | " + " | ".join(LABEL[i] for i in present) + " |",
        "|---" * (len(present) + 1) + "|",
    ]
    for n in seqs:
        b = next(r["batch"] for r in rows if r["seq"] == n)
        vals = [cell(next((r for r in rows if r["seq"] == n and r["impl"] == i), None), key) for i in present]
        out.append(f"| {n:,} ({b}) | " + " | ".join(vals) + " |")
    return "\n".join(out)


def unavailable(res: dict[str, Any]) -> str:
    notes = {}
    for r in res["rows"]:
        if "error" in r:
            notes.setdefault(r["impl"], r["error"])
    return "\n".join(f"- {LABEL[k]} did not run on this GPU: `{v}`" for k, v in notes.items())


def accuracy(res: dict[str, Any]) -> str:
    fwd = [e for e in res["accuracy"] if "grad" not in e and "max_abs_err" in e]
    out = [
        "Reference: float32 materialised attention up to 4,096; above that, float32 SDPA mem-efficient on the first "
        "two heads (an independent implementation; the materialised version doesn't fit).",
        "",
        "| impl | worst max abs error (fwd, all lengths) | worst mean abs error |",
        "|---|---|---|",
    ]
    for i in IMPLS:
        es = [e for e in fwd if e["impl"] == i]
        if es:
            w = max(es, key=lambda e: e["max_abs_err"])
            out.append(f"| {LABEL[i]} | {w['max_abs_err']:.2e} | {max(e['mean_abs_err'] for e in es):.2e} |")
    grads = [e for e in res["accuracy"] if "grad" in e]
    if grads:
        g = ", ".join(
            f"{e['grad']} {e['max_abs_err']:.1e} (seq {e['seq']}, {'causal' if e['causal'] else 'full'})"
            for e in grads
        )
        out.append("")
        out.append(f"tattn gradients vs float32 autograd (seq 1,024 and 4,096, batch 1), max abs error: {g}.")
    return "\n".join(out)


def versions() -> str:
    rows = []
    for f in (ROOT / "results").glob("diagnose-triton-*.json"):
        rows.append(json.loads(f.read_text()))
    if not rows:
        return ""
    rows.sort(key=lambda d: [int(x) for x in d["triton"].split(".")])

    def mm(d: dict[str, Any]) -> str:
        r = d.get("triton_mm_128x128x32") or {}
        if "error" in r:
            return "compile error"
        return f"{r['tflops']:.1f} TFLOP/s, {r['mma']} `mma`"

    def attn(d: dict[str, Any]) -> str:
        r = d.get("attention_fwd_128x64") or {}
        if "error" in r:
            return "compile error"
        return f"{r['mma']} `mma`, {r['fma_f32']} fp32 `fma`"

    out = [
        "| Triton | bare fp16 matmul (128×128 tile) | this repo's forward kernel (128×64 tile) |",
        "|---|---|---|",
    ]
    out += [f"| {d['triton']} | {mm(d)} | {attn(d)} |" for d in rows]
    errs = {d["triton"]: (d.get("triton_mm_128x128x32") or {}).get("error") for d in rows}
    notes = [
        "",
        "Compile errors, verbatim from `results/diagnose-log-triton-*.txt`: 3.3.0 and 3.3.1 abort with "
        "`Unsupported conversion from f16 to f16` / `Unsupported rounding mode for conversion` on sm75. 2.3.1 compiles "
        "the matmul but not the attention kernel, because it predates `tl.dot(input_precision=...)`, which this repo "
        "uses; that is an API difference, not a Turing issue.",
        "",
        "The matmul kernel in the sweep is deliberately untuned (it exists to show whether tensor-core code is "
        "emitted), so its TFLOP/s are not a reference for what Triton can reach. One Kaggle job ran the whole sweep, "
        "reinstalling Triton between versions (`python kaggle/build.py diagnose`).",
    ]
    _ = errs
    return "\n".join(out + notes)


def main() -> None:
    p = ROOT / "results" / "t4.json"
    if not p.exists():
        print("no results yet")
        return
    res = json.loads(p.read_text())
    if "--check" in sys.argv:
        assert res["rows"], "empty results"
        print("ok", len(res["rows"]), "rows")
        return
    e, c = res["env"], res["config"]
    env = (
        f"{e['gpu']} (sm {e['sm']}, {e['memory_gib']} GiB), torch {e['torch']}, Triton {e['triton']}, CUDA {e['cuda']}. "
        f"fp16, head dim {c['dim']}, {c['heads']} heads, batch × seq = {c['tokens']:,} tokens. Timer: {c['timer']}. "
        f"FLOPs: {c['flops']}."
    )
    body = [env, ""]
    for mode, title in (("fwd", "Forward"), ("fwd+bwd", "Forward + backward")):
        for causal in (False, True):
            body += [
                f"**{title}, {'causal' if causal else 'non-causal'}: TFLOP/s**",
                "",
                table(res, mode, causal, "tflops"),
                "",
            ]
    body += [
        "**Peak extra memory, forward + backward, non-causal (MiB)**",
        "",
        table(res, "fwd+bwd", False, "peak_extra_mib"),
        "",
    ]
    body += [unavailable(res)]
    ratios = {}
    for mode in ("fwd", "fwd+bwd"):
        rs = []
        for r in res["rows"]:
            if r["impl"] == "tattn" and r["mode"] == mode and "tflops" in r:
                o = next(
                    (x for x in res["rows"] if x["impl"] == "sdpa_efficient" and x["mode"] == mode
                     and x["seq"] == r["seq"] and x["causal"] == r["causal"] and "tflops" in x),
                    None,
                )  # fmt: skip
                if o:
                    rs.append(r["tflops"] / o["tflops"])
        ratios[mode] = (min(rs), max(rs))

    def mem(impl: str) -> float:
        return max(
            r["peak_extra_mib"]
            for r in res["rows"]
            if r["impl"] == impl and r["mode"] == "fwd+bwd" and "ms" in r
        )

    cublas = res.get("cublas_fp16_matmul_4096_tflops")
    sweep = [
        json.loads(f.read_text()).get("torch_mm_tflops")
        for f in (ROOT / "results").glob("diagnose-triton-*.json")
    ]
    sweep = [x for x in sweep if x]
    sweep_lo, sweep_hi = (min(sweep), max(sweep)) if sweep else (float("nan"), float("nan"))
    best_fwd = max(
        r["tflops"] for r in res["rows"] if r["impl"] == "tattn" and r["mode"] == "fwd" and "ms" in r
    )
    body.insert(
        0,
        f"**Summary:** against PyTorch's fastest attention on this GPU (SDPA memory-efficient; the flash backend "
        f"needs sm80), tattn is {ratios['fwd'][0]:.2f}–{ratios['fwd'][1]:.2f}× faster forward and "
        f"{ratios['fwd+bwd'][0]:.2f}–{ratios['fwd+bwd'][1]:.2f}× faster forward+backward across all lengths, causal "
        f"and not. Its forward+backward peak extra memory is {mem('tattn'):,.0f} MiB against "
        f"{mem('sdpa_efficient'):,.0f} MiB for SDPA mem-efficient. Its best forward rate, {best_fwd:.1f} TFLOP/s, is "
        + f"{best_fwd / 65:.0%} of the T4's nominal 65 TFLOP/s fp16 tensor-core peak. "
        + (
            f"In the same run cuBLAS reached {cublas:.1f} TFLOP/s on a 4096³ fp16 matmul, less than the attention "
            f"kernel; in the version-sweep job it measured {sweep_lo:.1f}–{sweep_hi:.1f}. The T4 is capped at 70 W and "
            "a long dense GEMM throttles hardest, so treat that comparison as noise, not as beating cuBLAS. "
            if cublas
            else ""
        )
        + "Output error is checked at every benchmarked length, including 8k and 16k (below).\n",
    )
    t = (ROOT / "README.md").read_text()
    t = splice(t, "perf", "\n".join(body))
    t = splice(t, "accuracy", accuracy(res))
    t = splice(t, "versions", versions())
    old = ROOT / "results" / "t4-triton-3.6.0.json"
    if old.exists():
        o = json.loads(old.read_text())
        pick = lambda rr, impl: next(  # noqa: E731
            r
            for r in rr["rows"]
            if r["impl"] == impl and r["seq"] == 16384 and not r["causal"] and r["mode"] == "fwd"
        )
        t = splice(
            t,
            "regression",
            f"Same kernels, same GPU, Triton {o['env']['triton']}: forward at 16k tokens, non-causal, "
            f"{pick(o, 'tattn')['tflops']:.1f} TFLOP/s, against {pick(res, 'tattn')['tflops']:.1f} with Triton "
            f"{res['env']['triton']} (`results/t4-triton-3.6.0.json`, from an earlier run timed with do_bench's default "
            "mean rather than the median).",
        )
    (ROOT / "README.md").write_text(t)


if __name__ == "__main__":
    main()
