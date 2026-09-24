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
    out = ["| impl | worst max abs error (fwd, seq ≤ 4096) | mean abs error |", "|---|---|---|"]
    for i in IMPLS:
        es = [e for e in fwd if e["impl"] == i]
        if es:
            w = max(es, key=lambda e: e["max_abs_err"])
            out.append(f"| {LABEL[i]} | {w['max_abs_err']:.2e} | {max(e['mean_abs_err'] for e in es):.2e} |")
    grads = [e for e in res["accuracy"] if "grad" in e]
    if grads:
        g = ", ".join(
            f"{e['grad']} {e['max_abs_err']:.1e} ({'causal' if e['causal'] else 'full'})" for e in grads
        )
        out.append("")
        out.append(f"tattn gradients vs float32 autograd at seq 1,024, max abs error: {g}.")
    return "\n".join(out)


def versions() -> str:
    rows = []
    for f in sorted((ROOT / "results").glob("diagnose-triton-*.json")):
        d = json.loads(f.read_text())
        mm = d.get("triton_mm_128x128x32") or {}
        small = d.get("triton_mm_64x64x32") or {}
        rows.append(
            (d["triton"], small.get("tflops"), mm.get("tflops"), small.get("mma"), d.get("torch_mm_tflops"))
        )
    if not rows:
        return ""
    torch_tf = next((r[4] for r in rows if r[4]), None)
    out = [
        "| Triton | fp16 matmul 4096³, 64×64 tile (TFLOP/s) | 128×128 tile | `mma.sync` instructions in PTX |",
        "|---|---|---|---|",
    ]
    for ver, a, b, mma, _ in sorted(rows, key=lambda r: [int(x) for x in r[0].split(".")]):
        out.append(f"| {ver} | {a:.1f} | {b:.1f} | {mma} |")
    out.append("")
    out.append(
        f"cuBLAS (`torch.matmul`) on the same GPU: {torch_tf:.1f} TFLOP/s. Triton 3.3.1 failed to compile the kernel "
        "(`Unsupported rounding mode for conversion`), so it has no row. Source: `bench/diagnose.py`, "
        "`results/diagnose-triton-*.json`. The matmul kernel in the sweep is deliberately untuned (it exists to show "
        "whether tensor-core code is emitted), so its TFLOP/s are not a reference for what Triton can reach."
    )
    return "\n".join(out)


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
    body.insert(
        0,
        f"**Summary:** against PyTorch's fastest attention on this GPU (SDPA memory-efficient; the flash backend "
        f"needs sm80), tattn is {ratios['fwd'][0]:.2f}–{ratios['fwd'][1]:.2f}× faster forward and "
        f"{ratios['fwd+bwd'][0]:.2f}–{ratios['fwd+bwd'][1]:.2f}× faster forward+backward across all lengths, "
        "causal and not, with the same memory. The forward reaches roughly the fp16 matmul rate cuBLAS gets on this "
        "power-limited card, which is higher than I expected; output correctness at every benchmarked length up to "
        "4,096 is checked in the same run (below), so it is not skipping work.\n",
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
            f"{res['env']['triton']} (`results/t4-triton-3.6.0.json`).",
        )
    (ROOT / "README.md").write_text(t)


if __name__ == "__main__":
    main()
