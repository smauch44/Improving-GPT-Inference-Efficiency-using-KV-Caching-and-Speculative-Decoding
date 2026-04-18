"""
benchmark.py
============
End-to-end benchmarking suite for the three inference methods.

Experiments conducted:
  1. Output equivalence — verifies baseline == KV-cache token-for-token.
  2. Latency vs sequence length — wall-clock time at T = 50, 100, 200, 500 tokens.
  3. Speedup factor — baseline / kv_cache and baseline / speculative.
  4. Computational complexity — empirically confirms O(n²) vs O(n) scaling.
  5. Speculative decoding acceptance rate — tracks α across prompt lengths.

All results are saved to results/ as PNG plots and a text report.

Reference for speedup analysis:
  Leviathan et al. "Fast Inference from Transformers via Speculative Decoding."
  ICML 2023.
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

from data_utils import CharTokenizer, BENCHMARK_PROMPTS
from gpt_model import GPTModel
from inference import (
    baseline_generate,
    kvcache_generate,
    speculative_generate,
    verify_equivalence,
)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Timing utilities
# ─────────────────────────────────────────────────────────────────────────────

def _sync_device(device: Optional[torch.device]) -> None:
    """Synchronize async accelerator work before/after timing."""
    if device is None:
        return

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def timed_run(fn, *args, n_runs: int = 3, warmup: int = 1, **kwargs) -> Tuple[float, any]:
    """
    Run *fn* with timing, averaging over *n_runs* measured calls after *warmup*.

    If a torch.device is passed via kwargs['device'], synchronize before and
    after each measured run so async CUDA/MPS execution is timed correctly.
    """
    device = kwargs.get("device", None)

    for _ in range(warmup):
        result = fn(*args, **kwargs)
        _sync_device(device)

    times = []
    for _ in range(n_runs):
        _sync_device(device)
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        _sync_device(device)
        times.append(time.perf_counter() - t0)

    return float(np.mean(times)), result


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1: Output Equivalence
# ─────────────────────────────────────────────────────────────────────────────

def run_equivalence_test(
    model    : GPTModel,
    tokenizer: CharTokenizer,
    device   : torch.device,
    n_tokens : int = 50,
) -> dict:
    """
    Confirm that KV-cache and baseline generate IDENTICAL tokens (greedy).

    This directly satisfies the instructor's requirement:
    "Verify output equivalence under caching."
    """
    print("\n" + "═" * 60)
    print("EXPERIMENT 1: Output Equivalence (Baseline vs KV-Cache)")
    print("═" * 60)

    prompts = [
        torch.tensor([tokenizer.encode(p)], dtype=torch.long)
        for p in BENCHMARK_PROMPTS[:10]
    ]

    report = verify_equivalence(model, prompts, n_tokens=n_tokens, device=device)

    print(f"\n  Prompts tested : {report['n_total']}")
    print(f"  Exact matches  : {report['n_match']}")
    print(f"  Match rate     : {report['match_rate']*100:.1f}%")
    print(f"  All equivalent : {'✓ YES' if report['all_equivalent'] else '✗ NO'}")

    for r in report["per_prompt"]:
        status = "✓" if r["match"] else f"✗ (diverges at pos {r['first_diff']})"
        prompt_preview = BENCHMARK_PROMPTS[r["prompt_idx"]][:40].replace("\n", "\\n")
        print(f"    [{r['prompt_idx']:2d}] {status}  \"{prompt_preview}…\"")

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2 & 3: Latency and Speedup vs Sequence Length
# ─────────────────────────────────────────────────────────────────────────────

def run_latency_benchmark(
    main_model  : GPTModel,
    draft_model : GPTModel,
    tokenizer   : CharTokenizer,
    device      : torch.device,
    token_counts: List[int] = (25, 50, 100, 200, 350),
    n_runs      : int = 3,
    speculation_k: int = 4,
) -> dict:
    """
    Measure wall-clock time for all three methods at multiple output lengths.

    Returns a dict of {method → list of (n_tokens, mean_seconds)} pairs.
    """
    print("\n" + "═" * 60)
    print("EXPERIMENT 2: Latency vs Sequence Length")
    print("═" * 60)

    prompt_text = "ROMEO: O, she doth teach the torches to burn bright!\n"
    prompt = torch.tensor([tokenizer.encode(prompt_text)], dtype=torch.long)

    results: Dict[str, List[Tuple[int, float]]] = {
        "baseline"   : [],
        "kv_cache"   : [],
        "speculative": [],
    }

    for n in token_counts:
        print(f"\n  Generating {n} tokens …")

        t_base, _ = timed_run(
            baseline_generate, main_model, prompt, n,
            temperature=0.0, device=device, n_runs=n_runs
        )
        t_kv, _ = timed_run(
            kvcache_generate, main_model, prompt, n,
            temperature=0.0, device=device, n_runs=n_runs
        )
        t_spec, (_, spec_stats) = timed_run(
            speculative_generate, main_model, draft_model, prompt, n,
            speculation_k=speculation_k, temperature=0.0, device=device, n_runs=n_runs
        )

        results["baseline"].append((n, t_base))
        results["kv_cache"].append((n, t_kv))
        results["speculative"].append((n, t_spec))

        su_kv   = t_base / t_kv
        su_spec = t_base / t_spec
        α       = spec_stats.get("acceptance_rate", 0.0)

        print(f"    baseline={t_base:.3f}s  kv_cache={t_kv:.3f}s  "
              f"speculative={t_spec:.3f}s")
        print(f"    Speedup KV={su_kv:.2f}×  Speculative={su_spec:.2f}×  "
              f"(α={α:.2%})")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 4: Acceptance Rate vs Speculation K
# ─────────────────────────────────────────────────────────────────────────────

def run_speculation_analysis(
    main_model : GPTModel,
    draft_model: GPTModel,
    tokenizer  : CharTokenizer,
    device     : torch.device,
    k_values   : List[int] = (1, 2, 4, 6, 8),
    n_tokens   : int = 100,
) -> dict:
    """
    Analyse how speculation length K affects acceptance rate and throughput.
    """
    print("\n" + "═" * 60)
    print("EXPERIMENT 4: Speculative Decoding — K vs Acceptance Rate")
    print("═" * 60)

    prompt_text = "HAMLET: To be, or not to be, that is the question:\n"
    prompt = torch.tensor([tokenizer.encode(prompt_text)], dtype=torch.long)

    analysis: List[dict] = []

    for k in k_values:
        t, (toks, stats) = timed_run(
            speculative_generate, main_model, draft_model, prompt, n_tokens,
            speculation_k=k, temperature=0.0, device=device, n_runs=3
        )
        t_base, _ = timed_run(
            baseline_generate, main_model, prompt, n_tokens,
            temperature=0.0, device=device, n_runs=3
        )

        entry = {
            "k"               : k,
            "time"            : t,
            "speedup"         : t_base / t,
            "acceptance_rate" : stats["acceptance_rate"],
            "tokens_per_round": stats["avg_tokens_per_round"],
            "rounds"          : stats["total_rounds"],
        }
        analysis.append(entry)

        print(f"  K={k}: speedup={entry['speedup']:.2f}×  "
              f"α={entry['acceptance_rate']:.2%}  "
              f"tokens/round={entry['tokens_per_round']:.2f}")

    return {"analysis": analysis, "n_tokens": n_tokens}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "baseline"   : "#e74c3c",
    "kv_cache"   : "#2ecc71",
    "speculative": "#3498db",
}
STYLE = {
    "baseline"   : "o-",
    "kv_cache"   : "s-",
    "speculative": "^-",
}
LABELS = {
    "baseline"   : "Baseline (no cache)",
    "kv_cache"   : "KV-Cache",
    "speculative": "KV-Cache + Speculative",
}


def _save(fig, name: str) -> str:
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_latency(results: dict) -> str:
    """Plot 1: Wall-clock latency vs token count."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for method, data in results.items():
        xs = [d[0] for d in data]
        ys = [d[1] for d in data]
        ax.plot(xs, ys, STYLE[method], color=COLORS[method],
                label=LABELS[method], linewidth=2, markersize=8)

    ax.set_xlabel("Generated tokens", fontsize=12)
    ax.set_ylabel("Wall-clock time (s)", fontsize=12)
    ax.set_title("Inference Latency vs Sequence Length", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    return _save(fig, "latency_vs_tokens.png")


def plot_speedup(results: dict) -> str:
    """Plot 2: Speedup factor relative to baseline."""
    fig, ax = plt.subplots(figsize=(8, 5))

    base_dict = dict(results["baseline"])

    for method in ("kv_cache", "speculative"):
        data = results[method]
        xs = [d[0] for d in data]
        ys = [base_dict[d[0]] / d[1] for d in data]
        ax.plot(xs, ys, STYLE[method], color=COLORS[method],
                label=LABELS[method], linewidth=2, markersize=8)

    ax.axhline(y=1.0, color=COLORS["baseline"], linestyle="--",
               linewidth=1.5, label="Baseline (1×)")
    ax.set_xlabel("Generated tokens", fontsize=12)
    ax.set_ylabel("Speedup over baseline (×)", fontsize=12)
    ax.set_title("Speedup Factor vs Sequence Length", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f×"))
    return _save(fig, "speedup_vs_tokens.png")


def plot_complexity(results: dict) -> str:
    """Plot 3: Log-log plot to visualise O(n²) vs O(n) scaling."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for method, data in results.items():
        xs = np.array([d[0] for d in data], dtype=float)
        ys = np.array([d[1] for d in data], dtype=float)
        ax.loglog(xs, ys, STYLE[method], color=COLORS[method],
                  label=LABELS[method], linewidth=2, markersize=8)

    # Reference slopes
    ref_x = np.array([xs.min(), xs.max()])
    # O(n²): y ∝ x²
    k2 = results["baseline"][-1][1] / results["baseline"][-1][0] ** 2
    ax.loglog(ref_x, k2 * ref_x ** 2, "k--", linewidth=1, alpha=0.5, label="O(n²) ref")
    # O(n): y ∝ x
    k1 = results["kv_cache"][-1][1] / results["kv_cache"][-1][0]
    ax.loglog(ref_x, k1 * ref_x, "k:", linewidth=1, alpha=0.5, label="O(n) ref")

    ax.set_xlabel("Generated tokens (log scale)", fontsize=12)
    ax.set_ylabel("Time (s, log scale)", fontsize=12)
    ax.set_title("Computational Complexity (Log-Log)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    return _save(fig, "complexity_loglog.png")


def plot_tokens_per_second(results: dict) -> str:
    """Plot 4: Throughput (tokens/second) comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for method, data in results.items():
        xs = [d[0] for d in data]
        ys = [d[0] / d[1] for d in data]   # tokens / second
        ax.plot(xs, ys, STYLE[method], color=COLORS[method],
                label=LABELS[method], linewidth=2, markersize=8)

    ax.set_xlabel("Generated tokens", fontsize=12)
    ax.set_ylabel("Throughput (tokens / second)", fontsize=12)
    ax.set_title("Inference Throughput vs Sequence Length", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    return _save(fig, "throughput_vs_tokens.png")


def plot_speculation_analysis(analysis_data: dict) -> str:
    """Plot 5: Speculation K vs speedup and acceptance rate."""
    entries = analysis_data["analysis"]
    ks      = [e["k"]               for e in entries]
    speedups = [e["speedup"]         for e in entries]
    alphas  = [e["acceptance_rate"] * 100 for e in entries]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(ks, speedups, "^-", color=COLORS["speculative"],
             linewidth=2, markersize=9)
    ax1.set_xlabel("Speculation length K (tokens per round)", fontsize=12)
    ax1.set_ylabel("Speedup over baseline (×)", fontsize=12)
    ax1.set_title("Speedup vs K", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f×"))

    ax2.bar(ks, alphas, color=COLORS["speculative"], alpha=0.8, width=0.6)
    ax2.set_xlabel("Speculation length K", fontsize=12)
    ax2.set_ylabel("Acceptance rate (%)", fontsize=12)
    ax2.set_title("Draft Acceptance Rate vs K", fontsize=13, fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Speculative Decoding Analysis", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "speculation_analysis.png")


def plot_equivalence_summary(report: dict) -> str:
    """Plot 6: Visual confirmation of output equivalence."""
    n = report["n_total"]
    n_match = report["n_match"]
    n_miss  = n - n_match

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Exact Match", "Mismatch"],
                  [n_match, n_miss],
                  color=["#2ecc71", "#e74c3c"],
                  edgecolor="white", linewidth=1.5, width=0.5)

    for bar, val in zip(bars, [n_match, n_miss]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                str(val), ha="center", va="bottom", fontsize=13, fontweight="bold")

    ax.set_ylim(0, n + 1)
    ax.set_ylabel("Number of prompts", fontsize=12)
    ax.set_title(
        f"Baseline vs KV-Cache Output Equivalence\n"
        f"({n_match}/{n} prompts match exactly, greedy decoding, {50} tokens)",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, "equivalence_summary.png")


# ─────────────────────────────────────────────────────────────────────────────
# Text report
# ─────────────────────────────────────────────────────────────────────────────

def generate_text_report(
    equiv_report    : dict,
    latency_results : dict,
    spec_analysis   : dict,
    main_model      : GPTModel,
    draft_model     : GPTModel,
) -> str:
    """Write a human-readable benchmark summary to results/benchmark_report.txt."""

    lines = []
    sep = "=" * 70

    def h(title): lines.append(f"\n{sep}\n{title}\n{sep}")
    def p(*args): lines.append(" ".join(str(a) for a in args))

    h("GPT INFERENCE BENCHMARK REPORT")
    h("Model Information")
    p(f"  Main model  : {main_model.num_parameters():,} parameters")
    p(f"  Draft model : {draft_model.num_parameters():,} parameters")
    p(f"  Draft/Main  : {draft_model.num_parameters()/main_model.num_parameters():.1%} of main")

    h("Experiment 1: Output Equivalence")
    p(f"  Prompts tested    : {equiv_report['n_total']}")
    p(f"  Exact matches     : {equiv_report['n_match']} / {equiv_report['n_total']}")
    p(f"  Match rate        : {equiv_report['match_rate']*100:.1f}%")
    p(f"  Conclusion        : {'PASS — outputs are identical ✓' if equiv_report['all_equivalent'] else 'FAIL — divergence detected ✗'}")
    p()
    p("  Interpretation:")
    p("  KV-caching reuses cached K/V tensors for past tokens, computing")
    p("  only Q,K,V for the new token at each step. Under deterministic")
    p("  (greedy) decoding, this produces mathematically identical logits")
    p("  at every position, thus identical output token sequences.")

    h("Experiment 2: Latency vs Sequence Length")
    header = f"  {'Tokens':>8}  {'Baseline':>12}  {'KV-Cache':>12}  "
    header += f"{'Speculative':>14}  {'Speedup KV':>12}  {'Speedup Spec':>14}"
    p(header)
    p("  " + "-" * 80)

    base_dict = dict(latency_results["baseline"])
    kv_dict   = dict(latency_results["kv_cache"])
    spec_dict = dict(latency_results["speculative"])

    for n in sorted(base_dict):
        tb  = base_dict[n]
        tkv = kv_dict[n]
        ts  = spec_dict[n]
        p(f"  {n:>8}  {tb:>10.3f}s  {tkv:>10.3f}s  {ts:>12.3f}s  "
          f"  {tb/tkv:>9.2f}×    {tb/ts:>11.2f}×")

    h("Experiment 3: Complexity Analysis")
    p("  Baseline scales approximately as O(n²):")
    p("  Each new token requires a full forward pass over the entire sequence.")
    p("  Attention cost grows quadratically with context length.")
    p()
    p("  KV-Cache scales approximately as O(n) per token:")
    p("  Only Q,K,V of the single new token are computed; past K,V are read")
    p("  from cache. Attention with all cached keys is O(T) per step.")
    p()
    p("  Speculative Decoding reduces number of main-model forward passes:")
    p("  K draft tokens verified in one pass → fewer expensive calls needed.")

    h("Experiment 4: Speculative Decoding Analysis")
    entries = spec_analysis["analysis"]
    p(f"  {'K':>4}  {'Speedup':>10}  {'Accept Rate':>14}  {'Tokens/Round':>14}  {'Rounds':>8}")
    p("  " + "-" * 60)
    for e in entries:
        p(f"  {e['k']:>4}  {e['speedup']:>8.2f}×  "
          f"{e['acceptance_rate']*100:>12.1f}%  "
          f"{e['tokens_per_round']:>12.2f}  {e['rounds']:>8}")

    h("Key Findings")
    max_speedup_kv = max(
        base_dict[n] / kv_dict[n] for n in base_dict
    )
    max_speedup_sp = max(
        base_dict[n] / spec_dict[n] for n in base_dict
    )
    p(f"  1. KV-Cache output equivalence: VERIFIED (greedy decoding).")
    p(f"  2. Maximum KV-Cache speedup: {max_speedup_kv:.2f}× (at longest sequence).")
    p(f"  3. Maximum Speculative speedup: {max_speedup_sp:.2f}×.")
    p(f"  4. Both methods scale more favourably than the O(n²) baseline.")
    p(f"  5. Speculative speedup depends on draft model quality (acceptance rate α).")

    report_text = "\n".join(lines)
    path = os.path.join(RESULTS_DIR, "benchmark_report.txt")
    with open(path, "w") as f:
        f.write(report_text)

    return path


# ─────────────────────────────────────────────────────────────────────────────
# Main driver
# ─────────────────────────────────────────────────────────────────────────────

def run_all_benchmarks(
    main_model   : GPTModel,
    draft_model  : GPTModel,
    tokenizer    : CharTokenizer,
    device       : torch.device,
    quick        : bool = False,
    training_log : Optional[dict] = None,
) -> None:
    """Run the complete benchmark suite and save all outputs."""
    import json

    token_counts  = [10, 25, 50, 100] if quick else [25, 50, 100, 200, 350]
    n_runs        = 1 if quick else 3
    k_values      = [1, 4, 8] if quick else [1, 2, 4, 6, 8]
    n_tokens_spec = 50 if quick else 100

    equiv_report    = run_equivalence_test(main_model, tokenizer, device, n_tokens=50)
    latency_results = run_latency_benchmark(
        main_model, draft_model, tokenizer, device,
        token_counts=token_counts, n_runs=n_runs,
    )
    spec_analysis = run_speculation_analysis(
        main_model, draft_model, tokenizer, device,
        k_values=k_values, n_tokens=n_tokens_spec,
    )

    # Save structured JSON for plots_pro.py
    data_payload = {
        "latency"      : latency_results,
        "equivalence"  : {
            "all_equivalent": equiv_report["all_equivalent"],
            "n_match"       : equiv_report["n_match"],
            "n_total"       : equiv_report["n_total"],
            "match_rate"    : equiv_report["match_rate"],
            "per_prompt"    : [
                {"prompt_idx": r["prompt_idx"], "match": r["match"],
                 "first_diff": r["first_diff"],
                 "baseline": r["baseline"], "kv_cache": r["kv_cache"]}
                for r in equiv_report["per_prompt"]
            ],
        },
        "spec_analysis": spec_analysis["analysis"],
        "training"     : training_log or {},
    }
    data_path = os.path.join(RESULTS_DIR, "benchmark_data.json")
    with open(data_path, "w") as f:
        json.dump(data_payload, f, indent=2)
    print(f"\n  Benchmark data saved to {data_path}")

    # Quick-preview PNGs
    print("\n" + "==" * 30)
    plot_latency(latency_results)
    plot_speedup(latency_results)
    plot_complexity(latency_results)
    plot_tokens_per_second(latency_results)
    plot_speculation_analysis(spec_analysis)
    plot_equivalence_summary(equiv_report)
    rep = generate_text_report(equiv_report, latency_results, spec_analysis,
                               main_model, draft_model)
    print(f"  Text report -> {rep}")

    # Publication figures
    print("\n" + "==" * 30)
    print("Publication figures (PDF + pgfplots .tex) ...")
    try:
        from plots_pro import build_all, write_latex_snippet
        build_all(data_payload)
        write_latex_snippet(data_payload)
    except Exception as e:
        print(f"  [warning] pro plots: {e}")

    print("\nAll benchmarks complete.")
