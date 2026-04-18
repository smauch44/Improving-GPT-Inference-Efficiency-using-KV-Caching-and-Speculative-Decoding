"""
main.py
=======
Primary entry point for the KV-Cache / Speculative Decoding project.

Running `python main.py` will:

  1. Download Tiny Shakespeare (if not already cached).
  2. Train a main GPT model and a smaller draft model (or load existing checkpoints).
  3. Run the full benchmark suite:
       • Output equivalence verification (baseline == KV-cache)
       • Wall-clock latency at multiple sequence lengths
       • Speedup factors
       • Speculative decoding acceptance rate analysis
  4. Save plots and a text report to ./results/.
  5. Print a qualitative generation demo for all three methods.

Flags:
  --quick         Fast smoke test (few training iters, fewer benchmark points).
  --no-train      Skip training; requires existing checkpoints in ./checkpoints/.
  --demo-only     Skip benchmarks; only run the generation demo.
  --n-tokens N    Number of tokens to generate in the demo (default 150).
  --spec-k K      Speculation window size (default 4).
  --temperature T Temperature for demo generation (default 0.8).

References:
  [1] Vaswani et al. "Attention Is All You Need." NeurIPS 2017.
  [2] Leviathan et al. "Fast Inference from Transformers via Speculative
      Decoding." ICML 2023.
  [3] Chen et al. "Accelerating Large Language Model Decoding with
      Speculative Sampling." arXiv 2023.
"""

import argparse
import os
import time

import torch

from benchmark import run_all_benchmarks
from data_utils import CharTokenizer, load_shakespeare
from gpt_model import GPTModel
from inference import baseline_generate, kvcache_generate, speculative_generate
from train import (
    CHECKPOINT_DIR,
    DRAFT_CFG,
    MAIN_CFG,
    TRAIN_CFG,
    get_device,
    train_model,
    make_loaders,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sync_device(device: torch.device) -> None:
    """Synchronize async accelerator work so timings are accurate."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(name: str, device: torch.device):
    """Load model + tokenizer state from a checkpoint file."""
    path = os.path.join(CHECKPOINT_DIR, f"{name}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Run `python main.py` (without --no-train) to train first."
        )

    ckpt = torch.load(path, map_location=device)

    cfg = ckpt["model_cfg"]
    model = GPTModel(
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        layers=cfg["layers"],
        vocab_size=ckpt["vocab_size"],
        max_seq_len=cfg["max_seq_len"],
        dropout=0.0,  # no dropout at inference time
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Rebuild tokenizer from saved vocab
    tokenizer = CharTokenizer.__new__(CharTokenizer)
    tokenizer._c2i = ckpt["char_to_idx"]
    tokenizer._i2c = ckpt["idx_to_char"]
    tokenizer.vocab_size = ckpt["vocab_size"]

    print(f"  Loaded: {path}  (val_loss={ckpt['val_loss']:.4f})")
    return model, tokenizer


def checkpoint_exists(name: str) -> bool:
    """Return True if checkpoint file exists."""
    return os.path.exists(os.path.join(CHECKPOINT_DIR, f"{name}.pt"))


# ─────────────────────────────────────────────────────────────────────────────
# Generation demo
# ─────────────────────────────────────────────────────────────────────────────

def run_demo(
    main_model: GPTModel,
    draft_model: GPTModel,
    tokenizer: CharTokenizer,
    device: torch.device,
    n_tokens: int,
    spec_k: int,
    temperature: float,
) -> None:
    """Print side-by-side generation samples and timing for all three methods."""
    demo_prompt = (
        "ROMEO:\n"
        "What light through yonder window breaks?\n"
        "It is the east, and Juliet is the sun.\n"
    )

    prompt_ids = torch.tensor(
        [tokenizer.encode(demo_prompt)],
        dtype=torch.long,
    )

    print("\n" + "═" * 70)
    print("GENERATION DEMO")
    print("═" * 70)
    print(f"Prompt    : {demo_prompt!r}")
    print(f"New tokens: {n_tokens}    Temperature: {temperature}    Spec-K: {spec_k}")

    methods = [
        ("Baseline (no cache)", "baseline"),
        ("KV-Cache", "kv_cache"),
        (f"Speculative (K={spec_k})", "speculative"),
    ]

    for label, method in methods:
        _sync_device(device)
        t0 = time.perf_counter()

        if method == "baseline":
            toks, stats = baseline_generate(
                main_model,
                prompt_ids,
                n_tokens,
                temperature=temperature,
                device=device,
            )
        elif method == "kv_cache":
            toks, stats = kvcache_generate(
                main_model,
                prompt_ids,
                n_tokens,
                temperature=temperature,
                device=device,
            )
        else:
            toks, stats = speculative_generate(
                main_model,
                draft_model,
                prompt_ids,
                n_tokens,
                speculation_k=spec_k,
                temperature=temperature,
                device=device,
            )

        _sync_device(device)
        elapsed = time.perf_counter() - t0
        text = tokenizer.decode(toks)

        print(f"\n{'─' * 70}")
        print(f"Method: {label}")
        print(f"  Time      : {elapsed:.3f}s  ({n_tokens / elapsed:.1f} tok/s)")
        if method == "speculative":
            print(
                f"  Accept rate: {stats['acceptance_rate']:.1%}  "
                f"Tokens/round: {stats['avg_tokens_per_round']:.2f}"
            )
        print(f"\n  {demo_prompt.rstrip()}{text}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="KV-Cache & Speculative Decoding — GPT Inference Efficiency"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: very few training iterations and benchmark points",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Skip training; load existing checkpoints",
    )
    parser.add_argument(
        "--demo-only",
        action="store_true",
        help="Skip benchmarks; run generation demo only",
    )
    parser.add_argument(
        "--n-tokens",
        type=int,
        default=150,
        help="Tokens to generate in demo (default: 150)",
    )
    parser.add_argument(
        "--spec-k",
        type=int,
        default=4,
        help="Speculation window size K (default: 4)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Generation temperature for demo (default: 0.8)",
    )
    return parser.parse_args()


def main():
    """Run training/loading, benchmarking, and demo."""
    args = parse_args()
    device = get_device()

    print("=" * 70)
    print("  GPT INFERENCE EFFICIENCY: KV-Cache & Speculative Decoding")
    print("  EN.705.743.8VL.SP26 — Stefan Mauch")
    print("=" * 70)
    print(f"\nDevice: {device}")

    # ── Load data ────────────────────────────────────────────────────────────
    print("\nLoading Tiny Shakespeare …")
    tokenizer, train_data, val_data = load_shakespeare()
    print(
        f"  Vocab size: {tokenizer.vocab_size}  |  "
        f"Train: {len(train_data):,} tokens  |  Val: {len(val_data):,} tokens"
    )

    # ── Train or load ────────────────────────────────────────────────────────
    training_logs: dict = {}

    if args.no_train:
        print("\nLoading checkpoints (--no-train) …")
        main_model, tokenizer = load_checkpoint("main_model", device)
        draft_model, _ = load_checkpoint("draft_model", device)
    else:
        max_iters = TRAIN_CFG["quick_iters"] if args.quick else TRAIN_CFG["max_iters"]

        train_loader, val_loader = make_loaders(
            train_data,
            val_data,
            block_size=TRAIN_CFG["block_size"],
            batch_size=TRAIN_CFG["batch_size"],
        )

        if checkpoint_exists("main_model") and not args.quick:
            print("\nFound existing main_model checkpoint — loading …")
            main_model, tokenizer = load_checkpoint("main_model", device)
        else:
            main_model, main_log = train_model(
                MAIN_CFG,
                "main_model",
                tokenizer,
                train_loader,
                val_loader,
                device,
                max_iters,
            )
            training_logs["main_model"] = main_log
            main_model.eval()

        if checkpoint_exists("draft_model") and not args.quick:
            print("Found existing draft_model checkpoint — loading …")
            draft_model, _ = load_checkpoint("draft_model", device)
        else:
            draft_model, draft_log = train_model(
                DRAFT_CFG,
                "draft_model",
                tokenizer,
                train_loader,
                val_loader,
                device,
                max_iters,
            )
            training_logs["draft_model"] = draft_log
            draft_model.eval()

    # ── Print model sizes ────────────────────────────────────────────────────
    print(f"\nMain model  : {main_model.num_parameters():,} parameters")
    print(
        f"Draft model : {draft_model.num_parameters():,} parameters "
        f"({draft_model.num_parameters() / main_model.num_parameters():.0%} of main)"
    )

    # ── Benchmarks ───────────────────────────────────────────────────────────
    if not args.demo_only:
        print("\nRunning benchmarks …")
        run_all_benchmarks(
            main_model,
            draft_model,
            tokenizer,
            device,
            quick=args.quick,
            training_log=training_logs,
        )

    # ── Generation demo ──────────────────────────────────────────────────────
    run_demo(
        main_model,
        draft_model,
        tokenizer,
        device,
        n_tokens=args.n_tokens,
        spec_k=args.spec_k,
        temperature=args.temperature,
    )

    print("\n" + "=" * 70)
    print("Done!  Results saved to ./results/")
    print("=" * 70)


if __name__ == "__main__":
    main()