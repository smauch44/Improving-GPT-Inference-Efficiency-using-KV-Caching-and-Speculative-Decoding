# GPT Inference Efficiency: KV-Caching & Speculative Decoding
**EN.705.743.8VL.SP26 — ChatGPT from Scratch | Stefan Mauch**

---

## Overview

This project implements and benchmarks two modern inference optimisations for
GPT-style language models on Tiny Shakespeare:

| Method | Description | Complexity |
|---|---|---|
| **Baseline** | Standard autoregressive generation (course GPT) | O(n²) |
| **KV-Cache** | Stores past Key/Value tensors; only processes new token | O(n) |
| **KV-Cache + Speculative** | Draft model proposes K tokens; main model verifies in one pass | O(n/K·α) |

---

## Quick Start

```bash
# Install dependencies
pip install torch numpy matplotlib

# Full run: train both models, benchmark, demo
python main.py

# Quick smoke-test (a few minutes, all features exercised)
python main.py --quick

# Skip training (requires existing checkpoints)
python main.py --no-train

# Generation  
python main.py 
```

---

## File Structure

```
├── main.py             ← Entry point (run this)
├── gpt_model.py        ← GPT architecture with KV-Cache support
├── inference.py        ← Baseline / KV-Cache / Speculative generation
├── benchmark.py        ← Latency, speedup, equivalence experiments + plots
├── train.py            ← Training loop for main and draft models
├── data_utils.py       ← Tiny Shakespeare download & character tokeniser
│
├── checkpoints/        ← Saved model weights (created after training)
│   ├── main_model.pt
│   └── draft_model.pt
│
└── results/            ← All outputs (created after benchmarking)
    ├── latency_vs_tokens.png
    ├── speedup_vs_tokens.png
    ├── complexity_loglog.png
    ├── throughput_vs_tokens.png
    ├── speculation_analysis.png
    ├── equivalence_summary.png
    └── benchmark_report.txt
```

---

## Command-Line Flags

| Flag | Default | Description |
|---|---|---|
| `--quick` | off | Very fast run (50 training iterations, fewer bench points) |
| `--no-train` | off | Load existing checkpoints instead of training |
| `--demo-only` | off | Skip benchmarks; only show generation demo |
| `--n-tokens N` | 200 | Tokens to generate in the demo |
| `--spec-k K` | 4 | Speculation window size (draft tokens per round) |
| `--temperature T` | 0.8 | Sampling temperature for demo (0 = greedy) |

---

## Model Architecture

Both models use the same GPT decoder architecture (matching the course `gpt.py`):

```
Prompt tokens
     │
     ▼
 Token Embedding + Position Embedding
     │
     ▼
 ┌─────────────────────────────────────┐
 │  TransformerDecoderBlock × L        │
 │  ┌─────────────────────────────┐    │
 │  │  LayerNorm                  │    │
 │  │  MultiHeadAttention         │    │  ← KV-Cache lives here
 │  │  + residual                 │    │
 │  ├─────────────────────────────┤    │
 │  │  LayerNorm                  │    │
 │  │  MLP (d → 4d → d)           │    │
 │  │  + Dropout + residual       │    │
 │  └─────────────────────────────┘    │
 └─────────────────────────────────────┘
     │
     ▼
 LayerNorm → Linear → Logits (vocab_size)
```

| | Main Model | Draft Model |
|---|---|---|
| `d_model` | 128 | 64 |
| `n_heads` | 8 | 4 |
| `layers` | 4 | 2 |
| Parameters | ~810K | ~130K |

---

## How KV-Caching Works

During standard autoregressive generation, each forward pass recomputes
Query, Key, and Value tensors for **every** token in the context:

```
Step 1:  forward([t₁])          → logit → t₂
Step 2:  forward([t₁, t₂])     → logit → t₃   ← recomputes K,V for t₁
Step 3:  forward([t₁,t₂,t₃])   → logit → t₄   ← recomputes K,V for t₁,t₂
```

With KV-Caching:

```
Prefill:  forward([t₁])       → logit + cache {K₁,V₁} → t₂
Step 2:   forward([t₂], cache) → logit + cache {K₁,K₂,V₁,V₂} → t₃
Step 3:   forward([t₃], cache) → logit + cache {K₁,K₂,K₃,...} → t₄
```

Each decode step only processes **one token**, reading cached K,V from memory.
Attention cost per step drops from O(T²) to O(T).

**Output equivalence**: Under greedy decoding, both methods produce identical
token sequences because the attention computation yields the same values
(the cache contains the exact K,V matrices that would have been computed).

---

## How Speculative Decoding Works

Based on: *Leviathan et al. "Fast Inference from Transformers via Speculative
Decoding." ICML 2023.*

```
For each round:
  1. Draft model proposes K tokens [x̃₁, …, x̃ₖ] with probabilities q(·)
  2. Main model evaluates all K positions in ONE forward pass → p(·)
  3. Accept x̃ᵢ with probability min(1, p(x̃ᵢ)/q(x̃ᵢ))
  4. On first rejection: resample from corrected distribution max(0, p-q)/Z
  5. If all K accepted: draw one bonus token from p(·|full context)

Speedup ≈ K × α  (α = acceptance rate)
```

The draft model is ~6× smaller than the main model, so K draft steps cost
roughly the same as one main-model forward pass.

---

## Outputs

After running `python main.py`, the following are saved to `./results/`:

| File | Description |
|---|---|
| `latency_vs_tokens.png` | Wall-clock time vs token count (3 methods) |
| `speedup_vs_tokens.png` | Speedup factor over baseline vs token count |
| `complexity_loglog.png` | Log-log plot confirming O(n²) vs O(n) scaling |
| `throughput_vs_tokens.png` | Tokens-per-second vs token count |
| `speculation_analysis.png` | Speedup and acceptance rate vs K |
| `equivalence_summary.png` | Bar chart: exact match count across prompts |
| `benchmark_report.txt` | Full numerical results and interpretation |

---

## References

[1] Vaswani, A., et al. "Attention Is All You Need." *NeurIPS* 2017.

[2] Leviathan, Y., Kalman, M., & Matias, Y. "Fast Inference from Transformers
via Speculative Decoding." *ICML* 2023.

[3] Chen, C., et al. "Accelerating Large Language Model Decoding with
Speculative Sampling." *arXiv:2302.01318*, 2023.

[4] Raschka, S. *Build a Large Language Model (From Scratch).* Simon and
Schuster, 2024.
