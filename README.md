# GPT Inference Efficiency: KV-Caching & Speculative Decoding
**EN.705.743.8VL.SP26 — ChatGPT from Scratch | Stefan Mauch**

---

## Overview

This project implements and evaluates two modern inference optimisations for GPT-style language models using the Tiny Shakespeare dataset.

The goal is to analyze correctness, latency, and real-world speedup.

### Methods

| Method | Description |
|---|---|
| Baseline | Standard autoregressive decoding (recomputes full attention each step) |
| KV-Cache | Reuses past Key/Value tensors to avoid redundant computation |
| Speculative Decoding | Uses a smaller draft model to propose tokens and a main model to verify them |

---

## Key Findings

- KV-cache preserves exact output equivalence under greedy decoding (10/10 prompts matched)
- KV-cache improves latency by ~1.04×–1.41×, with stronger gains at longer sequences
- Speculative decoding underperformed in this setup due to:
  - low acceptance rates
  - draft-model mismatch
  - verification overhead (MPS backend)
- Conclusion: theoretical speedups do not always translate into real-world gains

---

## Quick Start

```bash
pip install torch numpy matplotlib

python main.py
python main.py --quick
python main.py --no-train
```

---

## Project Structure

```
── benchmark.py
├── checkpoints
│   ├── draft_model.pt
│   └── main_model.pt
├── data
│   └── shakespeare.txt
├── data_utils.py
├── gpt_model.py
├── inference.py
├── installation.txt
├── main.py
├── plots.py
├── README.md
├── results
│   ├── benchmark_data.json
│   ├── benchmark_report.txt
│   ├── complexity_loglog.png
│   ├── equivalence_summary.png
│   ├── fig1_latency.pdf
│   ├── fig1_latency.png
│   ├── fig1_latency.tex
│   ├── fig2_speedup.pdf
│   ├── fig2_speedup.png
│   ├── fig2_speedup.tex
│   ├── fig3_complexity.pdf
│   ├── fig3_complexity.png
│   ├── fig3_complexity.tex
│   ├── fig4_throughput.pdf
│   ├── fig4_throughput.png
│   ├── fig4_throughput.tex
│   ├── fig5_speculation.pdf
│   ├── fig5_speculation.png
│   ├── fig5_speculation.tex
│   ├── fig6_equivalence.pdf
│   ├── fig6_equivalence.png
│   ├── fig6_equivalence.tex
│   ├── fig7_kv_memory.pdf
│   ├── fig7_kv_memory.png
│   ├── fig7_kv_memory.tex
│   ├── latency_vs_tokens.png
│   ├── latex_figures.tex
│   ├── speculation_analysis.png
│   ├── speedup_vs_tokens.png
│   └── throughput_vs_tokens.png
├── run.sh
└── train.py
```

---

## Model Architecture

Standard GPT decoder:

- Token + positional embeddings  
- Transformer blocks (attention + MLP)  
- LayerNorm → Linear → logits  

### Model Sizes

| | Main Model | Draft Model |
|---|---|---|
| d_model | 128 | 64 |
| heads | 8 | 4 |
| layers | 4 | 2 |
| parameters | 829,824 | 119,488 |

---

## KV-Caching

### Idea

Avoid recomputing attention for past tokens.

### Benefit

- reduces redundant computation
- improves decoding efficiency
- preserves correctness

### Correctness

Under greedy decoding:
- identical logits
- identical outputs (experimentally verified)

---

## Speculative Decoding

### Idea

- draft model proposes K tokens
- main model verifies in one pass

### Acceptance

min(1, p(x) / q(x))

### Insight

Speedup depends on:
- acceptance rate
- model alignment
- hardware

---

## Experimental Results

### Output Equivalence

- 10 / 10 exact matches
- KV-cache = correct

### Latency

- KV-cache consistently faster
- up to ~1.41× speedup

### Speculative Decoding

- speedup < 1 in all tests
- overhead dominates

---

## Notes on Generation

- Greedy decoding → identical outputs
- Sampling → outputs may differ

---

## Outputs

Saved to ./results/

- latency_vs_tokens.png
- speedup_vs_tokens.png
- complexity_loglog.png
- throughput_vs_tokens.png
- speculation_analysis.png
- equivalence_summary.png
- benchmark_report.txt

---

## References

- Vaswani et al. (2017)
- Leviathan et al. (2023)
- Chen et al. (2023)
- Raschka (2024)

---

## Final Takeaway

KV-caching is a reliable and effective optimisation.

Speculative decoding is powerful in theory, but requires:
- larger models
- better draft alignment
- suitable hardware

In small-scale setups, KV-cache is the most practical improvement.
