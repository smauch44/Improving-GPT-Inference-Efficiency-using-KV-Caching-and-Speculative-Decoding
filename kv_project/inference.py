"""
inference.py
============
Three autoregressive text-generation modes, each returning both the
generated token sequence and timing information.

  1. baseline_generate    – Standard GPT inference (recomputes all keys/values
                            at every step). O(T²) attention cost per sequence.

  2. kvcache_generate     – Prefill the prompt once, then step one token at a
                            time using cached K/V tensors. O(T) attention per
                            step, identical outputs to baseline under greedy
                            decoding.

  3. speculative_generate – Draft model proposes K tokens; main model verifies
                            all K in a single forward pass. Reduces number of
                            expensive main-model calls by factor ≈ K × α (α =
                            acceptance rate). Based on:
                            Leviathan et al. "Fast Inference from Transformers
                            via Speculative Decoding." ICML 2023.

All three functions share the same signature for easy comparison.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from gpt_model import GPTModel


def _sample_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: Optional[int],
) -> torch.Tensor:
    """
    Sample the next token from logits of shape (B, V).

    Args:
        logits      : Raw logits.
        temperature : 0.0 means greedy argmax.
        top_k       : Optional top-k restriction.

    Returns:
        token : (B, 1) long tensor.
    """
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k is not None:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        threshold = values[:, -1].unsqueeze(-1)
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _distribution_from_logits(
    logits: torch.Tensor,
    temperature: float,
    top_k: Optional[int],
) -> torch.Tensor:
    """
    Convert logits of shape (1, V) into a sampling distribution consistent
    with _sample_token().

    Returns:
        probs : (1, V)
    """
    if temperature == 0.0:
        probs = torch.zeros_like(logits)
        argmax_idx = logits.argmax(dim=-1, keepdim=True)
        probs.scatter_(1, argmax_idx, 1.0)
        return probs

    work = logits / temperature

    if top_k is not None:
        values, _ = torch.topk(work, min(top_k, work.size(-1)))
        threshold = values[:, -1].unsqueeze(-1)
        work = work.masked_fill(work < threshold, float("-inf"))

    return F.softmax(work, dim=-1)


def _sample_from_corrected_distribution(
    p_probs: torch.Tensor,
    q_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Sample from the corrected rejection distribution:
        r(x) ∝ max(0, p(x) - q(x))

    Args:
        p_probs : (1, V)
        q_probs : (1, V)

    Returns:
        token : (1, 1) long tensor
    """
    corrected = torch.clamp(p_probs - q_probs, min=0.0)
    z = corrected.sum(dim=-1, keepdim=True)

    # Numerical fallback: if z == 0 due to precision, sample from p directly.
    # In exact arithmetic this branch should be rare.
    if torch.all(z <= 0):
        return torch.multinomial(p_probs, num_samples=1)

    corrected = corrected / z
    return torch.multinomial(corrected, num_samples=1)


@torch.no_grad()
def baseline_generate(
    model: GPTModel,
    prompt: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[List[int], dict]:
    """
    Standard autoregressive generation WITHOUT caching.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    ctx = prompt.to(device)
    new_tokens: List[int] = []

    for _ in range(max_new_tokens):
        ctx_in = ctx[:, -model.max_seq_len:]
        logits, _ = model(ctx_in)
        next_logits = logits[:, -1, :]
        next_tok = _sample_token(next_logits, temperature, top_k)
        new_tokens.append(next_tok.item())
        ctx = torch.cat([ctx, next_tok], dim=1)

    return new_tokens, {"method": "baseline", "steps": max_new_tokens}


@torch.no_grad()
def kvcache_generate(
    model: GPTModel,
    prompt: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[List[int], dict]:
    """
    KV-cache inference: prefill + single-token decode.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    prompt = prompt.to(device)

    logits, past_kvs = model(prompt, use_cache=True)
    next_logits = logits[:, -1, :]
    next_tok = _sample_token(next_logits, temperature, top_k)
    new_tokens: List[int] = [next_tok.item()]

    for _ in range(max_new_tokens - 1):
        cache_len = past_kvs[0][0].shape[2] if past_kvs else 0
        if cache_len >= model.max_seq_len - 1:
            break

        logits, past_kvs = model(next_tok, past_kvs=past_kvs, use_cache=True)
        next_logits = logits[:, -1, :]
        next_tok = _sample_token(next_logits, temperature, top_k)
        new_tokens.append(next_tok.item())

    cache_depth = past_kvs[0][0].shape[2] if past_kvs else 0

    return new_tokens, {
        "method": "kv_cache",
        "steps": len(new_tokens),
        "cache_depth": cache_depth,
    }


@torch.no_grad()
def speculative_generate(
    main_model: GPTModel,
    draft_model: GPTModel,
    prompt: torch.Tensor,
    max_new_tokens: int,
    speculation_k: int = 4,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[List[int], dict]:
    """
    Correct speculative decoding with rejection sampling.

    Greedy mode:
      Accept draft token i iff argmax(p_i) == draft_token_i.

    Stochastic mode:
      Accept draft token i with probability min(1, p_i(x_i) / q_i(x_i)).
      On rejection, sample from max(0, p_i - q_i) / Z.
      If all K are accepted, sample one bonus token from p_{K+1}.
    """
    if device is None:
        device = next(main_model.parameters()).device

    main_model.eval()
    draft_model.eval()
    prompt = prompt.to(device)

    generated: List[int] = []
    ctx = prompt.clone()

    total_accepted = 0
    total_proposed = 0
    total_rounds = 0

    while len(generated) < max_new_tokens:
        if ctx.shape[1] >= main_model.max_seq_len - speculation_k - 2:
            break

        remaining = max_new_tokens - len(generated)
        K = min(speculation_k, remaining)
        total_rounds += 1

        # Step 1: draft model proposes K tokens and stores full q_i(.)
        draft_tokens: List[int] = []
        q_dists: List[torch.Tensor] = []

        draft_ctx = ctx.clone()
        for _ in range(K):
            draft_input = draft_ctx[:, -draft_model.max_seq_len:]
            draft_logits, _ = draft_model(draft_input)
            last_logits = draft_logits[:, -1, :]
            q_probs = _distribution_from_logits(last_logits, temperature, top_k)

            if temperature == 0.0:
                tok = last_logits.argmax(dim=-1, keepdim=True)
            else:
                tok = torch.multinomial(q_probs, num_samples=1)

            draft_tokens.append(tok.item())
            q_dists.append(q_probs)
            draft_ctx = torch.cat([draft_ctx, tok], dim=1)

        # Step 2: main model verifies the K proposed positions plus one bonus position
        proposal_tensor = torch.tensor(draft_tokens, device=device, dtype=torch.long).unsqueeze(0)
        verify_ctx = torch.cat([ctx, proposal_tensor], dim=1)

        if verify_ctx.shape[1] > main_model.max_seq_len:
            verify_ctx = verify_ctx[:, -main_model.max_seq_len:]

        main_logits, _ = main_model(verify_ctx)

        prompt_len = ctx.shape[1]
        start_idx = main_logits.shape[1] - (K + 1)
        p_logits_seq = [main_logits[:, start_idx + i, :] for i in range(K + 1)]

        accepted_this_round = 0
        rejected = False

        for i in range(K):
            p_logits = p_logits_seq[i]
            q_probs = q_dists[i]
            proposed_token = draft_tokens[i]

            if temperature == 0.0:
                p_argmax = p_logits.argmax(dim=-1).item()
                accept = (p_argmax == proposed_token)
            else:
                p_probs = _distribution_from_logits(p_logits, temperature, top_k)
                q_prob_token = q_probs[0, proposed_token].item()
                p_prob_token = p_probs[0, proposed_token].item()

                # If q assigns zero probability to a proposed token, that should not happen
                # because the token was sampled from q. Guard anyway for numerical safety.
                if q_prob_token <= 0.0:
                    accept_prob = 1.0 if p_prob_token > 0.0 else 0.0
                else:
                    accept_prob = min(1.0, p_prob_token / q_prob_token)

                accept = (torch.rand(1, device=device).item() < accept_prob)

            if accept:
                tok_tensor = torch.tensor([[proposed_token]], device=device, dtype=torch.long)
                generated.append(proposed_token)
                ctx = torch.cat([ctx, tok_tensor], dim=1)
                accepted_this_round += 1
                total_accepted += 1

                if len(generated) >= max_new_tokens:
                    break
            else:
                rejected = True

                if temperature == 0.0:
                    corrected_tok = p_logits.argmax(dim=-1, keepdim=True)
                else:
                    p_probs = _distribution_from_logits(p_logits, temperature, top_k)
                    corrected_tok = _sample_from_corrected_distribution(p_probs, q_probs)

                generated.append(corrected_tok.item())
                ctx = torch.cat([ctx, corrected_tok.to(device=device, dtype=torch.long)], dim=1)
                break

        total_proposed += K

        if len(generated) >= max_new_tokens:
            break

        if not rejected and accepted_this_round == K:
            bonus_logits = p_logits_seq[K]
            bonus_tok = _sample_token(bonus_logits, temperature, top_k)
            generated.append(bonus_tok.item())
            ctx = torch.cat([ctx, bonus_tok], dim=1)

    generated = generated[:max_new_tokens]

    acceptance_rate = total_accepted / max(total_proposed, 1)
    avg_tokens_per_round = len(generated) / max(total_rounds, 1)

    return generated, {
        "method": "speculative",
        "steps": len(generated),
        "total_proposed": total_proposed,
        "total_accepted": total_accepted,
        "acceptance_rate": acceptance_rate,
        "total_rounds": total_rounds,
        "avg_tokens_per_round": avg_tokens_per_round,
        "speculation_k": speculation_k,
    }


def verify_equivalence(
    model: GPTModel,
    prompts: List[torch.Tensor],
    n_tokens: int = 50,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Verify that baseline and KV-cache generation produce identical outputs
    under greedy decoding.
    """
    if device is None:
        device = next(model.parameters()).device

    results = []

    for i, prompt in enumerate(prompts):
        base_toks, _ = baseline_generate(
            model, prompt, n_tokens, temperature=0.0, device=device
        )
        cache_toks, _ = kvcache_generate(
            model, prompt, n_tokens, temperature=0.0, device=device
        )

        match = (base_toks == cache_toks)
        if not match:
            first_diff = next(
                (j for j, (a, b) in enumerate(zip(base_toks, cache_toks)) if a != b),
                n_tokens,
            )
        else:
            first_diff = n_tokens

        results.append({
            "prompt_idx": i,
            "match": match,
            "first_diff": first_diff,
            "baseline": base_toks,
            "kv_cache": cache_toks,
        })

    n_match = sum(r["match"] for r in results)
    all_match = (n_match == len(prompts))

    return {
        "all_equivalent": all_match,
        "n_match": n_match,
        "n_total": len(prompts),
        "match_rate": n_match / len(prompts),
        "per_prompt": results,
    }


if __name__ == "__main__":
    device = torch.device("cpu")
    model = GPTModel(
        d_model=64,
        n_heads=4,
        layers=2,
        vocab_size=65,
        max_seq_len=256,
    ).to(device)

    prompt = torch.randint(65, (1, 10))

    t1, _ = baseline_generate(model, prompt, 20, temperature=0.0)
    t2, _ = kvcache_generate(model, prompt, 20, temperature=0.0)

    print("Baseline tokens:", t1[:10])
    print("KV-Cache tokens:", t2[:10])
    print("Identical:", t1 == t2)