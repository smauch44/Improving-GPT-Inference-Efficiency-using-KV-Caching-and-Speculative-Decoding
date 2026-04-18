"""
gpt_model.py
============
Complete GPT model implementation with optional KV-Cache support.

Matches the course architecture from gpt.py:
  - Token embeddings + learned positional embeddings
  - Stack of TransformerDecoderBlock (pre-norm, masked MHA, MLP with dropout)
  - Final linear projection to vocab logits

KV-Cache Extension:
  - MultiHeadAttention.forward() accepts past_kv and use_cache kwargs
  - TransformerDecoderBlock.forward() propagates KV cache through each layer
  - GPTModel.forward() returns present_kv when use_cache=True
  - This enables generate_cached() to perform O(1) attention per step
    instead of O(T^2) recomputation

References:
  - Vaswani et al. "Attention Is All You Need." NeurIPS 2017.
  - Leviathan et al. "Fast Inference from Transformers via Speculative Decoding." ICML 2023.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────
# KV Cache type alias: one (K, V) pair per transformer layer
# Shape of each tensor: (batch, n_heads, seq_len, d_head)
# ─────────────────────────────────────────────────────────────────
KVCache = List[Tuple[torch.Tensor, torch.Tensor]]


class MultiHeadAttention(nn.Module):
    """
    Masked multi-head self-attention with optional KV-Cache.

    During normal (non-cached) forward passes, this behaves exactly like
    the standard causal self-attention used in GPT.

    During cached inference:
      - On the PREFILL step  (T > 1): processes the whole prompt, builds cache.
      - On each DECODE step  (T = 1): processes a single new token, reads and
        updates the cache; no causal mask is needed because the lone query is
        always at the last position.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)

        # Pre-allocate a large causal mask buffer; grows lazily if needed
        self._register_causal_mask(512)

    def _register_causal_mask(self, max_len: int) -> None:
        """Register a causal (upper-triangular) mask as a non-parameter buffer."""
        mask = torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("_causal_mask", mask, persistent=False)

    def _get_causal_mask(self, T_q: int, T_total: int, device: torch.device) -> torch.Tensor:
        """Return causal mask of shape (T_q, T_total), growing buffer if needed."""
        needed = max(T_q, T_total)
        if needed > self._causal_mask.shape[0]:
            self._register_causal_mask(needed * 2)
            self._causal_mask = self._causal_mask.to(device)
        # Query positions start at T_total - T_q (after the cached prefix)
        T_past = T_total - T_q
        return self._causal_mask[T_past : T_past + T_q, :T_total]  # (T_q, T_total)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x        : (B, T, D) – input; T=1 during single-step cached decode.
            past_kv  : Optional cached (K, V), each (B, H, T_past, D_head).
            use_cache: If True, compute and return updated (K, V) tensors.

        Returns:
            output    : (B, T, D)
            present_kv: updated (K, V) cache tuple, or None.
        """
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head

        # ── Project to queries, keys, values ──────────────────────────────
        Q = self.wq(x).view(B, T, H, Dh).transpose(1, 2)  # (B, H, T, Dh)
        K = self.wk(x).view(B, T, H, Dh).transpose(1, 2)  # (B, H, T, Dh)
        V = self.wv(x).view(B, T, H, Dh).transpose(1, 2)  # (B, H, T, Dh)

        # ── Append cached K, V from previous steps ─────────────────────────
        if past_kv is not None:
            K_past, V_past = past_kv
            K = torch.cat([K_past, K], dim=2)  # (B, H, T_past + T, Dh)
            V = torch.cat([V_past, V], dim=2)

        present_kv = (K, V) if use_cache else None
        T_total = K.shape[2]

        # ── Scaled dot-product attention ───────────────────────────────────
        scale = math.sqrt(Dh)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, H, T, T_total)

        # Apply causal mask only when processing multiple query positions.
        # A single query (T=1) is always at the last position and can legally
        # attend to all cached keys → no masking required.
        if T > 1:
            causal = self._get_causal_mask(T, T_total, x.device)
            scores = scores.masked_fill(causal[None, None], float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)

        out = torch.matmul(weights, V)                            # (B, H, T, Dh)
        out = out.transpose(1, 2).contiguous().view(B, T, D)     # (B, T, D)
        out = self.wo(out)

        return out, present_kv


class TransformerDecoderBlock(nn.Module):
    """
    Single GPT decoder block (pre-norm variant, matching gpt.py):

        x = x + MHA(LayerNorm(x))
        x = x + Dropout(FFN(LayerNorm(x)))

    Supports KV-Cache pass-through for efficient inference.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mha   = MultiHeadAttention(d_model, n_heads, dropout)

        self.norm2   = nn.LayerNorm(d_model)
        self.ff1     = nn.Linear(d_model, 4 * d_model, bias=False)
        self.ff2     = nn.Linear(4 * d_model, d_model, bias=False)
        self.act     = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x        : (B, T, D)
            past_kv  : Optional cached (K, V) for this layer.
            use_cache: If True, return updated (K, V).

        Returns:
            x         : (B, T, D)
            present_kv: Updated (K, V) or None.
        """
        # 1) Attention sub-layer with residual
        attn_out, present_kv = self.mha(self.norm1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out

        # 2) Feed-forward sub-layer with residual
        y = self.norm2(x)
        y = self.ff2(self.act(self.ff1(y)))
        y = self.dropout(y)
        x = x + y

        return x, present_kv


class GPTModel(nn.Module):
    """
    Full GPT language model.

    Identical architecture to the course's gpt.py but implemented with
    standard nn.Linear / nn.Embedding so it is fully self-contained.
    Supports KV-Cache for O(1)-per-step autoregressive generation.

    Args:
        d_model    : Embedding / hidden dimension.
        n_heads    : Number of attention heads (must divide d_model).
        layers     : Number of TransformerDecoderBlock layers.
        vocab_size : Output vocabulary size (also input embedding size).
        max_seq_len: Maximum supported sequence length (positional embedding table size).
        dropout    : Dropout probability (used during training).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        layers: int,
        vocab_size: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model     = d_model
        self.vocab_size  = vocab_size
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.drop      = nn.Dropout(dropout)

        self.blocks    = nn.ModuleList(
            [TransformerDecoderBlock(d_model, n_heads, dropout) for _ in range(layers)]
        )
        self.norm_out  = nn.LayerNorm(d_model)
        self.to_logits = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share token embedding and output projection weights
        # (common practice in GPT-style models, reduces parameters)
        self.to_logits.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier / GPT-style weight initialisation."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ─────────────────────────────────────────────────────────────────────
    # Forward pass (used during training and baseline inference)
    # ─────────────────────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,
        past_kvs: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """
        Args:
            x        : (B, T) token-id tensor.
            past_kvs : Optional list of (K, V) per layer from previous steps.
            use_cache: If True, compute and return updated KV caches.

        Returns:
            logits    : (B, T, vocab_size)
            present_kvs: Updated KV cache list (one pair per layer), or None.
        """
        if x.dtype != torch.long:
            x = x.long()

        B, T = x.shape

        # Compute absolute position offsets (accounting for cached prefix)
        past_len = past_kvs[0][0].shape[2] if past_kvs is not None else 0
        if past_len + T > self.max_seq_len:
            raise ValueError(
                f"Sequence length {past_len + T} exceeds max_seq_len={self.max_seq_len}"
            )

        # Build position ids for the current token(s)
        pos_ids = torch.arange(past_len, past_len + T, device=x.device).unsqueeze(0)  # (1, T)

        # Embeddings
        h = self.drop(self.token_emb(x) + self.pos_emb(pos_ids))  # (B, T, D)

        # Pass through decoder blocks, collecting KV caches
        present_kvs: KVCache = []
        for i, block in enumerate(self.blocks):
            layer_past = past_kvs[i] if past_kvs is not None else None
            h, pkv = block(h, past_kv=layer_past, use_cache=use_cache)
            if use_cache:
                present_kvs.append(pkv)

        h = self.norm_out(h)
        logits = self.to_logits(h)  # (B, T, V)

        return logits, (present_kvs if use_cache else None)

    # ─────────────────────────────────────────────────────────────────────
    # Model metadata helpers
    # ─────────────────────────────────────────────────────────────────────
    def num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def config_dict(self) -> dict:
        return {
            "d_model": self.d_model,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    D, H, L, V, S = 128, 8, 4, 65, 256
    model = GPTModel(d_model=D, n_heads=H, layers=L, vocab_size=V, max_seq_len=S)
    print(f"Parameters: {model.num_parameters():,}")

    x = torch.randint(V, (2, 32))

    # Standard forward
    logits, _ = model(x)
    print(f"Standard forward output shape: {logits.shape}")  # (2, 32, 65)

    # Cached forward — prefill then single step
    logits_pre, cache = model(x, use_cache=True)
    x_new = torch.randint(V, (2, 1))
    logits_step, cache2 = model(x_new, past_kvs=cache, use_cache=True)
    print(f"Cached single-step output shape: {logits_step.shape}")  # (2, 1, 65)
    print("Sanity check passed ✓")
