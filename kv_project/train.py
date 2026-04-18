"""
train.py
========
Training loop for the main (large) and draft (small) GPT models on
Tiny Shakespeare.

Models are saved to disk so they can be loaded for inference benchmarks
without re-training.

Usage:
    python train.py                 # trains both models
    python train.py --model main    # train main model only
    python train.py --model draft   # train draft model only
    python train.py --quick         # very fast run for testing (few iters)
"""

import argparse
import os
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from data_utils import load_shakespeare, make_loaders
from gpt_model import GPTModel


# ─────────────────────────────────────────────────────────────────────────────
# Model configurations
# ─────────────────────────────────────────────────────────────────────────────

MAIN_CFG: Dict = {
    "d_model"    : 128,
    "n_heads"    : 8,
    "layers"     : 4,
    "max_seq_len": 256,
    "dropout"    : 0.1,
}

DRAFT_CFG: Dict = {
    "d_model"    : 64,
    "n_heads"    : 4,
    "layers"     : 2,
    "max_seq_len": 256,
    "dropout"    : 0.1,
}

TRAIN_CFG: Dict = {
    "block_size" : 64,        # shorter context, memory-friendly
    "batch_size" : 16,        # small enough for CPU/sandbox environments
    "lr"         : 3e-4,
    "max_iters"  : 3000,      # full training
    "quick_iters": 80,        # --quick flag
    "eval_every" : 200,
    "eval_iters" : 20,
    "grad_clip"  : 1.0,
    "weight_decay": 0.01,
}

CHECKPOINT_DIR = "checkpoints"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    val_loader,
    device: torch.device,
    n_batches: int = 50,
) -> float:
    """Estimate validation loss over n_batches mini-batches."""
    model.eval()
    total, count = 0.0, 0
    for i, (x, y) in enumerate(val_loader):
        if i >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        B, T, V = logits.shape
        loss = nn.functional.cross_entropy(logits.view(B * T, V), y.view(B * T))
        total += loss.item()
        count += 1
    model.train()
    return total / max(count, 1)


def save_checkpoint(model: nn.Module, tokenizer, name: str, val_loss: float) -> str:
    """Save model weights + metadata to checkpoints/<name>.pt"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"{name}.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab_size"  : tokenizer.vocab_size,
            "char_to_idx" : tokenizer._c2i,
            "idx_to_char" : tokenizer._i2c,
            "val_loss"    : val_loss,
            "model_cfg"   : MAIN_CFG if name == "main_model" else DRAFT_CFG,
        },
        path,
    )
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    cfg: Dict,
    name: str,
    tokenizer,
    train_loader,
    val_loader,
    device: torch.device,
    max_iters: int,
) -> GPTModel:
    """
    Train a GPTModel defined by *cfg* for *max_iters* gradient steps.

    Returns the trained model (still on *device*).
    """
    model = GPTModel(
        d_model    = cfg["d_model"],
        n_heads    = cfg["n_heads"],
        layers     = cfg["layers"],
        vocab_size = tokenizer.vocab_size,
        max_seq_len= cfg["max_seq_len"],
        dropout    = cfg["dropout"],
    ).to(device)

    print(f"\n{'─'*60}")
    print(f"Training: {name}")
    print(f"  Parameters : {model.num_parameters():,}")
    print(f"  Device     : {device}")
    print(f"  Iterations : {max_iters:,}")
    print(f"{'─'*60}")

    optimizer = AdamW(
        model.parameters(),
        lr=TRAIN_CFG["lr"],
        weight_decay=TRAIN_CFG["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=1e-5)

    train_iter   = iter(train_loader)
    criterion    = nn.CrossEntropyLoss()
    best_val     = float("inf")
    t0           = time.time()
    training_log: list = []

    for step in range(1, max_iters + 1):
        # ── Fetch next batch (cycle over dataset) ──────────────────────────
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        # ── Forward + backward ─────────────────────────────────────────────
        model.train()
        logits, _ = model(x)
        B, T, V = logits.shape
        loss = criterion(logits.view(B * T, V), y.view(B * T))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CFG["grad_clip"])
        optimizer.step()
        scheduler.step()

        # ── Periodic evaluation ────────────────────────────────────────────
        if step % TRAIN_CFG["eval_every"] == 0 or step == max_iters:
            val_loss = estimate_loss(model, val_loader, device, TRAIN_CFG["eval_iters"])
            elapsed  = time.time() - t0
            print(
                f"  step {step:>5}/{max_iters} | "
                f"train_loss={loss.item():.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e} | "
                f"elapsed={elapsed:.1f}s"
            )
            training_log.append({"step": step, "val_loss": val_loss,
                                  "train_loss": loss.item()})
            if val_loss < best_val:
                best_val = val_loss
                path = save_checkpoint(model, tokenizer, name, val_loss)
                print(f"    ✓ Checkpoint saved → {path}")

    print(f"\n{name} training complete. Best val loss: {best_val:.4f}\n")
    return model, training_log


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train GPT models for KV-Cache project")
    parser.add_argument("--model", choices=["main", "draft", "both"], default="both")
    parser.add_argument("--quick", action="store_true",
                        help="Run a quick smoke-test with very few iterations")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Load data
    print("\nLoading Tiny Shakespeare …")
    tokenizer, train_data, val_data = load_shakespeare()
    print(f"  Vocab size : {tokenizer.vocab_size}")
    print(f"  Train size : {len(train_data):,} tokens")
    print(f"  Val size   : {len(val_data):,} tokens")

    max_iters = TRAIN_CFG["quick_iters"] if args.quick else TRAIN_CFG["max_iters"]

    train_loader, val_loader = make_loaders(
        train_data, val_data,
        block_size=TRAIN_CFG["block_size"],
        batch_size=TRAIN_CFG["batch_size"],
    )

    if args.model in ("main", "both"):
        train_model(MAIN_CFG, "main_model", tokenizer, train_loader, val_loader, device, max_iters)

    if args.model in ("draft", "both"):
        train_model(DRAFT_CFG, "draft_model", tokenizer, train_loader, val_loader, device, max_iters)

    print("All training complete. Checkpoints saved in ./checkpoints/")


if __name__ == "__main__":
    main()
