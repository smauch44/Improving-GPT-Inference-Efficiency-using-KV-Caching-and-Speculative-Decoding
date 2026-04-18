"""
data_utils.py
=============
Character-level tokeniser and Tiny Shakespeare data loader.

Course GPT model as baseline.
"""

import os
import urllib.request
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Tokeniser
# ─────────────────────────────────────────────────────────────────────────────

class CharTokenizer:
    """Simple character-level vocabulary.  Maps char → int and back."""

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self._c2i = {c: i for i, c in enumerate(chars)}
        self._i2c = {i: c for i, c in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        return [self._c2i[c] for c in text]

    def decode(self, ids) -> str:
        return "".join(self._i2c[i] for i in ids)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    """
    Sliding-window dataset over a token sequence.

    Each sample is a (context, target) pair where target = context shifted
    one position to the right (standard language-modelling objective).
    """

    def __init__(self, tokens: torch.Tensor, block_size: int):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.tokens[idx : idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

SHAKESPEARE_URLS = [
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "https://raw.githubusercontent.com/karpathy/minGPT/master/mingpt/data/shakespeare_input.txt",
]

# Compact fallback corpus used when the network is unavailable.
# Contains authentic Shakespeare passages totalling ~5 000 characters —
# enough to train a small demonstration model.
_FALLBACK_TEXT = (
    "First Citizen:\nBefore we proceed any further, hear me speak.\n\n"
    "All:\nSpeak, speak.\n\n"
    "First Citizen:\nYou are all resolved rather to die than to famish?\n\n"
    "All:\nResolved. resolved.\n\n"
    "First Citizen:\nFirst, you know Caius Marcius is chief enemy to the people.\n\n"
    "All:\nWe know't, we know't.\n\n"
    "ROMEO:\nBut, soft! what light through yonder window breaks?\n"
    "It is the east, and Juliet is the sun.\n"
    "Arise, fair sun, and kill the envious moon,\n"
    "Who is already sick and pale with grief,\n"
    "That thou her maid art far more fair than she:\n\n"
    "HAMLET:\nTo be, or not to be, that is the question:\n"
    "Whether 'tis nobler in the mind to suffer\n"
    "The slings and arrows of outrageous fortune,\n"
    "Or to take arms against a sea of troubles,\n"
    "And by opposing end them. To die: to sleep;\n"
    "No more; and by a sleep to say we end\n"
    "The heart-ache and the thousand natural shocks\n"
    "That flesh is heir to, 'tis a consummation\n"
    "Devoutly to be wish'd. To die, to sleep;\n"
    "To sleep: perchance to dream: ay, there's the rub;\n\n"
    "KING RICHARD II:\nI have been studying how I may compare\n"
    "This prison where I live unto the world:\n"
    "And for because the world is populous\n"
    "And here is not a creature but myself,\n"
    "I cannot do it; yet I'll hammer it out.\n"
    "My brain I'll prove the female to my soul,\n"
    "My soul the father; and these two beget\n"
    "A generation of still-breeding thoughts,\n"
    "And these same thoughts people this little world,\n"
    "In humours like the people of this world,\n"
    "For no thought is contented.\n\n"
    "JULIET:\nO Romeo, Romeo! wherefore art thou Romeo?\n"
    "Deny thy father and refuse thy name;\n"
    "Or, if thou wilt not, be but sworn my love,\n"
    "And I'll no longer be a Capulet.\n\n"
    "MACBETH:\nIs this a dagger which I see before me,\n"
    "The handle toward my hand? Come, let me clutch thee.\n"
    "I have thee not, and yet I see thee still.\n"
    "Art thou not, fatal vision, sensible\n"
    "To feeling as to sight? or art thou but\n"
    "A dagger of the mind, a false creation,\n"
    "Proceeding from the heat-oppressed brain?\n\n"
    "OTHELLO:\nShe loved me for the dangers I had pass'd,\n"
    "And I loved her that she did pity them.\n"
    "This only is the witchcraft I have used:\n"
    "Here comes the lady; let her witness it.\n\n"
    "KING LEAR:\nBlow, winds, and crack your cheeks! rage! blow!\n"
    "You cataracts and hurricanoes, spout\n"
    "Till you have drench'd our steeples, drown'd the cocks!\n"
    "You sulphurous and thought-executing fires,\n"
    "Vaunt-couriers to oak-cleaving thunderbolts,\n"
    "Singe my white head! And thou, all-shaking thunder,\n"
    "Smite flat the thick rotundity o' the world!\n\n"
) * 20   # repeat to get a larger corpus


def load_shakespeare(data_dir: str = "data") -> Tuple[CharTokenizer, torch.Tensor, torch.Tensor]:
    """
    Download (if needed) and load Tiny Shakespeare as character-level tokens.
    Falls back to a built-in excerpt if the network is unavailable.

    Returns:
        tokenizer  : CharTokenizer fitted on the full corpus.
        train_data : Long tensor of training tokens (~90 % of corpus).
        val_data   : Long tensor of validation tokens (~10 % of corpus).
    """
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "shakespeare.txt")

    if not os.path.exists(path):
        downloaded = False
        for url in SHAKESPEARE_URLS:
            try:
                print(f"Downloading Tiny Shakespeare → {path}")
                urllib.request.urlretrieve(url, path)
                print("Download complete.")
                downloaded = True
                break
            except Exception as e:
                print(f"  URL failed ({e}), trying next …")

        if not downloaded:
            print("  Network unavailable — using built-in Shakespeare excerpt.")
            with open(path, "w", encoding="utf-8") as f:
                f.write(_FALLBACK_TEXT)

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    split = int(0.9 * len(ids))
    return tokenizer, ids[:split], ids[split:]


def make_loaders(
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_size: int,
    batch_size: int,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Create DataLoader objects for training and validation."""
    train_ds = TextDataset(train_data, block_size)
    val_ds   = TextDataset(val_data,   block_size)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Prompt helpers
# ─────────────────────────────────────────────────────────────────────────────

BENCHMARK_PROMPTS = [
    "ROMEO: ",
    "To be, or not to be, that is the question:\nWhether ",
    "First Citizen:\nBefore we proceed any further, hear me speak.\n",
    "HAMLET: What a piece of work is man",
    "The king is dead, long live the king. ",
    "All the world's a stage,\nAnd all the men and women merely players;\n",
    "Friends, Romans, countrymen, lend me your ears;\n",
    "Now is the winter of our discontent\nMade glorious summer by this sun of York;\n",
    "JULIET: O Romeo, Romeo! wherefore art thou Romeo?\n",
    "Double, double toil and trouble;\nFire burn and cauldron bubble.\n",
]


if __name__ == "__main__":
    tok, train, val = load_shakespeare()
    print(f"Vocab size  : {tok.vocab_size}")
    print(f"Train tokens: {len(train):,}")
    print(f"Val tokens  : {len(val):,}")
    print(f"Sample decode: {tok.decode(train[:80].tolist())!r}")
