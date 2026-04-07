"""
tokenizer_trainer.py — BPE tokenizer training and loading for Thanos.

Saves/loads the tokenizer as thanos_tokenizer.json.
"""

import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

TOKENIZER_PATH = "thanos_tokenizer.json"
SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]


def train_tokenizer(texts: list[str], vocab_size: int = 10000, save_path: str = TOKENIZER_PATH) -> Tokenizer:
    """Train a BPE tokenizer on the provided list of text strings and save it."""
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
    )

    tokenizer.train_from_iterator(texts, trainer=trainer)

    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[
            ("[BOS]", tokenizer.token_to_id("[BOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ],
    )

    tokenizer.save(save_path)
    return tokenizer


def load_tokenizer(path: str = TOKENIZER_PATH) -> Tokenizer:
    """Load a previously saved tokenizer."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tokenizer not found at {path}. Train it first.")
    return Tokenizer.from_file(path)


def get_or_train_tokenizer(
    texts: list[str],
    vocab_size: int = 10000,
    path: str = TOKENIZER_PATH,
    force_retrain: bool = False,
) -> Tokenizer:
    """Load tokenizer if it exists, otherwise train and save a new one."""
    if os.path.exists(path) and not force_retrain:
        return load_tokenizer(path)
    return train_tokenizer(texts, vocab_size=vocab_size, save_path=path)
