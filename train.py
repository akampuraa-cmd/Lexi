"""
train.py — Lexi Training Loop

Provides the ``Trainer`` class that encapsulates:
* Dataset preparation (encoding a text corpus into overlapping windows).
* A standard PyTorch training loop with gradient clipping.
* Periodic logging of loss values.
* Checkpoint saving / loading.

The trainer is designed to be driven either from the command-line or
from the PyQt6 GUI (via a ``QThread`` wrapper that calls
``Trainer.run_epoch``).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset

from model import DEVICE, LexiConfig, LexiModel

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Training hyper-parameters, adjustable from the GUI."""

    batch_size: int = 16
    learning_rate: float = 3e-4
    epochs: int = 5
    context_length: int = 256
    grad_clip: float = 1.0
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 50          # print loss every N steps


# ---------------------------------------------------------------------------
# Dataset — sliding-window over token IDs
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    """Converts a flat list of token IDs into (input, target) pairs.

    Each sample is a window of ``context_length`` tokens; the target
    is the same window shifted right by one position.  This is the
    standard language-modelling training scheme.
    """

    def __init__(self, token_ids: list[int], context_length: int) -> None:
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.context_length = context_length

    def __len__(self) -> int:
        # Number of non-overlapping windows.  We need context_length + 1
        # tokens per sample (input + 1 token for the last target).
        return max(0, len(self.data) - self.context_length)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.context_length + 1]
        x = chunk[:-1]    # input
        y = chunk[1:]      # target (shifted by 1)
        return x, y


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Orchestrates model creation, dataset preparation, and training.

    Parameters
    ----------
    model_config : LexiConfig
        Architecture configuration.
    train_config : TrainConfig
        Training hyperparameters.
    log_fn : callable, optional
        A function ``(str) -> None`` used to emit log messages (e.g.
        into a GUI text widget).  Falls back to ``print`` if *None*.
    progress_fn : callable, optional
        A function ``(int, int) -> None`` receiving ``(current_step,
        total_steps)`` for progress-bar updates in the GUI.
    stop_flag : callable, optional
        A function ``() -> bool``.  If it returns ``True`` the
        training loop exits early (used by the GUI stop button).
    """

    def __init__(
        self,
        model_config: LexiConfig | None = None,
        train_config: TrainConfig | None = None,
        log_fn: Callable[[str], None] | None = None,
        progress_fn: Callable[[int, int], None] | None = None,
        stop_flag: Callable[[], bool] | None = None,
    ) -> None:
        self.model_config = model_config or LexiConfig()
        self.train_config = train_config or TrainConfig()
        self.log_fn = log_fn or print
        self.progress_fn = progress_fn
        self.stop_flag = stop_flag or (lambda: False)

        # Ensure context lengths are consistent.
        self.model_config.context_length = self.train_config.context_length

        # Tokeniser — GPT-2 BPE via *tiktoken*.
        self.enc = tiktoken.get_encoding("gpt2")
        self.model_config.vocab_size = self.enc.n_vocab

        # Build model and move to the detected device.
        self.model = LexiModel(self.model_config).to(DEVICE)
        self.log_fn(
            f"Lexi model created — {self.model.num_params():,} parameters "
            f"on {DEVICE}"
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_config.learning_rate,
        )

    # ----- public API -------------------------------------------------------

    def prepare_dataset(self, corpus: str) -> DataLoader:
        """Tokenise *corpus* and return a ``DataLoader``."""
        token_ids = self.enc.encode(corpus)
        self.log_fn(f"Corpus tokenised — {len(token_ids):,} tokens")
        ds = TextDataset(token_ids, self.train_config.context_length)
        return DataLoader(
            ds,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def train(self, corpus: str) -> None:
        """Full training run over *corpus* for the configured epochs."""
        loader = self.prepare_dataset(corpus)
        if len(loader) == 0:
            self.log_fn("⚠ Dataset is too small to form a single batch.")
            return

        total_steps = self.train_config.epochs * len(loader)
        global_step = 0

        for epoch in range(1, self.train_config.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            t0 = time.time()

            for step, (xb, yb) in enumerate(loader, 1):
                if self.stop_flag():
                    self.log_fn("⛔ Training stopped by user.")
                    return

                xb, yb = xb.to(DEVICE), yb.to(DEVICE)

                _, loss = self.model(xb, yb)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_config.grad_clip,
                )
                self.optimizer.step()

                epoch_loss += loss.item()
                global_step += 1

                if step % self.train_config.log_interval == 0:
                    self.log_fn(
                        f"  Epoch {epoch}/{self.train_config.epochs} "
                        f"Step {step}/{len(loader)} — "
                        f"loss {loss.item():.4f}"
                    )

                if self.progress_fn is not None:
                    self.progress_fn(global_step, total_steps)

            avg = epoch_loss / len(loader)
            dt = time.time() - t0
            self.log_fn(
                f"✓ Epoch {epoch} done — avg loss {avg:.4f} "
                f"({dt:.1f}s)"
            )
            self.save_checkpoint(tag=f"epoch_{epoch}")

        self.save_checkpoint(tag="latest")
        self.log_fn("✅ Training complete.")

    # ----- checkpoints ------------------------------------------------------

    def save_checkpoint(self, tag: str = "latest") -> None:
        """Save model + optimiser state to disk."""
        ckpt_dir = Path(self.train_config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"lexi_{tag}.pt"
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "model_config": self.model_config,
                "train_config": self.train_config,
            },
            path,
        )
        self.log_fn(f"💾 Checkpoint saved → {path}")

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore model + optimiser state from *path*."""
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.log_fn(f"📂 Checkpoint loaded ← {path}")
