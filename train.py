"""
train.py — Lexi Training Loop

Provides the ``Trainer`` class that encapsulates:
* Dataset preparation (encoding a text corpus into overlapping windows).
* A standard PyTorch training loop with gradient clipping.
* **Fine-tuning mode** — load a pre-trained checkpoint, optionally
  freeze early layers, and train on new data with a lower learning rate.
* Learning-rate scheduling (cosine annealing with linear warmup).
* Gradient accumulation for effective larger batch sizes.
* Train / validation split for monitoring generalisation.
* Periodic logging of loss values.
* Checkpoint saving / loading.

The trainer is designed to be driven either from the command-line or
from the PyQt6 GUI (via a ``QThread`` wrapper that calls
``Trainer.train``).
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from device_manager import DEVICE
from model import LexiConfig, LexiModel

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

    # --- Fine-tuning ---
    fine_tune: bool = False          # enable fine-tuning mode
    freeze_layers: int = 0           # freeze the first N transformer blocks
    fine_tune_lr: float = 1e-5       # default LR when fine-tuning

    # --- Scheduler ---
    warmup_steps: int = 100          # linear warmup steps
    use_cosine_schedule: bool = True

    # --- Gradient accumulation ---
    gradient_accumulation_steps: int = 1

    # --- Validation ---
    val_split: float = 0.05          # fraction of data for validation (0 = no val)


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
# Learning-rate scheduler helpers
# ---------------------------------------------------------------------------

def _cosine_warmup_lr(step: int, warmup: int, total: int, base_lr: float) -> float:
    """Cosine annealing with linear warmup."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


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

        # Use fine-tuning LR when in fine-tune mode.
        lr = (
            self.train_config.fine_tune_lr
            if self.train_config.fine_tune
            else self.train_config.learning_rate
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
        )

    # ----- public API -------------------------------------------------------

    def prepare_dataset(
        self, corpus: str
    ) -> tuple[DataLoader, DataLoader | None]:
        """Tokenise *corpus* and return ``(train_loader, val_loader)``.

        If ``val_split > 0`` a small validation set is held out.
        """
        token_ids = self.enc.encode(corpus)
        self.log_fn(f"Corpus tokenised — {len(token_ids):,} tokens")
        full_ds = TextDataset(token_ids, self.train_config.context_length)

        val_loader = None
        if self.train_config.val_split > 0 and len(full_ds) >= 10:
            val_size = max(1, int(len(full_ds) * self.train_config.val_split))
            train_size = len(full_ds) - val_size
            train_ds, val_ds = random_split(full_ds, [train_size, val_size])
            self.log_fn(
                f"Split: {train_size:,} train / {val_size:,} validation samples"
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=self.train_config.batch_size,
                shuffle=False,
                drop_last=False,
            )
        else:
            train_ds = full_ds

        train_loader = DataLoader(
            train_ds,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            drop_last=True,
        )
        return train_loader, val_loader

    def freeze_layers(self) -> None:
        """Freeze the first *N* transformer blocks for fine-tuning."""
        n = self.train_config.freeze_layers
        if n <= 0:
            return
        n = min(n, len(self.model.blocks))
        # Freeze embeddings
        for param in self.model.tok_emb.parameters():
            param.requires_grad = False
        for param in self.model.pos_emb.parameters():
            param.requires_grad = False
        # Freeze the first N blocks
        for i in range(n):
            for param in self.model.blocks[i].parameters():
                param.requires_grad = False
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.log_fn(
            f"🔒 Froze embeddings + first {n} blocks — "
            f"{trainable:,} trainable parameters remain"
        )

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> float:
        """Run a validation pass and return average loss."""
        self.model.eval()
        total_loss = 0.0
        count = 0
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            _, loss = self.model(xb, yb)
            total_loss += loss.item()
            count += 1
        return total_loss / max(count, 1)

    def train(self, corpus: str) -> None:
        """Full training run over *corpus* for the configured epochs."""
        # Optionally freeze layers for fine-tuning.
        if self.train_config.fine_tune:
            self.freeze_layers()

        loader, val_loader = self.prepare_dataset(corpus)
        if len(loader) == 0:
            self.log_fn("⚠ Dataset is too small to form a single batch.")
            return

        accum = self.train_config.gradient_accumulation_steps
        total_steps = self.train_config.epochs * len(loader)
        base_lr = self.optimizer.param_groups[0]["lr"]
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
                loss = loss / accum  # scale for accumulation
                loss.backward()

                if step % accum == 0 or step == len(loader):
                    # LR scheduling
                    if self.train_config.use_cosine_schedule:
                        lr = _cosine_warmup_lr(
                            global_step,
                            self.train_config.warmup_steps,
                            total_steps,
                            base_lr,
                        )
                        for pg in self.optimizer.param_groups:
                            pg["lr"] = lr

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.train_config.grad_clip,
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                epoch_loss += loss.item() * accum
                global_step += 1

                if step % self.train_config.log_interval == 0:
                    self.log_fn(
                        f"  Epoch {epoch}/{self.train_config.epochs} "
                        f"Step {step}/{len(loader)} — "
                        f"loss {loss.item() * accum:.4f}"
                    )

                if self.progress_fn is not None:
                    self.progress_fn(global_step, total_steps)

            avg = epoch_loss / len(loader)
            dt = time.time() - t0

            val_msg = ""
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                val_msg = f" | val loss {val_loss:.4f}"

            self.log_fn(
                f"✓ Epoch {epoch} done — avg loss {avg:.4f}{val_msg} "
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
        """Restore model + optimiser state from *path*.

        This is used both for resuming training and as the starting
        point for fine-tuning.
        """
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)

        # Rebuild model from the saved config so architecture matches.
        saved_cfg: LexiConfig = ckpt["model_config"]
        self.model_config = saved_cfg
        self.model = LexiModel(saved_cfg).to(DEVICE)
        self.model.load_state_dict(ckpt["model_state"])

        # Only restore the optimizer if we are *not* fine-tuning
        # (fine-tuning resets the optimizer state).
        if not self.train_config.fine_tune:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.train_config.learning_rate,
            )
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.train_config.fine_tune_lr,
            )

        self.log_fn(f"📂 Checkpoint loaded ← {path}")

