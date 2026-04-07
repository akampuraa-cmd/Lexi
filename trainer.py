"""
trainer.py — Trainer class for Thanos: training from scratch and fine-tuning.

Features:
  - Splits data into sequences of max_seq_len tokens
  - Next-token prediction with CrossEntropyLoss
  - AdamW optimizer + CosineAnnealingLR scheduler
  - Checkpoint saving to ./checkpoints/ after each epoch
  - Training log to training_log.txt
  - Supports graceful stop via stop_flag
"""

import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

CHECKPOINTS_DIR = "checkpoints"
TRAINING_LOG_PATH = "training_log.txt"


class TokenDataset(Dataset):
    """Dataset that creates (input, target) pairs for next-token prediction."""

    def __init__(self, token_ids: list[int], seq_len: int):
        self.seq_len = seq_len
        self.samples = []
        for i in range(0, len(token_ids) - seq_len, seq_len):
            chunk = token_ids[i: i + seq_len + 1]
            if len(chunk) == seq_len + 1:
                self.samples.append(chunk)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk = self.samples[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


class Trainer:
    """Handles training and fine-tuning of the ThanosModel."""

    def __init__(
        self,
        model,
        tokenizer,
        config: dict,
        on_epoch_end=None,
        on_batch_end=None,
        stop_flag=None,
    ):
        """
        Args:
            model: ThanosModel instance
            tokenizer: trained HF tokenizers.Tokenizer
            config: dict with training hyperparameters
            on_epoch_end: callback(epoch, train_loss, val_loss) called after each epoch
            on_batch_end: callback(batch_idx, total_batches) for progress updates
            stop_flag: callable that returns True when training should stop
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.on_epoch_end = on_epoch_end
        self.on_batch_end = on_batch_end
        self.stop_flag = stop_flag or (lambda: False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    def _tokenize(self, text: str) -> list[int]:
        encoding = self.tokenizer.encode(text)
        return encoding.ids

    def _build_loaders(self, token_ids: list[int]):
        seq_len = self.config.get("max_seq_len", 512)
        batch_size = self.config.get("batch_size", 16)
        val_split = self.config.get("val_split", 0.1)

        dataset = TokenDataset(token_ids, seq_len)
        if len(dataset) == 0:
            raise ValueError("Not enough data to create training samples. Add more text data.")

        val_size = max(1, int(len(dataset) * val_split))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
        return train_loader, val_loader

    def _save_checkpoint(self, epoch: int):
        path = os.path.join(CHECKPOINTS_DIR, f"thanos_epoch_{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "config": self.model.config,
            },
            path,
        )
        # Also save as latest
        latest_path = os.path.join(CHECKPOINTS_DIR, "thanos_latest.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "config": self.model.config,
            },
            latest_path,
        )
        return path

    def _log(self, message: str):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line)
        with open(TRAINING_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _run_epoch(self, loader, optimizer, criterion, training: bool):
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_batches = len(loader)

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for batch_idx, (x, y) in enumerate(loader):
                if self.stop_flag():
                    break

                x, y = x.to(self.device), y.to(self.device)

                logits = self.model(x)
                B, T, V = logits.shape
                loss = criterion(logits.view(B * T, V), y.view(B * T))

                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                total_loss += loss.item()

                if self.on_batch_end and training:
                    self.on_batch_end(batch_idx + 1, total_batches)

        return total_loss / max(total_batches, 1)

    def train(self, text: str):
        """Train from scratch or continue training on the provided text."""
        self._log("=== Thanos Training Started ===")
        token_ids = self._tokenize(text)
        self._log(f"Total tokens: {len(token_ids)}")

        train_loader, val_loader = self._build_loaders(token_ids)

        epochs = self.config.get("epochs", 10)
        lr = self.config.get("learning_rate", 3e-4)
        weight_decay = self.config.get("weight_decay", 0.01)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, epochs + 1):
            if self.stop_flag():
                self._log("Training stopped by user.")
                break

            train_loss = self._run_epoch(train_loader, optimizer, criterion, training=True)
            val_loss = self._run_epoch(val_loader, optimizer, criterion, training=False)
            scheduler.step()

            self._log(f"Epoch {epoch}/{epochs} — Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            ckpt_path = self._save_checkpoint(epoch)
            self._log(f"Checkpoint saved: {ckpt_path}")

            if self.on_epoch_end:
                self.on_epoch_end(epoch, train_loss, val_loss)

        self._log("=== Thanos Training Complete ===")

    def fine_tune(self, text: str, checkpoint_path: str = None):
        """Fine-tune from an existing checkpoint."""
        if checkpoint_path is None:
            checkpoint_path = os.path.join(CHECKPOINTS_DIR, "thanos_latest.pt")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self._log(f"Loaded checkpoint: {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")

        # Use lower learning rate for fine-tuning
        import copy
        ft_lr = self.config.get("learning_rate", 3e-4) * 0.1
        self.config = copy.deepcopy(self.config)
        self.config["learning_rate"] = ft_lr

        self._log(f"Fine-tuning with lr={ft_lr}")
        self.train(text)
