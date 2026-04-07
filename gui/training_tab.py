"""
gui/training_tab.py — Tab 3: Training for Thanos GUI.
"""

import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox

TRAINING_DATA_PATH = "training_data.txt"


class TrainingTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._stop_requested = False
        self._training_thread = None
        self._build_ui()

    def _build_ui(self):
        title = tk.Label(
            self,
            text="Training — Thanos",
            font=("Helvetica", 14, "bold"),
            fg="#9B59B6",
        )
        title.pack(pady=(10, 4))

        # Config display
        cfg_frame = ttk.LabelFrame(self, text="Model Configuration")
        cfg_frame.pack(fill=tk.X, padx=10, pady=4)
        self.config_label = ttk.Label(cfg_frame, text=self._format_config(), justify=tk.LEFT)
        self.config_label.pack(padx=8, pady=4, anchor="w")

        # Training hyperparameters
        hp_frame = ttk.LabelFrame(self, text="Training Hyperparameters")
        hp_frame.pack(fill=tk.X, padx=10, pady=4)

        fields = [
            ("Epochs", "epochs", "10"),
            ("Batch Size", "batch_size", "16"),
            ("Learning Rate", "learning_rate", "3e-4"),
            ("Weight Decay", "weight_decay", "0.01"),
            ("Val Split", "val_split", "0.1"),
        ]
        self._hp_vars = {}
        for i, (label, key, default) in enumerate(fields):
            row, col = divmod(i, 3)
            ttk.Label(hp_frame, text=label + ":").grid(row=row, column=col * 2, sticky="e", padx=6, pady=4)
            var = tk.StringVar(value=str(self.app.config.get(key, default)))
            ttk.Entry(hp_frame, textvariable=var, width=10).grid(row=row, column=col * 2 + 1, sticky="w", padx=4)
            self._hp_vars[key] = var

        # Loss / log display
        log_frame = ttk.LabelFrame(self, text="Training Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        self.log_text = tk.Text(
            log_frame,
            height=8,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=("Consolas", 10),
            bg="#1E1E2E",
            fg="#89DCEB",
            insertbackground="white",
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Progress bar
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            self, variable=self.progress_var, maximum=100
        )
        self.progress_bar.pack(fill=tk.X, padx=10, pady=4)
        self.progress_label = ttk.Label(self, text="Epoch: 0 / 0")
        self.progress_label.pack()

        # Action buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=6)

        self.train_btn = ttk.Button(
            btn_frame, text="Train from Scratch", command=self._train_from_scratch
        )
        self.train_btn.pack(side=tk.LEFT, padx=6)

        self.finetune_btn = ttk.Button(
            btn_frame, text="Fine-tune from Checkpoint", command=self._fine_tune
        )
        self.finetune_btn.pack(side=tk.LEFT, padx=6)

        self.stop_btn = ttk.Button(
            btn_frame, text="Stop Training", command=self._stop_training, state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=6)

    def _format_config(self) -> str:
        cfg = self.app.config
        return (
            f"Model: {cfg.get('model_name', 'Thanos')}  |  "
            f"Vocab: {cfg.get('vocab_size', '?')}  |  "
            f"Embed: {cfg.get('embed_dim', '?')}  |  "
            f"Heads: {cfg.get('num_heads', '?')}  |  "
            f"Layers: {cfg.get('num_layers', '?')}  |  "
            f"FFDim: {cfg.get('ff_dim', '?')}  |  "
            f"MaxSeq: {cfg.get('max_seq_len', '?')}"
        )

    def refresh_config(self):
        self.config_label.configure(text=self._format_config())

    def _get_training_config(self) -> dict:
        cfg = dict(self.app.config)
        for key, var in self._hp_vars.items():
            try:
                val = float(var.get())
                cfg[key] = int(val) if key in ("epochs", "batch_size") else val
            except ValueError:
                pass
        return cfg

    def _log(self, msg: str):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _set_training_ui(self, is_training: bool):
        state = tk.DISABLED if is_training else tk.NORMAL
        stop_state = tk.NORMAL if is_training else tk.DISABLED
        self.train_btn.configure(state=state)
        self.finetune_btn.configure(state=state)
        self.stop_btn.configure(state=stop_state)

    def _stop_training(self):
        self._stop_requested = True
        self._log("Stop requested — will stop after current epoch.")

    def _train_from_scratch(self):
        if not os.path.exists(TRAINING_DATA_PATH):
            messagebox.showwarning("No Data", f"No training data found at {TRAINING_DATA_PATH}.")
            return
        with open(TRAINING_DATA_PATH, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            messagebox.showwarning("No Data", "Training data file is empty.")
            return

        self._start_training(text, fine_tune=False)

    def _fine_tune(self):
        if not os.path.exists(TRAINING_DATA_PATH):
            messagebox.showwarning("No Data", f"No training data found at {TRAINING_DATA_PATH}.")
            return
        with open(TRAINING_DATA_PATH, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            messagebox.showwarning("No Data", "Training data file is empty.")
            return

        latest = os.path.join("checkpoints", "thanos_latest.pt")
        if not os.path.exists(latest):
            messagebox.showwarning(
                "No Checkpoint",
                "No checkpoint found. Train from scratch first.",
            )
            return

        self._start_training(text, fine_tune=True)

    def _start_training(self, text: str, fine_tune: bool):
        import json
        from model import ThanosModel
        from tokenizer_trainer import get_or_train_tokenizer, TOKENIZER_PATH
        from trainer import Trainer

        training_config = self._get_training_config()
        epochs = training_config.get("epochs", 10)

        self._stop_requested = False
        self._set_training_ui(True)
        self.progress_var.set(0)
        self.progress_label.configure(text=f"Epoch: 0 / {epochs}")
        self.app.set_status("Training..." if not fine_tune else "Fine-tuning...")

        def on_epoch_end(epoch, train_loss, val_loss):
            self.after(0, lambda: self._log(
                f"Epoch {epoch}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}"
            ))
            self.after(0, lambda: self.progress_var.set(epoch / epochs * 100))
            self.after(0, lambda: self.progress_label.configure(text=f"Epoch: {epoch} / {epochs}"))
            self.after(0, lambda: self.app.tabs["chat"].output_area)  # trigger refresh

        def on_batch_end(batch_idx, total):
            pass  # Could update sub-progress here

        def run():
            try:
                # Rebuild model with updated config
                model_config = dict(self.app.config)
                model = ThanosModel(model_config)
                tokenizer = get_or_train_tokenizer(
                    [text],
                    vocab_size=model_config.get("vocab_size", 10000),
                    path=TOKENIZER_PATH,
                )

                trainer = Trainer(
                    model=model,
                    tokenizer=tokenizer,
                    config=training_config,
                    on_epoch_end=on_epoch_end,
                    on_batch_end=on_batch_end,
                    stop_flag=lambda: self._stop_requested,
                )

                if fine_tune:
                    trainer.fine_tune(text)
                else:
                    trainer.train(text)

                # After training, update app's model and generator
                from generator import Generator
                self.after(0, lambda: self.app.load_model_from_object(model, tokenizer))
                self.after(0, lambda: self._log("Training complete! Model is ready for generation."))
            except Exception as e:
                self.after(0, lambda: self._log(f"[Error] {e}"))
            finally:
                self.after(0, lambda: self._set_training_ui(False))
                self.after(0, lambda: self.app.set_status("Idle"))

        self._training_thread = threading.Thread(target=run, daemon=True)
        self._training_thread.start()
