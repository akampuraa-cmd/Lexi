"""
gui/settings_tab.py — Tab 4: Model Settings for Thanos GUI.
"""

import os
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import torch


class SettingsTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._build_ui()

    def _build_ui(self):
        title = tk.Label(
            self,
            text="Model Settings — Thanos",
            font=("Helvetica", 14, "bold"),
            fg="#9B59B6",
        )
        title.pack(pady=(10, 4))

        # Model hyperparameter fields
        hp_frame = ttk.LabelFrame(self, text="Model Hyperparameters")
        hp_frame.pack(fill=tk.X, padx=14, pady=6)

        fields = [
            ("Vocab Size", "vocab_size"),
            ("Embed Dim", "embed_dim"),
            ("Num Heads", "num_heads"),
            ("Num Layers", "num_layers"),
            ("FF Dim", "ff_dim"),
            ("Max Seq Len", "max_seq_len"),
            ("Dropout", "dropout"),
        ]
        self._vars = {}
        for i, (label, key) in enumerate(fields):
            row, col = divmod(i, 2)
            ttk.Label(hp_frame, text=label + ":").grid(row=row, column=col * 2, sticky="e", padx=8, pady=6)
            var = tk.StringVar(value=str(self.app.config.get(key, "")))
            ttk.Entry(hp_frame, textvariable=var, width=12).grid(row=row, column=col * 2 + 1, sticky="w", padx=6)
            self._vars[key] = var

        # Save Config button
        ttk.Button(hp_frame, text="Save Config", command=self._save_config).grid(
            row=len(fields), column=0, columnspan=4, pady=8
        )

        # Checkpoint management
        ckpt_frame = ttk.LabelFrame(self, text="Checkpoint Management")
        ckpt_frame.pack(fill=tk.X, padx=14, pady=6)

        ttk.Button(ckpt_frame, text="Load Checkpoint (.pt)", command=self._load_checkpoint).pack(
            side=tk.LEFT, padx=8, pady=8
        )
        ttk.Button(ckpt_frame, text="Export Model Weights", command=self._export_model).pack(
            side=tk.LEFT, padx=8, pady=8
        )

        self.ckpt_label = ttk.Label(ckpt_frame, text="No checkpoint loaded.", foreground="gray")
        self.ckpt_label.pack(side=tk.LEFT, padx=8)

        # Model info
        info_frame = ttk.LabelFrame(self, text="Model Info")
        info_frame.pack(fill=tk.X, padx=14, pady=6)
        self.model_info_label = ttk.Label(
            info_frame,
            text=self._model_info_text(),
            justify=tk.LEFT,
            font=("Consolas", 10),
        )
        self.model_info_label.pack(padx=8, pady=6, anchor="w")

        ttk.Button(info_frame, text="Refresh Info", command=self._refresh_info).pack(padx=8, pady=4)

    def _model_info_text(self) -> str:
        model = self.app.model
        if model is None:
            return "No model loaded."
        total = model.num_parameters()
        cfg = model.config
        return (
            f"Parameters: {total:,}\n"
            f"Layers: {cfg.get('num_layers')} | Heads: {cfg.get('num_heads')} | "
            f"Embed: {cfg.get('embed_dim')} | FF: {cfg.get('ff_dim')}\n"
            f"Vocab: {cfg.get('vocab_size')} | Max Seq: {cfg.get('max_seq_len')}"
        )

    def _refresh_info(self):
        self.model_info_label.configure(text=self._model_info_text())

    def _save_config(self):
        for key, var in self._vars.items():
            try:
                val = float(var.get())
                self.app.config[key] = int(val) if key != "dropout" else val
            except ValueError:
                messagebox.showerror("Invalid Value", f"Invalid value for {key}: {var.get()}")
                return

        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(self.app.config, f, indent=2)
        messagebox.showinfo("Saved", "Config saved to config.json")

        # Update training tab display
        if "training" in self.app.tabs:
            self.app.tabs["training"].refresh_config()

    def _load_checkpoint(self):
        path = filedialog.askopenfilename(
            title="Load Thanos Checkpoint",
            filetypes=[("PyTorch checkpoints", "*.pt"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            from model import ThanosModel
            from tokenizer_trainer import load_tokenizer, TOKENIZER_PATH

            ckpt = torch.load(path, map_location="cpu")
            model_cfg = ckpt.get("config", self.app.config)
            model = ThanosModel(model_cfg)
            model.load_state_dict(ckpt["model_state_dict"])

            tokenizer = None
            if os.path.exists(TOKENIZER_PATH):
                tokenizer = load_tokenizer(TOKENIZER_PATH)

            self.app.load_model_from_object(model, tokenizer)
            self.ckpt_label.configure(
                text=f"Loaded: {os.path.basename(path)} (epoch {ckpt.get('epoch', '?')})",
                foreground="green",
            )
            self._refresh_info()
            messagebox.showinfo("Loaded", f"Checkpoint loaded from:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load checkpoint:\n{e}")

    def _export_model(self):
        if self.app.model is None:
            messagebox.showwarning("No Model", "No model is currently loaded.")
            return
        path = filedialog.asksaveasfilename(
            title="Export Thanos Model",
            defaultextension=".pt",
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")],
            initialfile="thanos_export.pt",
        )
        if not path:
            return
        try:
            torch.save(
                {
                    "model_state_dict": self.app.model.state_dict(),
                    "config": self.app.model.config,
                },
                path,
            )
            messagebox.showinfo("Exported", f"Model exported to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{e}")

    def refresh(self):
        for key, var in self._vars.items():
            var.set(str(self.app.config.get(key, "")))
        self._refresh_info()
