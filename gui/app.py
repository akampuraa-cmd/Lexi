"""
gui/app.py — Main Tkinter application for Thanos AI.
"""

import os
import json
import tkinter as tk
from tkinter import ttk

from gui.chat_tab import ChatTab
from gui.data_tab import DataTab
from gui.training_tab import TrainingTab
from gui.settings_tab import SettingsTab
from data_cleaner import DataCleaner

CONFIG_PATH = "config.json"

DEFAULT_CONFIG = {
    "model_name": "Thanos",
    "vocab_size": 10000,
    "embed_dim": 256,
    "num_heads": 8,
    "num_layers": 6,
    "ff_dim": 1024,
    "max_seq_len": 512,
    "dropout": 0.1,
    "epochs": 10,
    "batch_size": 16,
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "val_split": 0.1,
}


class ThanosApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Thanos AI — Local Text Generation")
        self.geometry("960x680")
        self.resizable(True, True)
        self.configure(bg="#1E1E2E")

        # Load config
        self.config = self._load_config()

        # Shared state
        self.model = None
        self.tokenizer = None
        self.generator = None
        self.data_cleaner = DataCleaner()

        # Try to load latest checkpoint and tokenizer on startup
        self._try_load_on_startup()

        # Style
        self._apply_style()

        # Build UI
        self._build_ui()

    def _load_config(self) -> dict:
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return dict(DEFAULT_CONFIG)

    def _apply_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook", background="#1E1E2E", borderwidth=0)
        style.configure("TNotebook.Tab", background="#313244", foreground="#CDD6F4", padding=[12, 6])
        style.map("TNotebook.Tab", background=[("selected", "#9B59B6")], foreground=[("selected", "white")])
        style.configure("TFrame", background="#1E1E2E")
        style.configure("TLabel", background="#1E1E2E", foreground="#CDD6F4")
        style.configure("TLabelframe", background="#1E1E2E", foreground="#CDD6F4")
        style.configure("TLabelframe.Label", background="#1E1E2E", foreground="#9B59B6")
        style.configure("TButton", background="#9B59B6", foreground="white", padding=[8, 4])
        style.map("TButton", background=[("active", "#7D3C98")])
        style.configure("TEntry", fieldbackground="#313244", foreground="#CDD6F4")
        style.configure("TProgressbar", troughcolor="#313244", background="#9B59B6")

    def _build_ui(self):
        # Header
        header = tk.Frame(self, bg="#181825", pady=8)
        header.pack(fill=tk.X)
        tk.Label(
            header,
            text="⚡ Thanos AI",
            font=("Helvetica", 18, "bold"),
            bg="#181825",
            fg="#9B59B6",
        ).pack(side=tk.LEFT, padx=16)

        self.status_label = tk.Label(
            header,
            text="Status: Idle",
            font=("Helvetica", 10),
            bg="#181825",
            fg="#A6ADC8",
        )
        self.status_label.pack(side=tk.RIGHT, padx=16)

        # Notebook
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.tabs = {}

        chat = ChatTab(self.notebook, self)
        self.notebook.add(chat, text="  💬 Chat  ")
        self.tabs["chat"] = chat

        data = DataTab(self.notebook, self)
        self.notebook.add(data, text="  📂 Data  ")
        self.tabs["data"] = data

        training = TrainingTab(self.notebook, self)
        self.notebook.add(training, text="  🔥 Training  ")
        self.tabs["training"] = training

        settings = SettingsTab(self.notebook, self)
        self.notebook.add(settings, text="  ⚙ Settings  ")
        self.tabs["settings"] = settings

    def set_status(self, text: str):
        self.status_label.configure(text=f"Status: {text}")

    def load_model_from_object(self, model, tokenizer):
        """Update the app's active model and generator."""
        from generator import Generator
        self.model = model
        self.tokenizer = tokenizer
        if tokenizer is not None:
            self.generator = Generator(model, tokenizer)
        else:
            self.generator = None
        if "settings" in self.tabs:
            self.tabs["settings"].refresh()

    def _try_load_on_startup(self):
        """Silently try to load tokenizer and latest checkpoint on startup."""
        try:
            from tokenizer_trainer import load_tokenizer, TOKENIZER_PATH
            from model import ThanosModel
            import torch

            tokenizer = None
            if os.path.exists(TOKENIZER_PATH):
                tokenizer = load_tokenizer(TOKENIZER_PATH)

            latest = os.path.join("checkpoints", "thanos_latest.pt")
            if os.path.exists(latest):
                ckpt = torch.load(latest, map_location="cpu")
                model_cfg = ckpt.get("config", self.config)
                model = ThanosModel(model_cfg)
                model.load_state_dict(ckpt["model_state_dict"])
                self.load_model_from_object(model, tokenizer)
        except Exception:
            pass  # First run — no checkpoint yet
