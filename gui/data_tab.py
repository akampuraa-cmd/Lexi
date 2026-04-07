"""
gui/data_tab.py — Tab 2: Data Manager for Thanos GUI.
"""

import os
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox


TRAINING_DATA_PATH = "training_data.txt"


class DataTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._build_ui()

    def _build_ui(self):
        title = tk.Label(
            self,
            text="Data Manager",
            font=("Helvetica", 14, "bold"),
            fg="#9B59B6",
        )
        title.pack(pady=(10, 4))

        # Dataset info bar
        info_frame = ttk.Frame(self)
        info_frame.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(info_frame, text="Dataset file:").pack(side=tk.LEFT)
        ttk.Label(info_frame, text=TRAINING_DATA_PATH, foreground="gray").pack(side=tk.LEFT, padx=4)
        self.dataset_size_label = ttk.Label(info_frame, text="Size: 0 chars")
        self.dataset_size_label.pack(side=tk.RIGHT)
        self._update_dataset_info()

        # Paste area
        ttk.Label(self, text="Paste or type text data below:").pack(anchor="w", padx=10)
        self.paste_area = scrolledtext.ScrolledText(
            self,
            wrap=tk.WORD,
            font=("Consolas", 10),
            bg="#313244",
            fg="#CDD6F4",
            insertbackground="white",
            height=10,
        )
        self.paste_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        # Buttons row
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=10, pady=4)

        ttk.Button(btn_frame, text="Upload .txt File(s)", command=self._upload_files).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Preview Cleaned", command=self._preview_cleaned).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Add to Training Set", command=self._add_to_training_set).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Clear Input", command=self._clear_input).pack(side=tk.LEFT, padx=4)

        # Preview area
        ttk.Label(self, text="Cleaned preview:").pack(anchor="w", padx=10)
        self.preview_area = scrolledtext.ScrolledText(
            self,
            wrap=tk.WORD,
            font=("Consolas", 10),
            bg="#1E1E2E",
            fg="#A6E3A1",
            insertbackground="white",
            height=6,
            state=tk.DISABLED,
        )
        self.preview_area.pack(fill=tk.BOTH, padx=10, pady=(0, 8))

    def _update_dataset_info(self):
        if os.path.exists(TRAINING_DATA_PATH):
            size = os.path.getsize(TRAINING_DATA_PATH)
            with open(TRAINING_DATA_PATH, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            chars = len(content)
            estimated_tokens = chars // 4
            self.dataset_size_label.configure(
                text=f"Size: {size} bytes | ~{estimated_tokens} tokens"
            )
        else:
            self.dataset_size_label.configure(text="Size: 0 chars")

    def _upload_files(self):
        paths = filedialog.askopenfilenames(
            title="Select .txt files",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not paths:
            return
        combined = []
        for path in paths:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                combined.append(f.read())
        text = "\n\n".join(combined)
        self.paste_area.delete("1.0", tk.END)
        self.paste_area.insert(tk.END, text)

    def _get_raw_text(self) -> str:
        return self.paste_area.get("1.0", tk.END)

    def _preview_cleaned(self):
        raw = self._get_raw_text()
        if not raw.strip():
            messagebox.showinfo("Preview", "No text to clean.")
            return
        cleaned = self.app.data_cleaner.clean(raw)
        self.preview_area.configure(state=tk.NORMAL)
        self.preview_area.delete("1.0", tk.END)
        self.preview_area.insert(tk.END, cleaned)
        self.preview_area.configure(state=tk.DISABLED)

    def _add_to_training_set(self):
        raw = self._get_raw_text()
        if not raw.strip():
            messagebox.showwarning("No Data", "Nothing to add.")
            return
        cleaned = self.app.data_cleaner.clean(raw)
        if not cleaned:
            messagebox.showwarning("Empty", "Cleaned text is empty.")
            return
        with open(TRAINING_DATA_PATH, "a", encoding="utf-8") as f:
            f.write(cleaned + "\n")
        self._update_dataset_info()
        messagebox.showinfo("Done", f"Added {len(cleaned)} characters to training set.")
        self._clear_input()

    def _clear_input(self):
        self.paste_area.delete("1.0", tk.END)
        self.preview_area.configure(state=tk.NORMAL)
        self.preview_area.delete("1.0", tk.END)
        self.preview_area.configure(state=tk.DISABLED)

    def refresh(self):
        self._update_dataset_info()
