"""
gui/chat_tab.py — Tab 1: Chat / Text Generation for Thanos GUI.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading


class ChatTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._build_ui()

    def _build_ui(self):
        # Title
        title = tk.Label(
            self,
            text="Thanos AI — Text Generation",
            font=("Helvetica", 14, "bold"),
            fg="#9B59B6",
        )
        title.pack(pady=(10, 4))

        # Output area
        self.output_area = scrolledtext.ScrolledText(
            self,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=("Consolas", 11),
            bg="#1E1E2E",
            fg="#CDD6F4",
            insertbackground="white",
            height=18,
        )
        self.output_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=(6, 4))

        # Controls frame
        ctrl_frame = ttk.Frame(self)
        ctrl_frame.pack(fill=tk.X, padx=10, pady=4)

        # Max new tokens
        ttk.Label(ctrl_frame, text="Max Tokens:").grid(row=0, column=0, sticky="w", padx=4)
        self.max_tokens_var = tk.IntVar(value=100)
        max_tokens_slider = ttk.Scale(
            ctrl_frame, from_=10, to=500, variable=self.max_tokens_var, orient=tk.HORIZONTAL, length=120
        )
        max_tokens_slider.grid(row=0, column=1, padx=4)
        self.max_tokens_label = ttk.Label(ctrl_frame, text="100")
        self.max_tokens_label.grid(row=0, column=2, padx=4)
        max_tokens_slider.configure(command=lambda v: self.max_tokens_label.configure(text=str(int(float(v)))))

        # Temperature
        ttk.Label(ctrl_frame, text="Temperature:").grid(row=0, column=3, sticky="w", padx=4)
        self.temperature_var = tk.DoubleVar(value=1.0)
        temp_slider = ttk.Scale(
            ctrl_frame, from_=0.1, to=2.0, variable=self.temperature_var, orient=tk.HORIZONTAL, length=120
        )
        temp_slider.grid(row=0, column=4, padx=4)
        self.temp_label = ttk.Label(ctrl_frame, text="1.0")
        self.temp_label.grid(row=0, column=5, padx=4)
        temp_slider.configure(command=lambda v: self.temp_label.configure(text=f"{float(v):.1f}"))

        # Top-k
        ttk.Label(ctrl_frame, text="Top-k:").grid(row=0, column=6, sticky="w", padx=4)
        self.topk_var = tk.StringVar(value="50")
        ttk.Entry(ctrl_frame, textvariable=self.topk_var, width=6).grid(row=0, column=7, padx=4)

        # Prompt area
        prompt_frame = ttk.Frame(self)
        prompt_frame.pack(fill=tk.X, padx=10, pady=4)

        self.prompt_entry = tk.Text(
            prompt_frame,
            height=3,
            font=("Consolas", 11),
            wrap=tk.WORD,
            bg="#313244",
            fg="#CDD6F4",
            insertbackground="white",
        )
        self.prompt_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        btn_frame = ttk.Frame(prompt_frame)
        btn_frame.pack(side=tk.RIGHT)

        self.generate_btn = ttk.Button(
            btn_frame, text="Generate", command=self._on_generate
        )
        self.generate_btn.pack(fill=tk.X, pady=2)

        ttk.Button(btn_frame, text="Clear", command=self._clear_output).pack(fill=tk.X, pady=2)

    def _on_generate(self):
        prompt = self.prompt_entry.get("1.0", tk.END).strip()
        if not prompt:
            return

        if self.app.generator is None:
            self._append_output("[Thanos] No model loaded. Train or load a checkpoint first.\n")
            return

        max_new_tokens = int(self.max_tokens_var.get())
        temperature = float(self.temperature_var.get())
        try:
            top_k = int(self.topk_var.get())
        except ValueError:
            top_k = 50

        self.generate_btn.configure(state=tk.DISABLED)
        self.app.set_status("Generating...")

        def run():
            try:
                result = self.app.generator.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                )
                self.after(0, lambda: self._append_output(f"You: {prompt}\n\nThanos: {result}\n\n{'─'*60}\n\n"))
            except Exception as e:
                self.after(0, lambda: self._append_output(f"[Error] {e}\n"))
            finally:
                self.after(0, lambda: self.generate_btn.configure(state=tk.NORMAL))
                self.after(0, lambda: self.app.set_status("Idle"))

        threading.Thread(target=run, daemon=True).start()

    def _append_output(self, text: str):
        self.output_area.configure(state=tk.NORMAL)
        self.output_area.insert(tk.END, text)
        self.output_area.see(tk.END)
        self.output_area.configure(state=tk.DISABLED)

    def _clear_output(self):
        self.output_area.configure(state=tk.NORMAL)
        self.output_area.delete("1.0", tk.END)
        self.output_area.configure(state=tk.DISABLED)
