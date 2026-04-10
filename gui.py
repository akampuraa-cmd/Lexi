"""
gui.py — Lexi Native Windows GUI (PyQt6)

Provides the main application window with four tabs:

1. **Chat / Interaction** — talk to the trained Lexi model.
2. **Data Feeding** — load local files (.txt, .json, .jsonl, .csv) or
   paste text directly into the training corpus.
3. **Web Scraping** — enter a URL, scrape & clean text, preview it,
   and optionally add it to the corpus.
4. **Training / Settings** — configure hyperparameters for pre-training
   or fine-tuning, view hardware status, and start / stop training.

Threading strategy
~~~~~~~~~~~~~~~~~~
Heavy work (web scraping, training) is offloaded to ``QThread``
workers so the Qt event loop stays responsive.  Communication
between threads and the GUI uses Qt **signals & slots**, which are
thread-safe by design.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import tiktoken
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QMessageBox,
)

from data_cleaner import DataCleaner
from data_pipeline import SUPPORTED_EXTENSIONS, load_file
from device_manager import DEVICE, get_device_summary
from generator import Generator
from model import LexiConfig, LexiModel
from scraper import scrape_url
from train import Trainer, TrainConfig

# ---------------------------------------------------------------------------
# QThread workers
# ---------------------------------------------------------------------------

class ScrapeWorker(QThread):
    """Background thread that scrapes a URL and emits the cleaned text."""

    finished = pyqtSignal(str)     # cleaned text
    error = pyqtSignal(str)        # error message

    def __init__(self, url: str) -> None:
        super().__init__()
        self.url = url

    def run(self) -> None:
        try:
            text = scrape_url(self.url)
            self.finished.emit(text)
        except Exception as exc:
            self.error.emit(str(exc))


class TrainWorker(QThread):
    """Background thread that runs the training loop."""

    log = pyqtSignal(str)
    progress = pyqtSignal(int, int)       # (current_step, total_steps)
    done = pyqtSignal()

    def __init__(self, trainer: Trainer, corpus: str) -> None:
        super().__init__()
        self.trainer = trainer
        self.corpus = corpus
        self._stop = False

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        # Wire log/progress callbacks into signals.
        self.trainer.log_fn = lambda msg: self.log.emit(msg)
        self.trainer.progress_fn = lambda cur, tot: self.progress.emit(cur, tot)
        self.trainer.stop_flag = lambda: self._stop
        self.trainer.train(self.corpus)
        self.done.emit()


class GenerateWorker(QThread):
    """Background thread for text generation."""

    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, generator: Generator, prompt: str, max_tokens: int) -> None:
        super().__init__()
        self.generator = generator
        self.prompt = prompt
        self.max_tokens = max_tokens

    def run(self) -> None:
        try:
            text = self.generator.generate(self.prompt, max_new_tokens=self.max_tokens)
            self.finished.emit(text)
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

class LexiMainWindow(QMainWindow):
    """The primary application window for Lexi."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Lexi AI")
        self.setMinimumSize(900, 650)

        # Shared state -------------------------------------------------------
        self.corpus: str = ""               # accumulated training text
        self._files_loaded: int = 0         # number of data files loaded
        self.trainer: Trainer | None = None
        self.generator: Generator | None = None
        self._train_worker: TrainWorker | None = None
        self._scrape_worker: ScrapeWorker | None = None
        self._gen_worker: GenerateWorker | None = None

        # Tabs ----------------------------------------------------------------
        tabs = QTabWidget()
        tabs.addTab(self._build_chat_tab(), "💬 Chat")
        tabs.addTab(self._build_data_tab(), "📂 Data Feeding")
        tabs.addTab(self._build_scrape_tab(), "🌐 Web Scraping")
        tabs.addTab(self._build_train_tab(), "⚙️ Training / Settings")
        self.setCentralWidget(tabs)

    # ====================================================================
    # TAB 1 — Chat / Interaction
    # ====================================================================
    def _build_chat_tab(self) -> QWidget:
        """Build the Chat tab widget.

        Layout:
        - A read-only text area showing the conversation.
        - A text input + "Send" button at the bottom.
        """
        w = QWidget()
        layout = QVBoxLayout(w)

        self.chat_output = QPlainTextEdit()
        self.chat_output.setReadOnly(True)
        self.chat_output.setPlaceholderText(
            "Lexi's responses will appear here…\n\n"
            "Train a model first, then type a prompt below."
        )
        layout.addWidget(self.chat_output)

        row = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Type your message…")
        self.chat_input.returnPressed.connect(self._on_chat_send)
        row.addWidget(self.chat_input)

        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self._on_chat_send)
        row.addWidget(send_btn)

        layout.addLayout(row)
        return w

    def _on_chat_send(self) -> None:
        """Handle the Send button / Enter key in the chat tab."""
        prompt = self.chat_input.text().strip()
        if not prompt:
            return

        self.chat_output.appendPlainText(f"You: {prompt}")
        self.chat_input.clear()

        if self.generator is None:
            self.chat_output.appendPlainText(
                "Lexi: [No model loaded — please train first.]\n"
            )
            return

        # Run generation in background to keep the GUI responsive.
        self._gen_worker = GenerateWorker(self.generator, prompt, max_tokens=256)
        self._gen_worker.finished.connect(self._on_generate_done)
        self._gen_worker.error.connect(
            lambda err: self.chat_output.appendPlainText(f"Lexi: [Error: {err}]\n")
        )
        self._gen_worker.start()

    def _on_generate_done(self, text: str) -> None:
        self.chat_output.appendPlainText(f"Lexi: {text}\n")

    # ====================================================================
    # TAB 2 — Data Feeding
    # ====================================================================
    def _build_data_tab(self) -> QWidget:
        """Build the Data Feeding tab widget.

        Layout:
        - A button to browse for data files (.txt, .json, .jsonl, .csv).
        - A preview of the cleaned text.
        - A label showing corpus stats.
        """
        w = QWidget()
        layout = QVBoxLayout(w)

        btn_row = QHBoxLayout()
        btn = QPushButton("📁 Load Data File(s)…")
        btn.setToolTip("Supported: .txt, .json, .jsonl, .csv")
        btn.clicked.connect(self._on_load_files)
        btn_row.addWidget(btn)

        paste_btn = QPushButton("📋 Paste Text")
        paste_btn.setToolTip("Paste text directly into the training corpus")
        paste_btn.clicked.connect(self._on_paste_text)
        btn_row.addWidget(paste_btn)

        clear_btn = QPushButton("🗑 Clear Corpus")
        clear_btn.clicked.connect(self._on_clear_corpus)
        btn_row.addWidget(clear_btn)

        layout.addLayout(btn_row)

        self.data_preview = QPlainTextEdit()
        self.data_preview.setReadOnly(True)
        self.data_preview.setPlaceholderText(
            "Loaded data preview…\n\n"
            "Supported formats:\n"
            "  • .txt  — plain text\n"
            "  • .json — JSON array/object with prompt/completion pairs\n"
            "  • .jsonl — one JSON object per line\n"
            "  • .csv  — CSV with header row\n\n"
            "For fine-tuning, use files with paired columns:\n"
            "  prompt/completion, input/output, question/answer, instruction/response"
        )
        layout.addWidget(self.data_preview)

        self.corpus_label = QLabel("Corpus: 0 characters | 0 files loaded")
        layout.addWidget(self.corpus_label)

        return w

    def _on_load_files(self) -> None:
        """Open a file dialog, clean each file, and append to corpus."""
        ext_filter = (
            "All Supported (*.txt *.json *.jsonl *.csv);;"
            "Text Files (*.txt);;"
            "JSON Files (*.json);;"
            "JSONL Files (*.jsonl);;"
            "CSV Files (*.csv);;"
            "All Files (*)"
        )
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select data files", "", ext_filter
        )
        if not paths:
            return

        for p in paths:
            try:
                loaded = load_file(p)
                if loaded.text:
                    self.corpus += " " + loaded.text
                    self._files_loaded += 1
                    self.data_preview.appendPlainText(
                        f"--- {os.path.basename(p)} [{loaded.format}] "
                        f"({loaded.num_samples} samples) ---\n"
                        f"{loaded.text[:500]}…\n"
                    )
            except Exception as exc:
                self.data_preview.appendPlainText(
                    f"⚠ Error loading {os.path.basename(p)}: {exc}\n"
                )

        self.corpus = self.corpus.strip()
        self._update_corpus_label()

    def _on_paste_text(self) -> None:
        """Open a dialog to paste text directly."""
        from PyQt6.QtWidgets import QInputDialog
        text, ok = QInputDialog.getMultiLineText(
            self, "Paste Training Text", "Enter or paste text below:"
        )
        if ok and text.strip():
            cleaned = DataCleaner.clean(text)
            self.corpus += " " + cleaned
            self.corpus = self.corpus.strip()
            self._files_loaded += 1
            self.data_preview.appendPlainText(
                f"--- [pasted text] ---\n{cleaned[:500]}…\n"
            )
            self._update_corpus_label()

    def _on_clear_corpus(self) -> None:
        """Clear all loaded data."""
        if not self.corpus:
            return
        reply = QMessageBox.question(
            self, "Clear Corpus",
            "Are you sure you want to clear all loaded training data?",
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.corpus = ""
            self._files_loaded = 0
            self.data_preview.clear()
            self._update_corpus_label()

    def _update_corpus_label(self) -> None:
        self.corpus_label.setText(
            f"Corpus: {len(self.corpus):,} characters | "
            f"{self._files_loaded} file(s) loaded"
        )

    # ====================================================================
    # TAB 3 — Web Scraping
    # ====================================================================
    def _build_scrape_tab(self) -> QWidget:
        """Build the Web Scraping tab widget.

        Layout:
        - URL input + Scrape button.
        - A preview text area showing cleaned scraped text.
        - An "Authorize & Add to Dataset" button.
        """
        w = QWidget()
        layout = QVBoxLayout(w)

        row = QHBoxLayout()
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("https://example.com")
        row.addWidget(self.url_input)

        scrape_btn = QPushButton("🔍 Scrape")
        scrape_btn.clicked.connect(self._on_scrape)
        row.addWidget(scrape_btn)
        layout.addLayout(row)

        self.scrape_preview = QPlainTextEdit()
        self.scrape_preview.setReadOnly(True)
        self.scrape_preview.setPlaceholderText(
            "Scraped & cleaned text will appear here…"
        )
        layout.addWidget(self.scrape_preview)

        auth_btn = QPushButton("✅ Authorize && Add to Dataset")
        auth_btn.clicked.connect(self._on_authorize_scrape)
        layout.addWidget(auth_btn)

        return w

    def _on_scrape(self) -> None:
        """Kick off background scraping."""
        url = self.url_input.text().strip()
        if not url:
            return

        self.scrape_preview.setPlainText("Scraping…")
        self._scrape_worker = ScrapeWorker(url)
        self._scrape_worker.finished.connect(self._on_scrape_done)
        self._scrape_worker.error.connect(
            lambda err: self.scrape_preview.setPlainText(f"Error: {err}")
        )
        self._scrape_worker.start()

    def _on_scrape_done(self, text: str) -> None:
        self.scrape_preview.setPlainText(text)

    def _on_authorize_scrape(self) -> None:
        """Append the previewed scraped text to the training corpus."""
        text = self.scrape_preview.toPlainText().strip()
        if not text or text.startswith("Error:") or text == "Scraping…":
            QMessageBox.warning(self, "Nothing to add", "Scrape a URL first.")
            return

        self.corpus += " " + text
        self.corpus = self.corpus.strip()
        self._files_loaded += 1
        self._update_corpus_label()
        QMessageBox.information(
            self, "Added", "Scraped text has been added to the training corpus."
        )

    # ====================================================================
    # TAB 4 — Training / Settings
    # ====================================================================
    def _build_train_tab(self) -> QWidget:
        """Build the Training / Settings tab.

        Layout:
        - Hardware info panel.
        - Training mode selector (pre-training vs fine-tuning).
        - Hyperparameter controls.
        - Fine-tuning specific controls.
        - Start / Stop training buttons.
        - A log output area and a progress bar.
        """
        w = QWidget()
        layout = QVBoxLayout(w)

        # ----- Hardware info -----------------------------------------------
        hw_group = QGroupBox("🖥 Hardware Detection")
        hw_layout = QVBoxLayout(hw_group)
        hw_text = QPlainTextEdit()
        hw_text.setReadOnly(True)
        hw_text.setMaximumHeight(110)
        hw_text.setPlainText(get_device_summary())
        hw_layout.addWidget(hw_text)
        layout.addWidget(hw_group)

        # ----- Training mode -----------------------------------------------
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Training Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Pre-training (from scratch)", "Fine-tuning (from checkpoint)"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.mode_combo)
        layout.addLayout(mode_row)

        # ----- Checkpoint loading (for fine-tuning) -------------------------
        self.ft_group = QGroupBox("Fine-Tuning Settings")
        ft_layout = QVBoxLayout(self.ft_group)

        ckpt_row = QHBoxLayout()
        ckpt_row.addWidget(QLabel("Base Checkpoint:"))
        self.ckpt_path_edit = QLineEdit()
        self.ckpt_path_edit.setPlaceholderText("Path to .pt checkpoint file…")
        ckpt_row.addWidget(self.ckpt_path_edit)
        ckpt_browse = QPushButton("Browse…")
        ckpt_browse.clicked.connect(self._on_browse_checkpoint)
        ckpt_row.addWidget(ckpt_browse)
        ft_layout.addLayout(ckpt_row)

        freeze_row = QHBoxLayout()
        freeze_row.addWidget(QLabel("Freeze First N Layers:"))
        self.spin_freeze = QSpinBox()
        self.spin_freeze.setRange(0, 12)
        self.spin_freeze.setValue(2)
        self.spin_freeze.setToolTip(
            "Freeze the first N transformer blocks (and embeddings) "
            "so only later layers are updated."
        )
        freeze_row.addWidget(self.spin_freeze)

        freeze_row.addWidget(QLabel("Fine-Tune LR:"))
        self.spin_ft_lr = QDoubleSpinBox()
        self.spin_ft_lr.setDecimals(7)
        self.spin_ft_lr.setRange(1e-7, 1e-2)
        self.spin_ft_lr.setSingleStep(1e-5)
        self.spin_ft_lr.setValue(1e-5)
        freeze_row.addWidget(self.spin_ft_lr)
        ft_layout.addLayout(freeze_row)

        self.ft_group.setVisible(False)
        layout.addWidget(self.ft_group)

        # ----- Hyperparameters row -----------------------------------------
        hp_group = QGroupBox("Hyperparameters")
        hp_layout = QVBoxLayout(hp_group)
        hp_row = QHBoxLayout()

        hp_row.addWidget(QLabel("Batch Size:"))
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 512)
        self.spin_batch.setValue(16)
        hp_row.addWidget(self.spin_batch)

        hp_row.addWidget(QLabel("Learning Rate:"))
        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setDecimals(6)
        self.spin_lr.setRange(1e-6, 1.0)
        self.spin_lr.setSingleStep(1e-4)
        self.spin_lr.setValue(3e-4)
        hp_row.addWidget(self.spin_lr)

        hp_row.addWidget(QLabel("Epochs:"))
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 1000)
        self.spin_epochs.setValue(5)
        hp_row.addWidget(self.spin_epochs)

        hp_row.addWidget(QLabel("Context Length:"))
        self.spin_ctx = QSpinBox()
        self.spin_ctx.setRange(16, 2048)
        self.spin_ctx.setSingleStep(64)
        self.spin_ctx.setValue(256)
        hp_row.addWidget(self.spin_ctx)

        hp_layout.addLayout(hp_row)

        # Advanced settings row
        adv_row = QHBoxLayout()

        adv_row.addWidget(QLabel("Grad Accum Steps:"))
        self.spin_accum = QSpinBox()
        self.spin_accum.setRange(1, 64)
        self.spin_accum.setValue(1)
        self.spin_accum.setToolTip(
            "Gradient accumulation steps. Effective batch = batch_size × accum steps."
        )
        adv_row.addWidget(self.spin_accum)

        adv_row.addWidget(QLabel("Warmup Steps:"))
        self.spin_warmup = QSpinBox()
        self.spin_warmup.setRange(0, 10000)
        self.spin_warmup.setValue(100)
        adv_row.addWidget(self.spin_warmup)

        self.chk_cosine = QCheckBox("Cosine LR Schedule")
        self.chk_cosine.setChecked(True)
        adv_row.addWidget(self.chk_cosine)

        hp_layout.addLayout(adv_row)
        layout.addWidget(hp_group)

        # ----- Buttons ------------------------------------------------------
        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("▶ Start Training")
        self.start_btn.clicked.connect(self._on_start_training)
        btn_row.addWidget(self.start_btn)

        self.stop_btn = QPushButton("⏹ Stop Training")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop_training)
        btn_row.addWidget(self.stop_btn)

        layout.addLayout(btn_row)

        # ----- Progress bar -------------------------------------------------
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # ----- Log output ---------------------------------------------------
        self.train_log = QPlainTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setPlaceholderText("Training log…")
        layout.addWidget(self.train_log)

        return w

    def _on_mode_changed(self, index: int) -> None:
        """Toggle fine-tuning controls visibility."""
        self.ft_group.setVisible(index == 1)

    def _on_browse_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select checkpoint", "", "PyTorch Checkpoints (*.pt);;All Files (*)"
        )
        if path:
            self.ckpt_path_edit.setText(path)

    # ----- Training controls ------------------------------------------------

    def _on_start_training(self) -> None:
        """Collect hyper-params, build trainer, and launch worker thread."""
        if not self.corpus:
            QMessageBox.warning(
                self, "No Data", "Load or scrape some text data first."
            )
            return

        is_fine_tune = self.mode_combo.currentIndex() == 1

        tc = TrainConfig(
            batch_size=self.spin_batch.value(),
            learning_rate=self.spin_lr.value(),
            epochs=self.spin_epochs.value(),
            context_length=self.spin_ctx.value(),
            fine_tune=is_fine_tune,
            freeze_layers=self.spin_freeze.value() if is_fine_tune else 0,
            fine_tune_lr=self.spin_ft_lr.value(),
            warmup_steps=self.spin_warmup.value(),
            use_cosine_schedule=self.chk_cosine.isChecked(),
            gradient_accumulation_steps=self.spin_accum.value(),
        )

        self.trainer = Trainer(train_config=tc)

        # Load checkpoint for fine-tuning
        if is_fine_tune:
            ckpt_path = self.ckpt_path_edit.text().strip()
            if not ckpt_path or not Path(ckpt_path).exists():
                QMessageBox.warning(
                    self, "No Checkpoint",
                    "Please select a valid .pt checkpoint file for fine-tuning."
                )
                return
            try:
                self.trainer.load_checkpoint(ckpt_path)
            except Exception as exc:
                QMessageBox.critical(
                    self, "Checkpoint Error",
                    f"Failed to load checkpoint:\n{exc}"
                )
                return

        self.train_log.clear()
        self.progress_bar.setValue(0)

        self._train_worker = TrainWorker(self.trainer, self.corpus)
        self._train_worker.log.connect(
            lambda msg: self.train_log.appendPlainText(msg)
        )
        self._train_worker.progress.connect(self._on_train_progress)
        self._train_worker.done.connect(self._on_train_done)
        self._train_worker.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _on_stop_training(self) -> None:
        if self._train_worker is not None:
            self._train_worker.request_stop()

    def _on_train_progress(self, current: int, total: int) -> None:
        pct = int(current / total * 100) if total else 0
        self.progress_bar.setValue(pct)

    def _on_train_done(self) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)

        # Make the generator available for the Chat tab.
        if self.trainer is not None:
            enc = tiktoken.get_encoding("gpt2")
            self.generator = Generator(self.trainer.model, enc)
            self.train_log.appendPlainText(
                "🤖 Generator ready — switch to the Chat tab!"
            )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_gui() -> None:
    """Create the QApplication, show the window, and enter the event loop.

    ``QApplication.exec()`` blocks until the window is closed, processing
    all user-input events and signal/slot connections on the main thread.
    """
    app = QApplication(sys.argv)
    window = LexiMainWindow()
    window.show()
    sys.exit(app.exec())
