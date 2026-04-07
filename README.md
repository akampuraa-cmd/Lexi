# Lexi AI

A from-scratch GPT-style Transformer language model with a native Windows GUI, web scraping, and a robust data pipeline — built entirely in Python with PyTorch and PyQt6.

## Features

| Feature | Description |
|---|---|
| **Decoder-only Transformer** | Multi-head causal self-attention, positional embeddings, feed-forward networks, layer normalisation (pre-norm). |
| **PyQt6 Desktop GUI** | Four-tab interface: Chat, Data Feeding, Web Scraping, Training/Settings. |
| **Data Cleaning** | Aggressive text preprocessing — removes blank lines, line-breaks, and excess whitespace. |
| **Web Scraping** | Fetch & clean text from any URL, preview it, then optionally add it to the corpus. |
| **Configurable Training** | Batch size, learning rate, epochs, context length — all adjustable from the GUI. |
| **Auto Hardware Detection** | Automatically uses CUDA if available, otherwise falls back to CPU. |
| **Checkpoint System** | Save and load model checkpoints during / after training. |

## Project Structure

```
Lexi/
├── main.py            # Application entry point
├── model.py           # LexiModel — GPT-style decoder-only Transformer
├── train.py           # Trainer class with full training loop
├── generator.py       # Auto-regressive text generation (top-p sampling)
├── gui.py             # PyQt6 GUI with Chat, Data, Scraping, Training tabs
├── data_cleaner.py    # Text preprocessing / cleaning utilities
├── scraper.py         # Web crawling & scraping module
├── requirements.txt   # Python dependencies
└── README.md
```

## Quick Start

```bash
# 1. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the GUI
python main.py
```

## Usage

1. **Load Data** — Switch to the *Data Feeding* tab and load `.txt` files, or use the *Web Scraping* tab to fetch text from a URL.
2. **Configure Training** — In the *Training / Settings* tab, set your hyperparameters and click **Start Training**.
3. **Chat** — Once training completes, switch to the *Chat* tab and talk to Lexi!

## Requirements

- Python 3.10+
- PyTorch ≥ 2.0
- PyQt6 ≥ 6.5
- tiktoken
- beautifulsoup4
- requests

See `requirements.txt` for exact versions.
