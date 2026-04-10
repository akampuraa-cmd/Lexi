# Lexi AI

A from-scratch GPT-style Transformer language model with a native desktop GUI, web scraping, multi-format data ingestion, and a full **pre-training → fine-tuning** pipeline — built entirely in Python with PyTorch and PyQt6.

---

## Features

| Feature | Description |
|---|---|
| **Decoder-only Transformer** | Multi-head causal self-attention, positional embeddings, feed-forward networks, layer normalisation (pre-norm). |
| **PyQt6 Desktop GUI** | Four-tab interface: Chat, Data Feeding, Web Scraping, Training/Settings. |
| **Multi-Format Data Pipeline** | Load `.txt`, `.json`, `.jsonl`, and `.csv` files. Supports conversational prompt/completion pairs for SFT-style fine-tuning. |
| **Data Cleaning** | Aggressive text preprocessing — removes blank lines, line-breaks, and excess whitespace. |
| **Web Scraping** | Fetch & clean text from any URL, preview it, then optionally add it to the corpus. |
| **Pre-Training** | Train a model from scratch on your own text data with configurable hyperparameters. |
| **Fine-Tuning** | Load a pre-trained checkpoint and fine-tune on new data with layer freezing, lower learning rate, and fresh optimizer state. |
| **LR Scheduling** | Cosine annealing with linear warmup for stable, efficient training. |
| **Gradient Accumulation** | Simulate larger batch sizes on limited hardware. |
| **Validation Split** | Automatic train/val split to monitor generalisation during training. |
| **Smart Hardware Detection** | Automatically detects CUDA (NVIDIA), ROCm (AMD), Intel XPU, Apple MPS, and CPU — uses the best available device. |
| **Checkpoint System** | Save and load model checkpoints during / after training. |

## Project Structure

```
Lexi/
├── main.py              # Application entry point
├── model.py             # LexiModel — GPT-style decoder-only Transformer
├── train.py             # Trainer class with pre-training and fine-tuning loops
├── generator.py         # Auto-regressive text generation (top-p sampling)
├── gui.py               # PyQt6 GUI with Chat, Data, Scraping, Training tabs
├── device_manager.py    # Hardware detection (CUDA, ROCm, XPU, MPS, CPU)
├── data_pipeline.py     # Multi-format data loader (.txt, .json, .jsonl, .csv)
├── data_cleaner.py      # Text preprocessing / cleaning utilities
├── scraper.py           # Web crawling & scraping module
├── requirements.txt     # Python dependencies
└── README.md
```

---

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

---

## Hardware Detection

Lexi automatically detects and uses the **best available compute device** on your machine. The detection order (highest priority first) is:

| Priority | Backend | Hardware | How to Enable |
|----------|---------|----------|---------------|
| 1 | **CUDA** | NVIDIA GPUs (GeForce, RTX, Tesla, A100, etc.) | Install [NVIDIA drivers](https://www.nvidia.com/drivers) + PyTorch with CUDA support |
| 2 | **ROCm** | AMD GPUs (RX 7000, Instinct MI, etc.) | Install [ROCm](https://rocm.docs.amd.com/) + PyTorch ROCm build |
| 3 | **XPU** | Intel Arc / Data Center GPUs | Install [Intel Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch) |
| 4 | **MPS** | Apple Silicon (M1/M2/M3/M4) | macOS 12.3+ with PyTorch ≥ 1.12 (built-in) |
| 5 | **CPU** | Any processor | Always available (fallback) |

When multiple CUDA GPUs are present, Lexi selects the one with the **most VRAM**.

The **Training / Settings** tab in the GUI shows a detailed hardware summary including all detected devices, their VRAM, and which one is selected.

### Installing PyTorch for Your Hardware

```bash
# NVIDIA CUDA (most common for GPU training)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# AMD ROCm
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0

# Intel XPU
pip install torch intel-extension-for-pytorch

# Apple MPS / CPU (default pip install works)
pip install torch
```

---

## How to Feed Data

Lexi supports multiple data formats for maximum flexibility. Use the **📂 Data Feeding** tab in the GUI.

### Supported Formats

#### 1. Plain Text (`.txt`)

Simply load any `.txt` file. The entire contents become part of the training corpus.

```
This is some training text. Lexi will learn patterns from this data.
The more data you provide, the better the model will perform.
```

#### 2. JSON (`.json`)

A JSON file containing an **array of objects** with prompt/completion pairs:

```json
[
    {"prompt": "What is Python?", "completion": "Python is a high-level programming language."},
    {"prompt": "Explain AI.", "completion": "AI is the simulation of human intelligence by machines."}
]
```

#### 3. JSON Lines (`.jsonl`)

One JSON object per line — the preferred format for large datasets:

```jsonl
{"instruction": "Translate hello to French", "response": "Bonjour"}
{"instruction": "What is 2+2?", "response": "4"}
{"instruction": "Who wrote Romeo and Juliet?", "response": "William Shakespeare"}
```

#### 4. CSV (`.csv`)

A CSV file with a header row containing recognised column names:

```csv
question,answer
What is gravity?,Gravity is a force that attracts objects toward each other.
What is photosynthesis?,The process by which plants convert sunlight into energy.
```

### Recognised Column Names

For conversational / fine-tuning data, the pipeline looks for these column pairs (case-insensitive):

| Prompt Column | Completion Column |
|---|---|
| `prompt` | `completion` |
| `input` | `output` |
| `question` | `answer` |
| `instruction` | `response` |

If paired columns are found the data is formatted as conversational turns:

```
<|user|> What is Python? <|assistant|> Python is a high-level programming language. <|sep|>
```

If no paired columns are found, all text is concatenated for plain pre-training.

### Other Ways to Add Data

- **Web Scraping** — Use the 🌐 tab to scrape text from any URL.
- **Paste Text** — Click **📋 Paste Text** to paste text directly.
- **Multiple files** — Select and load many files at once.

---

## How to Pre-Train (Train from Scratch)

Pre-training teaches Lexi the structure of language from raw text data.

### Step-by-Step

1. **Load data** — Go to the **📂 Data Feeding** tab and load your text files. You can combine `.txt`, `.json`, `.jsonl`, and `.csv` files. More data = better results (aim for at least 1 MB of text).

2. **Configure training** — Go to the **⚙️ Training / Settings** tab:
   - Set **Training Mode** to **"Pre-training (from scratch)"**
   - Set **Batch Size** (16 is a good default; increase if you have more VRAM)
   - Set **Learning Rate** (3e-4 is standard for pre-training)
   - Set **Epochs** (5–20 for small datasets, 1–3 for large ones)
   - Set **Context Length** (256 tokens default; increase for longer text understanding)
   - Set **Grad Accum Steps** (increase to simulate larger batches on limited hardware)
   - Set **Warmup Steps** (100 is a good default)
   - Check **Cosine LR Schedule** (recommended)

3. **Start training** — Click **▶ Start Training**. Watch the log for loss values — they should decrease over time.

4. **Checkpoints** — Models are automatically saved after every epoch to the `checkpoints/` directory:
   - `lexi_epoch_1.pt`, `lexi_epoch_2.pt`, etc.
   - `lexi_latest.pt` — the final checkpoint

5. **Chat** — Once training finishes, switch to the **💬 Chat** tab and talk to Lexi!

### Recommended Pre-Training Settings

| Setting | Small Dataset (<1 MB) | Medium Dataset (1–100 MB) | Large Dataset (>100 MB) |
|---|---|---|---|
| Batch Size | 8–16 | 16–32 | 32–64 |
| Learning Rate | 3e-4 | 3e-4 | 1e-4 |
| Epochs | 10–20 | 3–5 | 1–2 |
| Context Length | 128–256 | 256–512 | 512–1024 |
| Warmup Steps | 50 | 100 | 200–500 |

---

## How to Fine-Tune

Fine-tuning takes a pre-trained model and specialises it on a specific task or dataset — this is exactly how ChatGPT is trained (pre-train on internet data, then fine-tune on conversations).

### Step-by-Step

1. **Start with a pre-trained checkpoint** — Either pre-train your own model first (see above) or use an existing `.pt` checkpoint.

2. **Prepare fine-tuning data** — Create a `.jsonl` file with prompt/completion pairs:

   ```jsonl
   {"prompt": "What is Lexi?", "completion": "Lexi is an AI assistant built from scratch using PyTorch."}
   {"prompt": "How do I train you?", "completion": "Load data in the Data Feeding tab, then start training in Settings!"}
   ```

   You can also use `.json`, `.csv`, or `.txt` files (see "How to Feed Data" above).

3. **Load fine-tuning data** — Go to the **📂 Data Feeding** tab and load your file(s).

4. **Configure fine-tuning** — Go to the **⚙️ Training / Settings** tab:
   - Set **Training Mode** to **"Fine-tuning (from checkpoint)"**
   - Click **Browse…** to select your base checkpoint (`.pt` file)
   - Set **Freeze First N Layers** (2–4 is typical — freezes early layers that capture general language patterns)
   - Set **Fine-Tune LR** (1e-5 is a good starting point — much lower than pre-training to avoid forgetting)
   - Set **Epochs** (1–3 for fine-tuning; too many epochs can cause overfitting)

5. **Start training** — Click **▶ Start Training**. The trainer will:
   - Load the pre-trained checkpoint
   - Freeze the specified layers (embeddings + first N blocks)
   - Train only the unfrozen layers on your new data
   - Save new checkpoints

6. **Chat** — Switch to the **💬 Chat** tab. The model should now respond in the style of your fine-tuning data!

### Recommended Fine-Tuning Settings

| Setting | Recommended Value |
|---|---|
| Fine-Tune LR | 1e-5 to 5e-5 |
| Epochs | 1–3 |
| Freeze Layers | 2–4 (out of 6 total by default) |
| Batch Size | 8–16 |
| Grad Accum Steps | 2–4 |
| Cosine Schedule | ✅ Yes |

### What Happens During Fine-Tuning

1. The pre-trained model weights are loaded from the checkpoint.
2. The optimizer is reset (fresh Adam state for fine-tuning).
3. Embeddings and the first N transformer blocks are **frozen** (their weights don't update).
4. Only the later layers and the output head are trained on the new data.
5. A much lower learning rate prevents catastrophic forgetting of pre-trained knowledge.

---

## The ChatGPT-Like Training Pipeline

Lexi follows the same high-level pipeline as ChatGPT:

```
┌─────────────────────────────────────────────────────┐
│  Step 1: Pre-Training                               │
│  Train on large amounts of raw text data.           │
│  The model learns grammar, facts, and reasoning.    │
│  Data: .txt files, web-scraped text, books, etc.    │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  Step 2: Supervised Fine-Tuning (SFT)               │
│  Fine-tune on prompt/completion pairs.              │
│  The model learns to follow instructions and        │
│  respond conversationally.                          │
│  Data: .jsonl/.json/.csv with prompt/completion     │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  Step 3: Chat!                                      │
│  Use the Chat tab to interact with your model.      │
│  Iterate: add more data, fine-tune again.           │
└─────────────────────────────────────────────────────┘
```

---

## Requirements

- Python 3.10+
- PyTorch ≥ 2.6
- PyQt6 ≥ 6.5
- tiktoken
- beautifulsoup4
- requests

See `requirements.txt` for exact versions.

---

## Tips & Best Practices

- **More data is better.** Even small models benefit from large, diverse training corpora.
- **Monitor validation loss.** If validation loss starts increasing while training loss keeps decreasing, you're overfitting — stop training or add more data.
- **Start small, then scale.** Try a quick 2-epoch run first to make sure everything works before committing to a long training run.
- **Use gradient accumulation** if you can't fit large batches in VRAM. Setting batch=8 with accum=4 gives an effective batch size of 32.
- **For fine-tuning, less is more.** 1–3 epochs with a low learning rate is usually enough. Too many epochs cause the model to forget its pre-training.
- **Save checkpoints frequently.** They're saved automatically after every epoch, so you can always go back to the best one.
