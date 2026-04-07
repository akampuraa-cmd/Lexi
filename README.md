# Thanos AI

**Thanos** is a locally trainable and fine-tunable text generation AI built from scratch with Python. It includes a decoder-only Transformer language model, a BPE tokenizer, a training pipeline, and a Tkinter-based GUI for chat, data management, training, and model settings.

---

## Features

- **Decoder-only Transformer** (GPT-style) implemented from scratch using raw PyTorch — no `transformers` library used for the model architecture.
- **BPE Tokenizer** built with the `tokenizers` library, saved as `thanos_tokenizer.json`.
- **Data Cleaning Pipeline** — removes blank lines, trims whitespace, collapses line breaks into a single clean string.
- **Training from scratch** with AdamW + CosineAnnealingLR, checkpoint saving, and loss logging.
- **Fine-tuning** from any saved checkpoint with a lower learning rate.
- **Tkinter GUI** with four tabs: Chat, Data Manager, Training, and Model Settings.
- **Background threading** keeps the GUI responsive during training and generation.

---

## Project Structure

```
Lexi/
├── main.py                  # Entry point — launches the GUI
├── model.py                 # ThanosModel: decoder-only Transformer
├── tokenizer_trainer.py     # BPE tokenizer training and loading
├── data_cleaner.py          # DataCleaner class
├── trainer.py               # Trainer class (train from scratch + fine-tune)
├── generator.py             # Text generation / inference logic
├── gui/
│   ├── app.py               # Main Tkinter app
│   ├── chat_tab.py          # Tab 1: Chat / Text Generation
│   ├── data_tab.py          # Tab 2: Data Manager
│   ├── training_tab.py      # Tab 3: Training
│   └── settings_tab.py      # Tab 4: Model Settings
├── checkpoints/             # Saved model checkpoints (created at runtime)
├── training_data.txt        # Accumulated cleaned training data (created at runtime)
├── thanos_tokenizer.json    # BPE tokenizer (created after first training run)
├── training_log.txt         # Training loss logs (created at runtime)
├── config.json              # Model and training configuration
├── sample_data.txt          # Example training text
└── requirements.txt
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`**:
```
torch>=2.0.0
tokenizers>=0.15.0
```

> Tkinter is included with Python 3.10+. On Linux you may need: `sudo apt-get install python3-tk`

### 2. Run the application

```bash
python main.py
```

---

## Usage

### Step 1 — Add Training Data

1. Open the **Data** tab.
2. Either paste text directly or upload `.txt` files.
3. Click **Preview Cleaned** to see the cleaned version.
4. Click **Add to Training Set** to append to `training_data.txt`.

### Step 2 — Train the Model

1. Open the **Training** tab.
2. Adjust hyperparameters (epochs, batch size, learning rate, etc.).
3. Click **Train from Scratch** to start training.
4. Monitor training loss in the log area.
5. Checkpoints are saved to `./checkpoints/` after each epoch.

### Step 3 — Fine-tune (Optional)

1. Continue adding more data via the Data tab.
2. Click **Fine-tune from Checkpoint** to resume from the latest saved checkpoint.

### Step 4 — Chat

1. Open the **Chat** tab.
2. Type a prompt and click **Generate**.
3. Adjust generation settings (max tokens, temperature, top-k).

### Step 5 — Model Settings

- Edit model hyperparameters and save config.
- Load any `.pt` checkpoint from disk.
- Export model weights to a chosen location.

---

## Configuration

Edit `config.json` or use the **Settings** tab in the GUI:

```json
{
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
  "learning_rate": 0.0003,
  "weight_decay": 0.01,
  "val_split": 0.1
}
```

---

## Checkpoints

- Saved as `checkpoints/thanos_epoch_N.pt` after each epoch.
- Latest checkpoint is always mirrored to `checkpoints/thanos_latest.pt`.
- On startup, the app automatically loads `thanos_latest.pt` if it exists.

---

## Notes

- Training runs on GPU if available, otherwise CPU.
- The tokenizer is retrained whenever you start a fresh training run if `thanos_tokenizer.json` does not exist.
- All text data is cleaned through the `DataCleaner` pipeline before being saved to `training_data.txt`.
