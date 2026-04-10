"""
data_pipeline.py — Lexi Data Pipeline

Supports ingesting data from multiple formats and preparing it for
both **pre-training** (plain text) and **fine-tuning** (conversational
prompt / completion pairs, like ChatGPT SFT data).

Supported input formats
-----------------------
* ``.txt``   — plain text (one continuous document).
* ``.json``  — a JSON array of objects, or a single object.
* ``.jsonl`` — newline-delimited JSON (one object per line).
* ``.csv``   — comma-separated values with a header row.

For **fine-tuning**, the JSON / JSONL / CSV data must contain paired
columns.  Recognised column names (case-insensitive):

    prompt  / completion
    input   / output
    question / answer
    instruction / response

The pipeline converts these pairs into a single training string using
a ChatGPT-style conversation template::

    <|user|> How are you? <|assistant|> I'm great, thanks!

This lets the model learn to respond conversationally.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from data_cleaner import DataCleaner

# ---------------------------------------------------------------------------
# Conversation template tokens
# ---------------------------------------------------------------------------

USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
SEP_TOKEN = "<|sep|>"

# Recognised column-name pairs for prompt / completion data.
_PROMPT_KEYS = {"prompt", "input", "question", "instruction"}
_COMPLETION_KEYS = {"completion", "output", "answer", "response"}


# ---------------------------------------------------------------------------
# Loaded data container
# ---------------------------------------------------------------------------

@dataclass
class LoadedData:
    """Result of loading a data file.

    Attributes
    ----------
    text : str
        The combined text, ready to be tokenised and trained on.
    format : str
        The detected format (``"plain"``, ``"conversational"``).
    num_samples : int
        Number of individual samples / documents found.
    source : str
        Original file path or description.
    """

    text: str
    format: str
    num_samples: int
    source: str


# ---------------------------------------------------------------------------
# Format detection and loading
# ---------------------------------------------------------------------------

def _find_pair_keys(keys: Sequence[str]) -> tuple[str, str] | None:
    """Given a list of column names, find the prompt/completion pair."""
    lower_keys = {k.lower(): k for k in keys}
    for pk in _PROMPT_KEYS:
        for ck in _COMPLETION_KEYS:
            if pk in lower_keys and ck in lower_keys:
                return lower_keys[pk], lower_keys[ck]
    return None


def _format_conversation(prompt: str, completion: str) -> str:
    """Format a single prompt/completion pair as a conversation turn."""
    prompt = DataCleaner.clean(prompt)
    completion = DataCleaner.clean(completion)
    return f"{USER_TOKEN} {prompt} {ASSISTANT_TOKEN} {completion} {SEP_TOKEN}"


def load_text_file(path: str | Path) -> LoadedData:
    """Load a plain ``.txt`` file."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    cleaned = DataCleaner.clean(text)
    return LoadedData(
        text=cleaned,
        format="plain",
        num_samples=1,
        source=str(path),
    )


def load_json_file(path: str | Path) -> LoadedData:
    """Load a ``.json`` file — either an array of objects or a single object."""
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    data = json.loads(raw)

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list) or len(data) == 0:
        # Treat as plain text
        return LoadedData(
            text=DataCleaner.clean(raw),
            format="plain",
            num_samples=1,
            source=str(path),
        )

    # Check for conversational pair keys
    if isinstance(data[0], dict):
        pair = _find_pair_keys(list(data[0].keys()))
        if pair:
            prompt_key, completion_key = pair
            parts = []
            for item in data:
                p = str(item.get(prompt_key, ""))
                c = str(item.get(completion_key, ""))
                if p and c:
                    parts.append(_format_conversation(p, c))
            if parts:
                return LoadedData(
                    text=" ".join(parts),
                    format="conversational",
                    num_samples=len(parts),
                    source=str(path),
                )

    # Fallback: concatenate all text values
    texts = []
    for item in data:
        if isinstance(item, dict):
            texts.append(" ".join(str(v) for v in item.values()))
        else:
            texts.append(str(item))
    combined = DataCleaner.clean(" ".join(texts))
    return LoadedData(
        text=combined,
        format="plain",
        num_samples=len(data),
        source=str(path),
    )


def load_jsonl_file(path: str | Path) -> LoadedData:
    """Load a ``.jsonl`` file (one JSON object per line)."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").strip().splitlines()
    records = []
    for line in lines:
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        return LoadedData(text="", format="plain", num_samples=0, source=str(path))

    # Check for conversational pair keys
    if isinstance(records[0], dict):
        pair = _find_pair_keys(list(records[0].keys()))
        if pair:
            prompt_key, completion_key = pair
            parts = []
            for item in records:
                p = str(item.get(prompt_key, ""))
                c = str(item.get(completion_key, ""))
                if p and c:
                    parts.append(_format_conversation(p, c))
            if parts:
                return LoadedData(
                    text=" ".join(parts),
                    format="conversational",
                    num_samples=len(parts),
                    source=str(path),
                )

    # Fallback
    texts = []
    for item in records:
        if isinstance(item, dict):
            texts.append(" ".join(str(v) for v in item.values()))
        else:
            texts.append(str(item))
    combined = DataCleaner.clean(" ".join(texts))
    return LoadedData(
        text=combined,
        format="plain",
        num_samples=len(records),
        source=str(path),
    )


def load_csv_file(path: str | Path) -> LoadedData:
    """Load a ``.csv`` file with a header row."""
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(raw))
    fieldnames = reader.fieldnames or []

    records = list(reader)
    if not records:
        return LoadedData(text="", format="plain", num_samples=0, source=str(path))

    # Check for conversational pair keys
    pair = _find_pair_keys(fieldnames)
    if pair:
        prompt_key, completion_key = pair
        parts = []
        for row in records:
            p = str(row.get(prompt_key, ""))
            c = str(row.get(completion_key, ""))
            if p and c:
                parts.append(_format_conversation(p, c))
        if parts:
            return LoadedData(
                text=" ".join(parts),
                format="conversational",
                num_samples=len(parts),
                source=str(path),
            )

    # Fallback: concatenate all cell values
    texts = []
    for row in records:
        texts.append(" ".join(str(v) for v in row.values()))
    combined = DataCleaner.clean(" ".join(texts))
    return LoadedData(
        text=combined,
        format="plain",
        num_samples=len(records),
        source=str(path),
    )


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------

_LOADERS = {
    ".txt": load_text_file,
    ".json": load_json_file,
    ".jsonl": load_jsonl_file,
    ".csv": load_csv_file,
}

SUPPORTED_EXTENSIONS = tuple(_LOADERS.keys())


def load_file(path: str | Path) -> LoadedData:
    """Auto-detect format by extension and load the file.

    Raises
    ------
    ValueError
        If the extension is not supported.
    """
    p = Path(path)
    ext = p.suffix.lower()
    loader = _LOADERS.get(ext)
    if loader is None:
        raise ValueError(
            f"Unsupported file format '{ext}'. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    return loader(p)
