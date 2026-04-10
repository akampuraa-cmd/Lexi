"""
model.py — Lexi GPT-Style Transformer (Decoder-Only)

This file implements the full Transformer decoder stack from scratch
using only PyTorch primitives.  No ``nn.Transformer`` convenience class
is used — every sub-layer is written explicitly so that the maths is
clear and auditable.

Architecture overview
---------------------
The model follows the standard GPT / decoder-only layout:

    Token Embeddings  +  Positional Embeddings
           │
           ▼
    ┌──────────────────┐
    │  N × DecoderBlock │   (each block = masked self-attention + FFN)
    └──────────────────┘
           │
           ▼
       LayerNorm
           │
           ▼
     Linear → logits   (tied with token embeddings, optional)

Key mathematical ideas
~~~~~~~~~~~~~~~~~~~~~~
* **Scaled Dot-Product Attention**:
      Attention(Q, K, V) = softmax(Q Kᵀ / √dₖ) V
  The scaling by √dₖ prevents the dot products from growing too large
  in magnitude, which would push softmax into regions with tiny
  gradients.

* **Multi-Head Attention**:
  Instead of one large attention function, we project Q, K, V into
  *h* smaller sub-spaces (heads), run attention in parallel, and
  concatenate the results:
      MultiHead(Q, K, V) = Concat(head₁, …, headₕ) Wᴼ
  This lets the model attend to information from different
  representation sub-spaces at different positions.

* **Causal (auto-regressive) mask**:
  A lower-triangular boolean mask ensures that position *i* can only
  attend to positions ≤ *i*.  This is what makes the model
  *decoder-only* / auto-regressive.

* **Position-wise Feed-Forward Network (FFN)**:
      FFN(x) = GELU(x W₁ + b₁) W₂ + b₂
  Applied identically to each position.  The inner dimension is
  typically 4× the model dimension.

* **Layer Normalisation** (pre-norm variant):
  Applied *before* each sub-layer (attention / FFN) rather than after.
  Pre-norm is empirically more stable during training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from device_manager import DEVICE  # noqa: F401  — re-exported for back-compat


# ---------------------------------------------------------------------------
# Hyperparameter container
# ---------------------------------------------------------------------------
@dataclass
class LexiConfig:
    """All tuneable knobs for the Lexi model.

    Attributes
    ----------
    vocab_size : int
        Number of tokens in the vocabulary (set by the tokeniser).
    context_length : int
        Maximum sequence length the model can process (T).
    n_embd : int
        Dimensionality of the token / positional embeddings (d_model).
    n_head : int
        Number of attention heads.  ``n_embd`` must be divisible by
        ``n_head`` so that each head has dimension ``n_embd // n_head``.
    n_layer : int
        Number of stacked decoder blocks.
    dropout : float
        Dropout probability used throughout the model.
    """

    vocab_size: int = 50257          # GPT-2 BPE vocabulary size
    context_length: int = 256        # max sequence length
    n_embd: int = 384                # embedding / model dimension
    n_head: int = 6                  # number of attention heads
    n_layer: int = 6                 # number of Transformer blocks
    dropout: float = 0.1


# ---------------------------------------------------------------------------
# Multi-Head Causal Self-Attention
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """Masked multi-head self-attention.

    The causal mask is registered as a buffer so it moves with the model
    to the correct device automatically.
    """

    def __init__(self, config: LexiConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0, (
            f"n_embd ({config.n_embd}) must be divisible by "
            f"n_head ({config.n_head})"
        )

        # Combined projection for Q, K, V (more efficient than three
        # separate linear layers).
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection after concatenating heads.
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.d_k = config.n_embd // config.n_head  # per-head dimension

        # Causal mask: lower-triangular matrix of shape (1, 1, T, T).
        # ``True`` means *masked* (i.e. attention weight = −∞).
        mask = torch.triu(
            torch.ones(config.context_length, config.context_length, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("mask", mask.unsqueeze(0).unsqueeze(0))

    # ----- forward ----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, T, C)
            B = batch size, T = sequence length, C = n_embd.

        Returns
        -------
        Tensor, shape (B, T, C)
        """
        B, T, C = x.size()

        # 1. Project to Q, K, V and split into heads.
        #    qkv shape: (B, T, 3*C)  →  three tensors of (B, T, C)
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)

        # Reshape: (B, T, C) → (B, n_head, T, d_k)
        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)

        # 2. Scaled dot-product attention:
        #    scores = Q Kᵀ / √d_k   →  (B, n_head, T, T)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply causal mask (positions that must not be attended to
        # receive −∞ so that softmax maps them to 0).
        scores = scores.masked_fill(self.mask[:, :, :T, :T], float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        # 3. Weighted sum of values → (B, n_head, T, d_k)
        out = attn_weights @ v

        # 4. Concatenate heads → (B, T, C)  and project.
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_drop(self.out_proj(out))
        return out


# ---------------------------------------------------------------------------
# Position-wise Feed-Forward Network
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    """Two-layer MLP with GELU activation (the "FFN" sub-layer).

    Inner dimension is 4× the model dimension, following the original
    Transformer and GPT designs.
    """

    def __init__(self, config: LexiConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Single Decoder Block
# ---------------------------------------------------------------------------
class DecoderBlock(nn.Module):
    """Pre-norm Transformer decoder block.

    Structure::

        x  →  LayerNorm → CausalSelfAttention → + (residual)
           →  LayerNorm → FeedForward          → + (residual)

    Pre-norm means we normalise *before* each sub-layer, which tends
    to be more numerically stable than post-norm.
    """

    def __init__(self, config: LexiConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection around the attention sub-layer.
        x = x + self.attn(self.ln1(x))
        # Residual connection around the FFN sub-layer.
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Full Lexi Model
# ---------------------------------------------------------------------------
class LexiModel(nn.Module):
    """GPT-style decoder-only Transformer language model.

    Forward pass
    ~~~~~~~~~~~~
    1. Look up token embeddings and add learned positional embeddings.
    2. Pass through ``n_layer`` DecoderBlocks.
    3. Apply a final LayerNorm.
    4. Project back to vocabulary logits with a linear head.

    The token embedding weight matrix is **tied** to the output
    projection (weight tying), which reduces the total parameter count
    and acts as a regulariser.
    """

    def __init__(self, config: LexiConfig) -> None:
        super().__init__()
        self.config = config

        # --- Embeddings ---
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.context_length, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # --- Transformer blocks ---
        self.blocks = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.n_layer)]
        )

        # --- Output head ---
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share the token-embedding matrix with the
        # output projection so that the model learns a single
        # representation space.
        self.head.weight = self.tok_emb.weight

        # Initialise weights.
        self.apply(self._init_weights)

    # ----- weight initialisation -------------------------------------------
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Apply a truncated-normal initialisation (std = 0.02).

        Biases are initialised to zero.  This scheme is standard for
        GPT-family models and helps with training stability.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ----- forward ----------------------------------------------------------
    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Parameters
        ----------
        idx : LongTensor, shape (B, T)
            Token indices for the input sequence.
        targets : LongTensor, shape (B, T), optional
            Token indices for the target sequence (shifted by 1).
            If provided the cross-entropy loss is computed.

        Returns
        -------
        logits : Tensor, shape (B, T, vocab_size)
        loss : Tensor or None
        """
        B, T = idx.size()
        assert T <= self.config.context_length, (
            f"Sequence length {T} exceeds context_length "
            f"{self.config.context_length}"
        )

        # Positional indices: 0, 1, 2, …, T-1
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        # Token + positional embeddings, shape (B, T, C).
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        # Pass through all decoder blocks.
        for block in self.blocks:
            x = block(x)

        # Final layer norm + linear projection to vocabulary logits.
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten for cross-entropy: (B*T, vocab_size) vs (B*T,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    # ----- parameter count helper ------------------------------------------
    def num_params(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
