"""
generator.py — Lexi Text Generation (Inference)

Provides auto-regressive text generation using nucleus (top-p) and
temperature sampling.

Usage::

    gen = Generator(model, enc)
    output = gen.generate("Once upon a time", max_new_tokens=200)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import tiktoken

from model import DEVICE, LexiModel


class Generator:
    """Wraps a trained ``LexiModel`` for interactive text generation.

    Parameters
    ----------
    model : LexiModel
        A trained (or partially trained) Lexi model.
    enc : tiktoken.Encoding
        The BPE tokeniser used during training (must match vocab).
    """

    def __init__(self, model: LexiModel, enc: tiktoken.Encoding) -> None:
        self.model = model
        self.enc = enc
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> str:
        """Generate text continuing from *prompt*.

        Parameters
        ----------
        prompt : str
            The seed text.
        max_new_tokens : int
            How many new tokens to generate.
        temperature : float
            Softmax temperature (>1 = more random, <1 = more greedy).
        top_p : float
            Nucleus-sampling threshold.  Tokens whose cumulative
            probability exceeds *top_p* are excluded.

        Returns
        -------
        str
            The full generated text (prompt + continuation).
        """
        token_ids = self.enc.encode(prompt)
        idx = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)
        ctx = self.model.config.context_length

        for _ in range(max_new_tokens):
            # Crop to the last ``context_length`` tokens so positional
            # embeddings stay in range.
            idx_cond = idx[:, -ctx:]
            logits, _ = self.model(idx_cond)

            # Take logits for the very last position.
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            # --- Nucleus (top-p) sampling ---
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                F.softmax(sorted_logits, dim=-1), dim=-1
            )
            # Remove tokens with cumulative probability above top_p.
            remove_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[remove_mask] = float("-inf")

            probs = F.softmax(sorted_logits, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1)
            next_token = sorted_idx.gather(-1, sampled)

            idx = torch.cat([idx, next_token], dim=1)

        return self.enc.decode(idx[0].tolist())
