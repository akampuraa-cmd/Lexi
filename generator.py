"""
generator.py — Text generation / inference logic for Thanos.
"""

import torch
import torch.nn.functional as F


class Generator:
    """Handles text generation from a trained ThanosModel."""

    def __init__(self, model, tokenizer, device: str = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text to condition on.
            max_new_tokens: Number of new tokens to generate.
            temperature: Sampling temperature (higher = more random).
            top_k: If > 0, only sample from the top-k most probable tokens.

        Returns:
            Generated text string (including the prompt).
        """
        encoding = self.tokenizer.encode(prompt)
        input_ids = encoding.ids

        if not input_ids:
            return prompt

        max_seq_len = self.model.config.get("max_seq_len", 512)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        generated_ids = list(input_ids)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Truncate to max_seq_len
                context = torch.tensor(
                    [generated_ids[-max_seq_len:]],
                    dtype=torch.long,
                    device=self.device,
                )

                logits = self.model(context)
                # Take logits for the last token
                next_logits = logits[0, -1, :]

                if temperature <= 0:
                    # Greedy decoding
                    next_token = torch.argmax(next_logits).item()
                else:
                    next_logits = next_logits / temperature

                    if top_k > 0:
                        top_k_val = min(top_k, next_logits.size(-1))
                        values, _ = torch.topk(next_logits, top_k_val)
                        threshold = values[-1]
                        next_logits[next_logits < threshold] = float("-inf")

                    probs = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()

                generated_ids.append(next_token)

                # Stop at EOS token
                eos_id = self.tokenizer.token_to_id("[EOS]")
                if eos_id is not None and next_token == eos_id:
                    break

        # Decode the generated tokens (skip the BOS token if present)
        bos_id = self.tokenizer.token_to_id("[BOS]")
        output_ids = [t for t in generated_ids if t != bos_id]
        return self.tokenizer.decode(output_ids)
