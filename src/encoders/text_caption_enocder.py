"""Text caption encoder class.

Expose a simple function `encode_caption(model, caption, tokenizer=None, output_path=None, ...)`
that returns token-level embeddings (last_hidden_state) and optionally saves them to disk.

The function accepts either a model name string.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from transformers import T5Tokenizer, T5EncoderModel

class TextEncoder():
    """Text encoder class wrapping T5EncoderModel and T5Tokenizer."""

    def __init__(self, model_name: str, device: Optional[torch.device] = None):
        self.device = device or torch.device("cpu")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(self, caption: str, max_length: int = 128) -> torch.Tensor:
        """Encode a caption to token-level embeddings.

        Args:
            caption: input caption string.
            max_length: tokenization length (padding/truncation).

        Returns:
            embeddings: `torch.Tensor` with shape (1, max_length, hidden_dim) corresponding to
                        `last_hidden_state` from the encoder.
        """
        inputs = self.tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )

        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state  # (1, seq_len, hidden_dim)

        return embeddings