from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TransformersPerplexityScorer:
    """CPU-only sliding-window PPL scorer for a Hugging Face causal language model."""

    model_id: str = "distilgpt2"
    stride: int = 512
    cpu_threads: int = 1
    _tokenizer: Any = field(init=False, default=None, repr=False)
    _model: Any = field(init=False, default=None, repr=False)
    _torch: Any = field(init=False, default=None, repr=False)

    def perplexity(self, text: str) -> float:
        if not text.strip():
            raise ValueError("Perplexity requires non-whitespace text.")
        self._ensure_loaded()
        encoded = self._tokenizer(text, return_tensors="pt")
        input_ids = encoded.input_ids
        sequence_length = input_ids.size(1)
        if sequence_length < 2:
            raise ValueError("Perplexity requires at least two model tokens.")

        max_length = self._context_length()
        total_negative_log_likelihood = 0.0
        total_scored_tokens = 0
        previous_end = 0

        with self._torch.no_grad():
            for begin in range(0, sequence_length, self.stride):
                end = min(begin + self.stride, sequence_length)
                target_length = end - previous_end
                window_begin = max(0, end - max_length)
                window_ids = input_ids[:, window_begin:end]
                labels = window_ids.clone()
                labels[:, :-target_length] = -100
                output = self._model(input_ids=window_ids, labels=labels)
                scored_here = max(target_length - (1 if window_begin == 0 else 0), 0)
                total_negative_log_likelihood += output.loss.item() * scored_here
                total_scored_tokens += scored_here
                previous_end = end

        if total_scored_tokens == 0:
            raise ValueError("Perplexity requires at least two model tokens.")
        return math.exp(total_negative_log_likelihood / total_scored_tokens)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(self.cpu_threads)
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self._model.to("cpu")
        self._model.eval()

    def _context_length(self) -> int:
        value = getattr(self._model.config, "n_positions", None) or getattr(self._model.config, "max_position_embeddings", None)
        if not isinstance(value, int) or value < 2:
            raise ValueError("Could not determine the model context window.")
        return value
