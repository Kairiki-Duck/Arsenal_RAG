import os
from collections.abc import Iterable

from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL


MODEL_PATH = EMBEDDING_MODEL

class EmbeddingModel:
    def __init__(
        self,
        model_path=MODEL_PATH,
        device=None,
        batch_size=32,
        model=None,
        trust_remote_code=True,
    ):
        if not isinstance(model_path, str) or not model_path.strip():
            raise ValueError("model_path must be a non-empty string")
        if device is not None and (not isinstance(device, str) or not device.strip()):
            raise ValueError("device must be a non-empty string or None")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.model_path = model_path
        self.batch_size = batch_size
        self.model = (
            SentenceTransformer(
                model_path,
                trust_remote_code=trust_remote_code,
                **({"device": device} if device is not None else {}),
            )
            if model is None
            else model
        )

    def encode(self, texts, batch_size=None):
        """Encode non-empty text inputs without terminal progress output."""
        if isinstance(texts, str) or not isinstance(texts, Iterable):
            raise TypeError("texts must be an iterable of strings")
        texts = list(texts)
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("texts must contain non-empty strings")
        resolved_batch_size = self.batch_size if batch_size is None else batch_size
        if (
            not isinstance(resolved_batch_size, int)
            or isinstance(resolved_batch_size, bool)
            or resolved_batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")

        embeddings = self.model.encode(
            texts,
            batch_size=resolved_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings


__all__ = ["EmbeddingModel", "MODEL_PATH"]