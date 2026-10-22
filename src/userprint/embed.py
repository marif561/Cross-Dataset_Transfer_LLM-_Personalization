"""Sentence embeddings. Sentence-BERT by default; a hashing embedder keeps tests offline."""
from __future__ import annotations

from typing import List

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """``model="hash"`` is a dependency-free bag-of-words embedder for offline smoke tests only; it
    cannot group paraphrases, so use Sentence-BERT (the default) for real persona extraction."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model_name = model
        if model == "hash":
            self._hv = HashingVectorizer(n_features=512, alternate_sign=False, norm="l2", ngram_range=(1, 2))
            self._st = None
        else:
            from sentence_transformers import SentenceTransformer
            self._st = SentenceTransformer(model)

    def encode(self, texts: List[str]) -> np.ndarray:
        if self._st is None:
            return self._hv.transform(texts).toarray().astype("float32")
        return np.asarray(self._st.encode(list(texts), normalize_embeddings=True, show_progress_bar=False),
                          dtype="float32")
