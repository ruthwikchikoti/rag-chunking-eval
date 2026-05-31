"""Embedding backend abstraction. Default: local sentence-transformers (MiniLM).

A single `Embedder.encode(list[str]) -> np.ndarray` interface, used by both the
semantic chunker and the retrieval index. Vectors are L2-normalised so that a
dot product equals cosine similarity.
"""
from __future__ import annotations

import numpy as np
import requests

from .config import SETTINGS


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class Embedder:
    """Wraps either sentence-transformers ("st") or Ollama embeddings ("ollama")."""

    def __init__(self, backend: str | None = None) -> None:
        self.backend = backend or SETTINGS.embed_backend
        self._st_model = None
        if self.backend == "st":
            self.model_name = SETTINGS.embed_model_st
        elif self.backend == "ollama":
            self.model_name = SETTINGS.embed_model_ollama
        else:
            raise ValueError(f"unknown embed backend: {self.backend}")

    # -- sentence-transformers -------------------------------------------------
    def _ensure_st(self):
        if self._st_model is not None:
            return self._st_model
        from sentence_transformers import SentenceTransformer

        try:
            self._st_model = SentenceTransformer(self.model_name)
        except Exception:
            # Some environments break the HF Hub online metadata HEAD check
            # (httpx "client has been closed"). If the model is already cached,
            # load it offline instead of failing the whole run.
            import os

            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            self._st_model = SentenceTransformer(self.model_name)
        return self._st_model

    def _encode_st(self, texts: list[str]) -> np.ndarray:
        model = self._ensure_st()
        vecs = model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
        return _l2_normalize(np.asarray(vecs, dtype=np.float32))

    # -- ollama embeddings -----------------------------------------------------
    def _encode_ollama(self, texts: list[str]) -> np.ndarray:
        url = f"{SETTINGS.ollama_host}/api/embeddings"
        out = []
        for t in texts:
            r = requests.post(url, json={"model": self.model_name, "prompt": t}, timeout=120)
            r.raise_for_status()
            out.append(r.json()["embedding"])
        return _l2_normalize(np.asarray(out, dtype=np.float32))

    # -- public ----------------------------------------------------------------
    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        if self.backend == "st":
            return self._encode_st(texts)
        return self._encode_ollama(texts)
