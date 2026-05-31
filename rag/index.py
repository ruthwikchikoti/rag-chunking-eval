"""A minimal in-memory vector index (exact cosine search over normalised vectors).

For a single-essay corpus (hundreds of chunks) an exact numpy dot-product search is
faster and simpler than pulling in FAISS, while exposing the same build/search API
you would swap a real ANN index behind.
"""
from __future__ import annotations

import numpy as np


class VectorIndex:
    def __init__(self) -> None:
        self._vecs: np.ndarray | None = None

    def build(self, vectors: np.ndarray) -> "VectorIndex":
        self._vecs = np.asarray(vectors, dtype=np.float32)
        return self

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        if self._vecs is None:
            raise RuntimeError("index not built")
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        scores = self._vecs @ q  # cosine, since everything is L2-normalised
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]
