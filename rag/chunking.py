"""Three chunking strategies, each returning a list of Chunk objects.

1. fixed     - fixed-size word windows, no overlap (baseline).
2. recursive - RecursiveCharacterTextSplitter-style: split on natural boundaries
               (paragraph -> line -> sentence -> word) toward a target size, with overlap.
3. semantic  - sentence embeddings + distance-breakpoint segmentation: start a new
               chunk where consecutive sentences are semantically dissimilar.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .config import SETTINGS

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    idx: int
    text: str
    strategy: str
    n_words: int = field(init=False)
    n_chars: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_words = len(self.text.split())
        self.n_chars = len(self.text)


# ---------------------------------------------------------------------------
# Sentence splitting (regex, abbreviation-aware — avoids an nltk dependency)
# ---------------------------------------------------------------------------

_ABBREV = {"mr", "mrs", "ms", "dr", "vs", "etc", "e.g", "i.e", "st", "no", "fig"}
_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def split_sentences(text: str) -> list[str]:
    raw = _SENT_END.split(text.replace("\n", " "))
    sentences: list[str] = []
    for s in raw:
        s = s.strip()
        if not s:
            continue
        # Merge fragments left dangling by abbreviations (e.g. "e.g.").
        last_word = re.sub(r"[^a-zA-Z.]", "", s.split()[-1].lower()) if s.split() else ""
        if sentences and last_word.rstrip(".") in _ABBREV:
            sentences[-1] = sentences[-1] + " " + s
        else:
            sentences.append(s)
    return sentences


# ---------------------------------------------------------------------------
# Strategy 1: fixed-size word windows
# ---------------------------------------------------------------------------


def fixed_size(text: str, words_per_chunk: int | None = None) -> list[Chunk]:
    n = words_per_chunk or SETTINGS.fixed_words
    words = text.split()
    chunks: list[Chunk] = []
    for i in range(0, len(words), n):
        piece = " ".join(words[i : i + n])
        if piece.strip():
            chunks.append(Chunk(idx=len(chunks), text=piece, strategy="fixed"))
    return chunks


# ---------------------------------------------------------------------------
# Strategy 2: recursive character splitter with overlap
# ---------------------------------------------------------------------------


def _split_on_separators(text: str, separators: list[str]) -> list[str]:
    """Recursively break text into pieces no larger than the target size."""
    target = SETTINGS.recursive_target_chars
    if len(text) <= target or not separators:
        return [text]
    sep, *rest = separators
    parts = text.split(sep) if sep else list(text)
    out: list[str] = []
    for part in parts:
        if len(part) <= target:
            out.append(part)
        else:
            out.extend(_split_on_separators(part, rest))
    return [p for p in out if p.strip()]


def recursive(text: str) -> list[Chunk]:
    target = SETTINGS.recursive_target_chars
    overlap = SETTINGS.recursive_overlap_chars
    pieces = _split_on_separators(text, ["\n\n", "\n", ". ", " "])

    # Greedily merge small pieces up to the target, then add overlap between chunks.
    merged: list[str] = []
    buf = ""
    for p in pieces:
        candidate = (buf + " " + p).strip() if buf else p
        if len(candidate) <= target:
            buf = candidate
        else:
            if buf:
                merged.append(buf)
            buf = p
    if buf:
        merged.append(buf)

    chunks: list[Chunk] = []
    for i, body in enumerate(merged):
        if overlap and i > 0:
            tail = merged[i - 1][-overlap:]
            body = (tail + " " + body).strip()
        chunks.append(Chunk(idx=i, text=body, strategy="recursive"))
    return chunks


# ---------------------------------------------------------------------------
# Strategy 3: semantic (embedding-breakpoint) chunking
# ---------------------------------------------------------------------------


def semantic(text: str, encode: Callable[[list[str]], np.ndarray]) -> list[Chunk]:
    """Segment at points where consecutive sentences are semantically dissimilar.

    `encode` returns L2-normalised vectors, so cosine distance = 1 - dot product.
    A breakpoint is placed where distance exceeds the configured percentile.
    """
    sentences = split_sentences(text)
    if len(sentences) < 3:
        return [Chunk(idx=0, text=text, strategy="semantic")]

    vecs = encode(sentences)
    distances = 1.0 - np.sum(vecs[:-1] * vecs[1:], axis=1)
    threshold = float(np.percentile(distances, SETTINGS.semantic_breakpoint_pct))

    chunks: list[Chunk] = []
    current: list[str] = [sentences[0]]
    for i, sent in enumerate(sentences[1:]):
        over_size = len(" ".join(current).split()) >= SETTINGS.semantic_max_words
        is_break = distances[i] > threshold
        if is_break or over_size:
            chunks.append(Chunk(idx=len(chunks), text=" ".join(current), strategy="semantic"))
            current = [sent]
        else:
            current.append(sent)
    if current:
        chunks.append(Chunk(idx=len(chunks), text=" ".join(current), strategy="semantic"))
    return chunks


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def chunk(text: str, strategy: str, encode: Callable[[list[str]], np.ndarray] | None = None) -> list[Chunk]:
    if strategy == "fixed":
        return fixed_size(text)
    if strategy == "recursive":
        return recursive(text)
    if strategy == "semantic":
        if encode is None:
            raise ValueError("semantic chunking requires an `encode` function")
        return semantic(text, encode)
    raise ValueError(f"unknown strategy: {strategy}")
