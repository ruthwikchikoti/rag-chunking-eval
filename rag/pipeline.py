"""Wires corpus -> chunks -> embeddings -> index, and answers a query with top-k context."""
from __future__ import annotations

from dataclasses import dataclass

from .chunking import Chunk, chunk
from .config import SETTINGS
from .embeddings import Embedder
from .index import VectorIndex
from .llm import OllamaClient

_ANSWER_SYSTEM = (
    "You are a careful assistant. Answer the question using ONLY the numbered context passages. "
    "If the answer is not contained in the context, reply exactly: I don't know. "
    "Do not use outside knowledge. Keep the answer to 1-3 sentences."
)


@dataclass
class Pipeline:
    strategy: str
    chunks: list[Chunk]
    index: VectorIndex
    embedder: Embedder
    llm: OllamaClient


def build_pipeline(text: str, strategy: str, embedder: Embedder, llm: OllamaClient) -> Pipeline:
    chunks = chunk(text, strategy, encode=embedder.encode)
    vecs = embedder.encode([c.text for c in chunks])
    index = VectorIndex().build(vecs)
    return Pipeline(strategy=strategy, chunks=chunks, index=index, embedder=embedder, llm=llm)


def retrieve(pipe: Pipeline, question: str, k: int | None = None) -> list[tuple[Chunk, float]]:
    k = k or SETTINGS.top_k
    qvec = pipe.embedder.encode([question])[0]
    hits = pipe.index.search(qvec, k)
    return [(pipe.chunks[i], score) for i, score in hits]


def _format_context(retrieved: list[tuple[Chunk, float]]) -> str:
    return "\n\n".join(f"[{n}] {c.text}" for n, (c, _) in enumerate(retrieved, 1))


def answer_query(pipe: Pipeline, question: str, k: int | None = None) -> dict:
    retrieved = retrieve(pipe, question, k)
    context = _format_context(retrieved)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    answer = pipe.llm.generate(prompt, system=_ANSWER_SYSTEM, temperature=0.0)
    return {
        "question": question,
        "answer": answer,
        "context": context,
        "retrieved": [
            {"chunk_idx": c.idx, "score": score, "n_words": c.n_words, "text": c.text}
            for c, score in retrieved
        ],
    }
