"""Central configuration. Reads from environment (.env), falls back to sane defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Tiny .env loader (avoids a python-dotenv dependency)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())


_load_dotenv()


@dataclass(frozen=True)
class Settings:
    # Corpus
    corpus_url: str = "https://www.paulgraham.com/greatwork.html"
    corpus_title: str = "How to Do Great Work — Paul Graham"
    corpus_path: Path = DATA_DIR / "corpus.txt"

    # LLM (Ollama)
    ollama_host: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

    # Embeddings
    embed_backend: str = os.environ.get("EMBED_BACKEND", "st")  # "st" | "ollama"
    embed_model_st: str = os.environ.get("EMBED_MODEL_ST", "sentence-transformers/all-MiniLM-L6-v2")
    embed_model_ollama: str = os.environ.get("EMBED_MODEL_OLLAMA", "nomic-embed-text")

    # Retrieval
    top_k: int = 3

    # Chunking parameters
    fixed_words: int = 220
    recursive_target_chars: int = 900
    recursive_overlap_chars: int = 150
    semantic_breakpoint_pct: float = 90.0
    semantic_max_words: int = 320

    strategies: tuple[str, ...] = ("fixed", "recursive", "semantic")


SETTINGS = Settings()
