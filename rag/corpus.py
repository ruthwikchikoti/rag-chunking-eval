"""Fetch, clean, and cache the long-form corpus (a Paul Graham essay)."""
from __future__ import annotations

import re
from pathlib import Path

import requests

from .config import SETTINGS


def _clean_html(html: str) -> str:
    """Extract readable body text from a Paul Graham essay page.

    PG essays are old-school HTML: the body text lives in <font> tags inside a table.
    We use BeautifulSoup to strip markup, then normalise whitespace.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()

    # The essay text is the largest text block on the page.
    text = soup.get_text("\n")

    # Normalise: collapse runs of blank lines, strip the nav/footer cruft.
    lines = [ln.strip() for ln in text.splitlines()]
    # Drop very short boilerplate lines near the top/bottom (e.g. "Want to start a startup?").
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Join hard-wrapped lines within a paragraph into flowing sentences.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    paragraphs = [re.sub(r"\s*\n\s*", " ", p) for p in paragraphs]
    cleaned = "\n\n".join(paragraphs)
    return cleaned.strip()


def load_corpus(force_refresh: bool = False) -> str:
    """Return the cached corpus, fetching + cleaning it on first run."""
    path: Path = SETTINGS.corpus_path
    if path.exists() and not force_refresh:
        return path.read_text(encoding="utf-8")

    path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(SETTINGS.corpus_url, timeout=30, headers={"User-Agent": "rag-eval/1.0"})
    resp.raise_for_status()
    text = _clean_html(resp.text)

    # Trim the trailing "Notes" / "Thanks" sections so the corpus is the essay body.
    for marker in ("\nNotes\n", "\nThanks to", "\n[1]"):
        idx = text.find(marker)
        if idx > 2000:  # only trim if we keep a substantial body
            text = text[:idx].rstrip()
            break

    path.write_text(text, encoding="utf-8")
    return text


def corpus_stats(text: str) -> dict:
    words = text.split()
    return {"chars": len(text), "words": len(words), "paragraphs": text.count("\n\n") + 1}


if __name__ == "__main__":
    body = load_corpus()
    print(corpus_stats(body))
    print(body[:500])
