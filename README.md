# RAG Chunking-Strategy Evaluation

A minimal but rigorous RAG pipeline that compares **three chunking strategies** on a
long-form corpus and measures retrieval + answer quality.

- **Corpus:** Paul Graham, *How to Do Great Work* (~14k words), fetched + cleaned + cached.
- **Strategies:** `fixed` (word windows), `recursive` (boundary-aware splitter w/ overlap),
  `semantic` (sentence-embedding breakpoints).
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (local, normalised). Ollama
  `nomic-embed-text` available as a no-torch fallback (`EMBED_BACKEND=ollama`).
- **Generation + judging:** local **Ollama** (`llama3.2:3b`), temperature 0.
- **Index:** exact cosine search over normalised vectors (numpy).

## Metrics
Per strategy, averaged over 5 hand-written queries (factual / multi-hop / negation /
comparison / summarisation):

| Metric | Definition |
| --- | --- |
| **Precision@3** | fraction of top-3 retrieved chunks an LLM judge marks relevant |
| **anchor_precision@3** | deterministic cross-check: top-3 chunk contains a gold anchor phrase |
| **Faithfulness** | fraction of the answer's claims supported by retrieved context (RAGAS-style) |
| **Answer relevance** | how directly the answer addresses the question |
| **chunk_count / avg size** | structural stats (count, avg words, avg chars) |

## Quickstart
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2:3b           # generation + judge model

python run.py --smoke             # fast wiring check (1 strategy x 1 query)
python run.py                     # full run -> writes outputs/
```

## Outputs (`outputs/`)
- `comparison_table.csv` / `.md` — the headline metrics table
- `metrics_bar.png` — grouped bars (Precision@3, faithfulness, answer relevance)
- `chunk_size_hist.png` — chunk-size distribution per strategy
- `failure_case.md` — one concrete failure with retrieved chunks + judge scores + why
- `results.json` — full per-query records (reproducibility)

## Layout
```
rag/        config, corpus, chunking, embeddings, index, llm, judges, pipeline, evaluate
run.py      entry point
queries.py  the 5 hand-written evaluation queries (in rag/)
```

## Hallucination guardrails
The generation prompt forces *answer-only-from-context* and an explicit **"I don't know"**
abstention; the faithfulness judge then scores claim-level support. See `WRITEUP.md` for
the deploy recommendation and mitigation strategy.
