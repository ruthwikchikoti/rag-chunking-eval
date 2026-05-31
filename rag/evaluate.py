"""Run all strategies over all queries, compute metrics, and write every output artifact."""
from __future__ import annotations

import json
from statistics import mean, pstdev

import numpy as np
import pandas as pd

from .chunking import Chunk
from .config import SETTINGS, OUTPUT_DIR
from .corpus import corpus_stats, load_corpus
from .embeddings import Embedder
from .judges import answer_relevance, chunk_relevance, faithfulness
from .llm import OllamaClient
from .pipeline import build_pipeline, answer_query
from .queries import QUERIES, Query


# ---------------------------------------------------------------------------
# Per-(strategy, query) evaluation
# ---------------------------------------------------------------------------


def _anchor_hit(chunk_text: str, anchors: list[str]) -> bool:
    low = chunk_text.lower()
    return any(a.lower() in low for a in anchors)


def evaluate_query(pipe, llm: OllamaClient, q: Query) -> dict:
    result = answer_query(pipe, q.question)
    retrieved = result["retrieved"]

    rel_flags = [chunk_relevance(llm, q.question, r["text"])["relevant"] for r in retrieved]
    precision_at_3 = sum(rel_flags) / len(rel_flags) if rel_flags else 0.0
    anchor_hits = [_anchor_hit(r["text"], q.anchors) for r in retrieved]
    anchor_precision = sum(anchor_hits) / len(anchor_hits) if anchor_hits else 0.0

    faith = faithfulness(llm, result["context"], result["answer"])
    relv = answer_relevance(llm, q.question, result["answer"])

    return {
        "query_id": q.id,
        "qtype": q.qtype,
        "question": q.question,
        "answer": result["answer"],
        "precision_at_3": precision_at_3,
        "anchor_precision_at_3": anchor_precision,
        "relevance_flags": rel_flags,
        "anchor_hits": anchor_hits,
        "faithfulness": faith["score"],
        "faithfulness_detail": faith,
        "answer_relevance": relv["score"],
        "retrieved": retrieved,
    }


# ---------------------------------------------------------------------------
# Strategy-level aggregation
# ---------------------------------------------------------------------------


def chunk_size_summary(chunks: list[Chunk]) -> dict:
    words = [c.n_words for c in chunks]
    chars = [c.n_chars for c in chunks]
    return {
        "chunk_count": len(chunks),
        "avg_words": mean(words) if words else 0,
        "std_words": pstdev(words) if len(words) > 1 else 0,
        "avg_chars": mean(chars) if chars else 0,
        "word_sizes": words,
    }


def run_all(smoke: bool = False) -> dict:
    text = load_corpus()
    print(f"[corpus] {SETTINGS.corpus_title} -> {corpus_stats(text)}")

    embedder = Embedder()
    llm = OllamaClient()
    if not llm.health():
        raise RuntimeError(f"Ollama not reachable at {SETTINGS.ollama_host}. Is `ollama serve` running?")

    strategies = SETTINGS.strategies[:1] if smoke else SETTINGS.strategies
    queries = QUERIES[:1] if smoke else QUERIES

    report = {"corpus": {"title": SETTINGS.corpus_title, **corpus_stats(text)},
              "config": {"embed_backend": embedder.backend, "embed_model": embedder.model_name,
                         "llm_model": llm.model, "top_k": SETTINGS.top_k},
              "strategies": {}}

    for strat in strategies:
        print(f"\n[strategy] building '{strat}' ...")
        pipe = build_pipeline(text, strat, embedder, llm)
        sizes = chunk_size_summary(pipe.chunks)
        print(f"  chunks={sizes['chunk_count']} avg_words={sizes['avg_words']:.0f}")

        per_query = []
        for q in queries:
            print(f"  - {q.id} ({q.qtype}) ...", end="", flush=True)
            ev = evaluate_query(pipe, llm, q)
            per_query.append(ev)
            print(f" P@3={ev['precision_at_3']:.2f} faith={ev['faithfulness']:.2f} rel={ev['answer_relevance']:.2f}")

        report["strategies"][strat] = {
            "chunk_count": sizes["chunk_count"],
            "avg_words": round(sizes["avg_words"], 1),
            "std_words": round(sizes["std_words"], 1),
            "avg_chars": round(sizes["avg_chars"], 1),
            "word_sizes": sizes["word_sizes"],
            "mean_precision_at_3": round(mean(e["precision_at_3"] for e in per_query), 3),
            "mean_anchor_precision_at_3": round(mean(e["anchor_precision_at_3"] for e in per_query), 3),
            "mean_faithfulness": round(mean(e["faithfulness"] for e in per_query), 3),
            "mean_answer_relevance": round(mean(e["answer_relevance"] for e in per_query), 3),
            "per_query": per_query,
        }

    return report


# ---------------------------------------------------------------------------
# Output artifacts
# ---------------------------------------------------------------------------


def _comparison_frame(report: dict) -> pd.DataFrame:
    rows = []
    for strat, s in report["strategies"].items():
        rows.append({
            "strategy": strat,
            "chunk_count": s["chunk_count"],
            "avg_words": s["avg_words"],
            "avg_chars": s["avg_chars"],
            "precision@3": s["mean_precision_at_3"],
            "anchor_precision@3": s["mean_anchor_precision_at_3"],
            "faithfulness": s["mean_faithfulness"],
            "answer_relevance": s["mean_answer_relevance"],
        })
    return pd.DataFrame(rows)


def write_outputs(report: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1) full results json
    (OUTPUT_DIR / "results.json").write_text(json.dumps(report, indent=2))

    # 2) comparison table (csv + md)
    df = _comparison_frame(report)
    df.to_csv(OUTPUT_DIR / "comparison_table.csv", index=False)
    (OUTPUT_DIR / "comparison_table.md").write_text(df.to_markdown(index=False))
    print("\n" + df.to_string(index=False))

    # 3) grouped bar chart of quality metrics
    metrics = ["precision@3", "faithfulness", "answer_relevance"]
    strategies = df["strategy"].tolist()
    x = np.arange(len(metrics))
    width = 0.8 / max(len(strategies), 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, strat in enumerate(strategies):
        vals = [df.loc[df.strategy == strat, m].values[0] for m in metrics]
        ax.bar(x + i * width, vals, width, label=strat)
    ax.set_xticks(x + width * (len(strategies) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Retrieval & answer quality by chunking strategy")
    ax.legend(title="strategy")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "metrics_bar.png", dpi=140)
    plt.close(fig)

    # 4) chunk-size distribution histogram (one subplot per strategy)
    strat_items = list(report["strategies"].items())
    fig, axes = plt.subplots(1, len(strat_items), figsize=(5 * len(strat_items), 4), sharey=True)
    if len(strat_items) == 1:
        axes = [axes]
    for ax, (strat, s) in zip(axes, strat_items):
        ax.hist(s["word_sizes"], bins=20, color="steelblue", edgecolor="white")
        ax.set_title(f"{strat}\n(n={s['chunk_count']}, avg={s['avg_words']:.0f}w)")
        ax.set_xlabel("chunk size (words)")
    axes[0].set_ylabel("count")
    fig.suptitle("Chunk-size distribution by strategy")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chunk_size_hist.png", dpi=140)
    plt.close(fig)

    # 5) failure case: lowest faithfulness*relevance across all (strategy, query)
    write_failure_case(report)
    print(f"\n[outputs] written to {OUTPUT_DIR}")


def write_failure_case(report: dict) -> None:
    worst = None
    for strat, s in report["strategies"].items():
        for ev in s["per_query"]:
            score = ev["faithfulness"] * ev["answer_relevance"] * (0.5 + 0.5 * ev["precision_at_3"])
            if worst is None or score < worst[0]:
                worst = (score, strat, ev)
    if worst is None:
        return
    _, strat, ev = worst
    lines = [
        f"# Failure Case\n",
        f"**Strategy:** `{strat}`  ",
        f"**Query ({ev['qtype']}):** {ev['question']}\n",
        f"**Scores:** Precision@3 = {ev['precision_at_3']:.2f} | "
        f"Faithfulness = {ev['faithfulness']:.2f} | Answer relevance = {ev['answer_relevance']:.2f}\n",
        f"**Generated answer:**\n\n> {ev['answer']}\n",
        "**Retrieved top-3 chunks (relevance judged):**\n",
    ]
    for n, (r, flag) in enumerate(zip(ev["retrieved"], ev["relevance_flags"]), 1):
        mark = "✅ relevant" if flag else "❌ not relevant"
        snippet = r["text"][:320].replace("\n", " ")
        lines.append(f"**[{n}]** (score={r['score']:.3f}, {mark})\n\n> {snippet}…\n")
    lines.append(
        "**Why it failed / mitigation:** The retriever surfaced passages that are topically "
        "near the query but miss the specific supporting facts, so the generator either abstains "
        "or stitches a partially-supported answer. This is the classic chunk-boundary problem: "
        "the evidence the query needs is split across chunk edges or diluted by unrelated sentences. "
        "Mitigations: smaller/overlapping or semantically-coherent chunks, retrieving more candidates "
        "then re-ranking, and the strict 'answer only from context, else say I don't know' prompt "
        "that keeps an unsupported answer from being emitted as fact."
    )
    (OUTPUT_DIR / "failure_case.md").write_text("\n".join(lines))
