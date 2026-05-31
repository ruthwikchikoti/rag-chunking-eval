"""Entry point for the RAG chunking-strategy evaluation.

Usage:
    python run.py            # full run: 3 strategies x 5 queries, writes outputs/
    python run.py --smoke    # fast wiring check: 1 strategy x 1 query, no files written
"""
from __future__ import annotations

import argparse

from rag.evaluate import run_all, write_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG chunking-strategy evaluation")
    parser.add_argument("--smoke", action="store_true", help="fast 1x1 wiring check")
    args = parser.parse_args()

    report = run_all(smoke=args.smoke)

    if args.smoke:
        strat = next(iter(report["strategies"]))
        ev = report["strategies"][strat]["per_query"][0]
        print("\n=== SMOKE RESULT ===")
        print(f"answer: {ev['answer']}")
        print(f"P@3={ev['precision_at_3']:.2f} faith={ev['faithfulness']:.2f} rel={ev['answer_relevance']:.2f}")
        print("Wiring OK." if ev["answer"] else "Empty answer — check Ollama.")
    else:
        write_outputs(report)


if __name__ == "__main__":
    main()
