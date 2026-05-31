"""Five hand-written evaluation queries over 'How to Do Great Work' (Paul Graham).

One per required type: factual, multi-hop, negation, comparison, summarisation.
Each carries a gold answer (for manual validation) and `anchors` — distinctive phrases
that a *relevant* chunk should contain, used for the deterministic anchor_hit@3 cross-check.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Query:
    id: str
    qtype: str
    question: str
    gold_answer: str
    anchors: list[str] = field(default_factory=list)


QUERIES: list[Query] = [
    Query(
        id="Q1",
        qtype="factual",
        question="According to the essay, what are the three things you need to do great work?",
        gold_answer="Choose a field you have a natural aptitude for and a deep interest in, "
                    "and that offers scope to do great work; in practice: pick something, get to "
                    "the frontier, notice gaps, and explore promising ones.",
        anchors=["natural aptitude", "deep interest", "scope to do great work"],
    ),
    Query(
        id="Q2",
        qtype="multi-hop",
        question="How does curiosity connect to both choosing what to work on and actually "
                 "producing original work?",
        gold_answer="Curiosity is the guide: it tells you what to work on (follow your interests "
                    "rather than prestige) and it drives original work because noticing and chasing "
                    "what puzzles you is what leads to new discoveries.",
        anchors=["curiosity", "interest", "original"],
    ),
    Query(
        id="Q3",
        qtype="negation",
        question="What does Paul Graham say you should NOT let determine what you work on?",
        gold_answer="Don't let prestige, fashion, or other people's opinions decide what you work "
                    "on; prestige is especially dangerous because it distorts your interests.",
        anchors=["prestige", "fashion"],
    ),
    Query(
        id="Q4",
        qtype="comparison",
        question="How does the essay contrast working hard with working on the right things?",
        gold_answer="Effort matters, but it must be aimed: working hard on the wrong thing is wasted, "
                    "so consistently working on the right things matters more than sheer hours, though "
                    "great work still requires hard work once you've chosen well.",
        anchors=["hard", "work on the right", "effort"],
    ),
    Query(
        id="Q5",
        qtype="summarisation",
        question="Summarise the essay's overall recipe for doing great work in two or three sentences.",
        gold_answer="Pick a field you're suited to and deeply curious about, get to its frontier, "
                    "notice and chase the gaps, and work hard with honesty and persistence. Follow "
                    "curiosity over prestige, and let interest compound over time.",
        anchors=["great work", "curiosity", "frontier"],
    ),
]
