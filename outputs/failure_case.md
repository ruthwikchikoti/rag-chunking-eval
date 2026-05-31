# Failure Case

**Strategy:** `fixed`  
**Query (factual):** According to the essay, what are the three things you need to do great work?

**Scores:** Precision@3 = 0.67 | Faithfulness = 0.67 | Answer relevance = 0.70

**Generated answer:**

> According to the context passage [1], the first step is to decide what to work on, and it needs to have three qualities: 

1. Something you have a natural aptitude for
2. Something you have a deep interest in
3. (Although not explicitly stated as necessary) something that offers scope to do great work

**Retrieved top-3 chunks (relevance judged):**

**[1]** (score=0.435, ✅ relevant)

> How to Do Great Work July 2023 If you collected lists of techniques for doing great work in a lot of different fields, what would the intersection look like? I decided to find out by making it. Partly my goal was to create a guide that could be used by someone working in any field. But I was also curious about the shap…

**[2]** (score=0.356, ✅ relevant)

> a bet as you'll find. [ 3 ] Four steps: choose a field, learn enough to get to the frontier, notice gaps, explore promising ones. This is how practically everyone who's done great work has done it, from painters to physicists. Steps two and four will require hard work. It may not be possible to prove that you have to w…

**[3]** (score=0.347, ❌ not relevant)

> of work. You can take ideas from quite distant fields if you let them be metaphors. Negative examples can be as inspiring as positive ones. In fact you can sometimes learn more from things done badly than from things done well; sometimes it only becomes clear what's needed when it's missing. If a lot of the best people…

**Why it failed / mitigation:** The retriever surfaced passages that are topically near the query but miss the specific supporting facts, so the generator either abstains or stitches a partially-supported answer. This is the classic chunk-boundary problem: the evidence the query needs is split across chunk edges or diluted by unrelated sentences. Mitigations: smaller/overlapping or semantically-coherent chunks, retrieving more candidates then re-ranking, and the strict 'answer only from context, else say I don't know' prompt that keeps an unsupported answer from being emitted as fact.