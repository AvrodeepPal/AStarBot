"""Calibrate MIN_RETRIEVAL_SCORE against the current knowledge base.

    python -m scripts.calibrate

Runs entirely locally: it embeds data/*.json exactly as scripts.embed would,
then scores a fixed probe set of on-topic and off-topic questions against it.
No Pinecone, no Groq, no API keys needed beyond what rag.config demands.

It prints the score distribution for both groups and suggests a threshold
halfway through the gap between the worst on-topic hit and the best off-topic
one. Re-run after editing data/*.json — a threshold calibrated against an old
knowledge base is worse than no threshold at all.
"""

import numpy as np

from rag.config import settings
from rag.knowledge import embedding_text
from rag.retriever import load_embedder
from scripts.embed import load_records

# Questions a real visitor would ask; every one SHOULD retrieve something.
ON_TOPIC = [
    "Who is Avrodeep?",
    "Where does he work?",
    "What does he do at GenAIus?",
    "Tell me about his education",
    "Which university did he go to?",
    "What programming languages does he know?",
    "Is he good at Python?",
    "What projects has he built?",
    "Tell me about the Bengali idioms project",
    "Has he worked with RAG systems?",
    "Does he know about LLMs?",
    "What is his experience with cybersecurity?",
    "Is he open to new job opportunities?",
    "Is he preparing for GATE?",
    "Does he want to do a PhD?",
    "How can I contact him?",
    "What is his GitHub?",
    "Where is he based?",
    "What does he value in a workplace?",
    "What are his strengths?",
    "Does he do freelance work?",
    "What is he working on right now?",
    "Tell me about his LeetCode practice",
    "What research is he doing?",
]

# Questions the bot SHOULD refuse; ideally none of these clear the threshold.
OFF_TOPIC = [
    "What's the weather in Paris?",
    "Write me a Python script to scrape a website",
    "Who won the 2024 election?",
    "What is the capital of Brazil?",
    "Can you help me debug my React app?",
    "Give me a recipe for biryani",
    "What do you think about cryptocurrency?",
]


def main() -> None:
    records, _ = load_records()
    embedder = load_embedder()

    doc_vecs = embedder.encode(
        [embedding_text(r) for r in records],
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=32,
    )

    def top_scores(questions: list[str]) -> list[tuple[float, str, str]]:
        """For each question: (best score, question, top-3 entry ids)."""
        out = []
        for q in questions:
            qv = embedder.encode(settings.query_prefix + q, normalize_embeddings=True, convert_to_numpy=True)
            sims = doc_vecs @ qv
            order = np.argsort(-sims)[:3]
            ids = ", ".join(records[i]["id"] for i in order)
            out.append((float(sims[order[0]]), q, ids))
        return out

    on = sorted(top_scores(ON_TOPIC))
    off = sorted(top_scores(OFF_TOPIC), reverse=True)

    print(f"\nKnowledge base: {len(records)} entries\n")
    print("ON-TOPIC (worst first) — top-3 retrieved ids shown; k=5 in production")
    for s, q, rid in on:
        print(f"  {s:.3f}  {q[:48]:<50} -> {rid}")

    print("\nOFF-TOPIC (best first) — a threshold can only help if these stay BELOW it")
    for s, q, rid in off:
        print(f"  {s:.3f}  {q[:48]:<50} -> {rid}")

    worst_on, best_off = on[0][0], off[0][0]
    print(f"\nworst on-topic : {worst_on:.3f}")
    print(f"best off-topic : {best_off:.3f}")
    print(f"gap            : {worst_on - best_off:+.3f}")

    if worst_on > best_off:
        suggested = round((worst_on + best_off) / 2, 2)
        print(f"\nSuggested MIN_RETRIEVAL_SCORE = {suggested}  (current: {settings.min_retrieval_score})")
    else:
        print(
            "\nThe two groups OVERLAP — no threshold separates them cleanly.\n"
            "Keep MIN_RETRIEVAL_SCORE at 0.0 and rely on the prompt's grounding\n"
            "rules to refuse, or add knowledge-base entries for the on-topic\n"
            "questions that scored lowest."
        )


if __name__ == "__main__":
    main()
