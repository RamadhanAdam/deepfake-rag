"""
evaluate_rag.py
Simple evaluation script for the RAG pipeline.

This is not a unit test. It checks if retrieval brings back useful sources,
and can optionally check if the generated explanation looks reasonable.

Usage:
    python tests/evaluate_rag.py

To also test Groq generation:
    RUN_GENERATION_EVAL=1 python tests/evaluate_rag.py
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag import RAGPipeline


# Small hand-made evaluation set.
# expected_sources are source names we hope to see in the retrieved chunks.
EVAL_QUERIES = [
    {
        "query": "What artifacts are common in GAN generated deepfake images?",
        "expected_sources": [
            "Watch Your Up-Convolution",
            "Deepfake Generation and Detection Survey",
        ],
    },
    {
        "query": "What is FaceForensics++ used for in deepfake detection?",
        "expected_sources": [
            "FaceForensics++",
        ],
    },
    {
        "query": "Why can deepfake detectors fail on unseen datasets?",
        "expected_sources": [
            "Deepfake Detection Reliability Survey",
            "Tug of War Deepfake Detection",
        ],
    },
    {
        "query": "How does Xception help with image classification?",
        "expected_sources": [
            "Xception",
        ],
    },
]


def source_matches(source, expected_sources):
    """Check if a retrieved source matches any expected source."""
    return any(expected.lower() in source.lower() for expected in expected_sources)


def evaluate_retrieval(rag, k=3):
    """Evaluate retrieval using simple Recall@k."""
    correct = 0

    print(f"Retrieval evaluation, Recall@{k}\n")

    for item in EVAL_QUERIES:
        query = item["query"]
        expected_sources = item["expected_sources"]

        retrieved_chunks, distances = rag.retrieve(query, k=k)
        retrieved_sources = [chunk["source"] for chunk in retrieved_chunks]

        hit = any(source_matches(source, expected_sources) for source in retrieved_sources)
        correct += int(hit)

        print(f"Query: {query}")
        print(f"Expected: {expected_sources}")
        print(f"Retrieved: {retrieved_sources}")
        print(f"Result: {'PASS' if hit else 'FAIL'}")
        print("-" * 60)

    recall = correct / len(EVAL_QUERIES)
    print(f"\nRecall@{k}: {recall:.2f} ({correct}/{len(EVAL_QUERIES)})")
    return recall


def evaluate_generation(rag):
    """Do a small sanity check on generated explanations."""
    examples = [
        ("DEEPFAKE", 0.94),
        ("REAL", 0.87),
    ]

    print("\nGeneration evaluation\n")

    for label, confidence in examples:
        explanation = rag.explain(label, confidence)

        long_enough = len(explanation) > 100
        mentions_prediction = label.lower() in explanation.lower()

        print(f"Prediction: {label}")
        print(f"Length: {len(explanation)} characters")
        print(f"Long enough: {'PASS' if long_enough else 'FAIL'}")
        print(f"Mentions prediction: {'PASS' if mentions_prediction else 'FAIL'}")
        print(f"\nExplanation:\n{explanation}")
        print("-" * 60)


if __name__ == "__main__":
    rag = RAGPipeline()
    rag.load()

    evaluate_retrieval(rag, k=3)

    # Generation calls Groq, so keep it optional.
    if os.getenv("RUN_GENERATION_EVAL") == "1":
        evaluate_generation(rag)
    else:
        print("\nSkipped generation evaluation.")
        print("Run with RUN_GENERATION_EVAL=1 to test generated explanations.")
