"""Lightweight starter script for retrieval evaluation.

This is intentionally simple. Expand it with labeled queries and expected documents.
"""

from news_rag.factory import build_pipeline

EVAL_QUERIES = [
    "What happened with Obama?",
    "What did Trump say about immigration?",
    "What happened in climate policy?",
]


def main():
    pipeline = build_pipeline()
    for query in EVAL_QUERIES:
        print("=" * 80)
        print(f"Query: {query}")
        print(pipeline.answer(query, top_k=3))


if __name__ == "__main__":
    main()
