from news_rag.factory import build_pipeline

EVAL_QUESTIONS = [
    "What happened with Obama?",
    "What did Trump say about immigration?",
    "What happened in climate policy?",
]


def main() -> None:
    pipeline = build_pipeline()

    for question in EVAL_QUESTIONS:
        print("=" * 80)
        print(f"Question: {question}")
        print(pipeline.answer(question, top_k=3))


if __name__ == "__main__":
    main()
