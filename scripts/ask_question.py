import argparse

from news_rag.factory import build_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question against the news RAG pipeline.")
    parser.add_argument("question", help="Question to ask")
    parser.add_argument("--top-k", type=int, default=3, help="Number of articles to retrieve")
    args = parser.parse_args()

    pipeline = build_pipeline()
    answer = pipeline.answer(args.question, top_k=args.top_k)
    print(answer)


if __name__ == "__main__":
    main()
