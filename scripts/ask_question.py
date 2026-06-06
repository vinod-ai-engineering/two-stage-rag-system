import argparse

from news_rag.factory import build_pipeline


def main():
    parser = argparse.ArgumentParser(description="Ask the News RAG pipeline a question.")
    parser.add_argument("question", type=str)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    pipeline = build_pipeline()
    answer = pipeline.answer(args.question, top_k=args.top_k)
    print(answer)


if __name__ == "__main__":
    main()
