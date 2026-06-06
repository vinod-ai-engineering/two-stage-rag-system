from news_rag.factory import build_indexer


def main() -> None:
    indexer = build_indexer()
    total = indexer.ingest()
    print(f"Total records indexed: {total}")


if __name__ == "__main__":
    main()
