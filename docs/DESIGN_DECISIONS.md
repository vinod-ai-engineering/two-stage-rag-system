# Design Decisions

- **Title-only embeddings**: Simpler, cheaper, faster indexing. Trade-off: weaker semantic retrieval on full content.
- **Metadata in Pinecone**: Enables rich context without re-fetching full articles.
- **Strict grounding prompt**: Reduces hallucinations.
- **Modular package (`src/news_rag`)**: Makes it reusable and testable.
- **Multi-interface**: CLI + FastAPI + Streamlit for flexibility.