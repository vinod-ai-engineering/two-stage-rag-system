# Limitations

This project is intentionally transparent about what is implemented and what is not.

## Current limitations

1. **Title-only embeddings**
   - The current implementation embeds article titles, not full article body chunks.
   - This is useful for fast candidate retrieval but not ideal for deep question answering.

2. **No production reranker yet**
   - Retrieved candidates are not reranked by a cross-encoder or LLM scoring model.

3. **Article snippets stored in metadata**
   - This is simple but limited by Pinecone metadata-size limits.

4. **Dataset not bundled**
   - The dataset must be downloaded separately depending on licensing and source availability.

5. **Evaluation is starter-level**
   - More labeled queries and retrieval metrics should be added.

## Future fixes

- Add article chunking.
- Store chunks as vectors.
- Add hybrid dense + sparse search.
- Add reranking.
- Add source-level citations.
- Add automated evaluation.
