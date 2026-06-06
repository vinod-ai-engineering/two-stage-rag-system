# Design Decisions

## Decision 1: Use OpenAI `text-embedding-3-small`

Reason: It provides strong semantic embeddings at a practical cost for a portfolio-scale RAG project.

Tradeoff: The embedding dimension is 1536, so the Pinecone index must be created with the same dimension.

## Decision 2: Use Pinecone Serverless

Reason: Pinecone provides managed vector search and is simple to use for RAG prototypes and portfolio projects.

Tradeoff: Requires external API access and careful management of index dimensions and metadata limits.

## Decision 3: Store article snippet in metadata

Reason: This keeps the project simple and allows the answer generator to use retrieved context without a second database.

Tradeoff: Pinecone metadata has size limits, so long articles must be trimmed.

## Decision 4: Batch ingestion

Reason: Embedding and upserting one row at a time is slow and expensive.

Tradeoff: Batch failures need better retry handling in future versions.

## Decision 5: Keep prompt grounding strict

Reason: The model should answer from retrieved articles only, not from general knowledge.

Tradeoff: If retrieval is weak, the answer should admit insufficient context instead of guessing.
