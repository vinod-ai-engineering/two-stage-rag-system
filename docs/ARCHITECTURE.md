# System Architecture

## High-Level Flow
1. **Data Ingestion** → Load CSV → Generate title embeddings → Store in Pinecone (with metadata)
2. **Query** → Embed user question → Retrieve top-k candidates by title similarity
3. **Context Building** → Construct grounded context from metadata + snippets
4. **Generation** → LLM answers strictly from provided context

## Two-Stage Design
- **Stage 1**: Fast title-based semantic search
- **Stage 2**: Metadata-aware context assembly + grounded generation

See `images/architecture.png` for diagram.