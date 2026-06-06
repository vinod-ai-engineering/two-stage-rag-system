# Architecture

## Overview

This project is a Retrieval-Augmented Generation pipeline for answering questions from a news article dataset.

The system has two main runtime paths:

1. **Ingestion path** — reads news data, creates embeddings, and indexes vectors in Pinecone.
2. **Query path** — retrieves relevant articles and generates a grounded answer.

## Ingestion Flow

```text
CSV Dataset
  ↓
Read rows in batches
  ↓
Validate title and article fields
  ↓
Embed article title
  ↓
Build Pinecone record
  ↓
Store vector + metadata in Pinecone
```

## Query Flow

```text
User Question
  ↓
Create query embedding
  ↓
Search Pinecone title vectors
  ↓
Retrieve top-k article candidates
  ↓
Build context block from metadata
  ↓
Send context + question to LLM
  ↓
Return grounded answer
```

## Why title embeddings?

The first version embeds article titles because titles are short, cheaper to embed, and useful for fast candidate retrieval.

## Tradeoff

Title-only retrieval is fast and simple, but it can miss relevant articles when the answer appears in the article body but not in the title.

## Recommended future architecture

```text
Article Body
  ↓
Chunking
  ↓
Chunk Embeddings
  ↓
Vector Search
  ↓
Reranking
  ↓
Context Compression
  ↓
Grounded Answer
```
