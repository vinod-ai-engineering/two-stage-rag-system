# Evaluation Plan

## Why evaluation matters

A RAG system is only useful if it retrieves relevant context and generates answers that are grounded in that context.

## Starter evaluation table

| Query | Expected Theme | Retrieved Relevant? | Answer Grounded? | Notes |
|---|---|---:|---:|---|
| What happened with Obama? | Politics / Obama news | TBD | TBD | Verify titles and context relevance |
| What did Trump say about immigration? | Politics / immigration | TBD | TBD | Check if title-only retrieval is enough |
| What happened in climate policy? | Climate / policy | TBD | TBD | Good candidate for semantic retrieval |

## Metrics to add

- Recall@K
- Precision@K
- Mean Reciprocal Rank
- Context relevance score
- Answer groundedness score
- Hallucination rate
- Average retrieval latency
- Average answer generation latency

## Manual groundedness checklist

For each answer:

1. Does the answer use only retrieved context?
2. Does it avoid unsupported claims?
3. Does it mention article titles/sources when useful?
4. Does it say insufficient context when needed?
5. Does the answer directly address the user question?
