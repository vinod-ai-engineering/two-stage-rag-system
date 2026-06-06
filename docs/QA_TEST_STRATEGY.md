# QA Test Strategy for GenAI RAG

## Tested Areas
- Retrieval relevance (manual + automated)
- Groundedness / Hallucination checks
- Empty result handling
- Token limit safety
- API error resilience

## Tools
- Unit tests (`tests/`)
- `scripts/evaluate_retrieval.py`
- Manual adversarial prompting