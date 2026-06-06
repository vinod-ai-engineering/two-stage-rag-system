import pytest

from news_rag.pipeline import NewsRAGPipeline


class FakeRetriever:
    def search(self, query: str, top_k: int = 3):
        return []


class FakeContextBuilder:
    def build(self, matches):
        return "context"


class FakeGenerator:
    def generate(self, query: str, context: str):
        return "answer"


def test_pipeline_returns_no_results_message():
    pipeline = NewsRAGPipeline(FakeRetriever(), FakeContextBuilder(), FakeGenerator())
    assert pipeline.answer("unknown") == "No relevant articles found."


def test_pipeline_rejects_empty_query():
    pipeline = NewsRAGPipeline(FakeRetriever(), FakeContextBuilder(), FakeGenerator())
    with pytest.raises(ValueError):
        pipeline.answer("")
