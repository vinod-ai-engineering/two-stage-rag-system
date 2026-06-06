from types import SimpleNamespace

from news_rag.pipeline import NewsRAGPipeline


class FakeRetriever:
    def search(self, query, top_k=3):
        return [SimpleNamespace(score=0.9, metadata={"title": "T", "article_content": "C"})]


class FakeContextBuilder:
    def build(self, matches):
        return "context"


class FakeGenerator:
    def generate(self, query, context):
        return f"answer for {query} using {context}"


def test_pipeline_returns_answer():
    pipeline = NewsRAGPipeline(FakeRetriever(), FakeContextBuilder(), FakeGenerator())
    assert pipeline.answer("question") == "answer for question using context"
