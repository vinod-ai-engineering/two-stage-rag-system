from types import SimpleNamespace

from news_rag.context_builder import ContextBuilder


def test_context_builder_includes_article_metadata():
    match = SimpleNamespace(
        score=0.91,
        metadata={
            "title": "Sample Title",
            "publication": "Sample Publication",
            "date": "2026-01-01",
            "url": "https://example.com",
            "article_content": "Sample article content.",
        },
    )

    context = ContextBuilder().build([match])

    assert "ARTICLE 1" in context
    assert "Sample Title" in context
    assert "Sample article content." in context
