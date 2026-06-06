from types import SimpleNamespace

from news_rag.context_builder import ContextBuilder


def test_context_builder_includes_article_fields():
    matches = [
        SimpleNamespace(
            score=0.91,
            metadata={
                "title": "Sample Title",
                "publication": "Sample Publication",
                "date": "2026-01-01",
                "url": "https://example.com",
                "article_content": "Sample article content",
            },
        )
    ]

    context = ContextBuilder().build(matches)

    assert "Sample Title" in context
    assert "Sample Publication" in context
    assert "Sample article content" in context
