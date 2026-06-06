from news_rag.utils import fit_metadata, metadata_size_bytes, safe_text


def test_safe_text_handles_none():
    assert safe_text(None) == ""


def test_fit_metadata_trims_large_article():
    metadata = {"title": "Test", "article_content": "x" * 5000}
    fitted = fit_metadata(metadata, max_metadata_bytes=1000)
    assert metadata_size_bytes(fitted) <= 1000
