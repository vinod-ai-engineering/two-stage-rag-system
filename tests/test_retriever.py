import pytest

from news_rag.retriever import NewsRetriever


class FakeEmbedder:
    def embed_texts(self, texts):
        return [[0.1, 0.2, 0.3]]


class FakeSettings:
    pinecone_namespace = "__default__"


class FakeIndex:
    def query(self, **kwargs):
        class Result:
            matches = []

        return Result()


def test_retriever_rejects_empty_query():
    retriever = NewsRetriever(FakeIndex(), FakeEmbedder(), FakeSettings())
    with pytest.raises(ValueError):
        retriever.search("   ")
