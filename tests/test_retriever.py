from news_rag.retriever import NewsRetriever


class FakeEmbedder:
    def embed_texts(self, texts):
        return [[0.1, 0.2, 0.3]]


class FakeIndex:
    def query(self, **kwargs):
        self.kwargs = kwargs

        class Result:
            matches = ["match-1"]

        return Result()


class FakeSettings:
    pinecone_namespace = "__default__"


def test_retriever_queries_index_with_namespace():
    index = FakeIndex()
    retriever = NewsRetriever(index, FakeEmbedder(), FakeSettings())
    matches = retriever.search("test", top_k=1)

    assert matches == ["match-1"]
    assert index.kwargs["top_k"] == 1
    assert index.kwargs["namespace"] == "__default__"
