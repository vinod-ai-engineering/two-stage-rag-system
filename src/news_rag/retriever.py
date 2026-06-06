from news_rag.config import Settings
from news_rag.embedder import OpenAIEmbedder


class NewsRetriever:
    """Stage 1: retrieve top candidate articles from Pinecone."""

    def __init__(self, index, embedder: OpenAIEmbedder, settings: Settings):
        self.index = index
        self.embedder = embedder
        self.settings = settings

    def search(self, query: str, top_k: int = 3):
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty.")

        query_vector = self.embedder.embed_texts([query])[0]
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            namespace=self.settings.pinecone_namespace,
        )
        return results.matches
