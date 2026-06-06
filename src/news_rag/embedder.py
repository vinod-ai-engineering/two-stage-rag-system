from typing import List

from openai import OpenAI

from news_rag.config import Settings


class OpenAIEmbedder:
    """Small wrapper around OpenAI embeddings for testability."""

    def __init__(self, client: OpenAI, settings: Settings):
        self.client = client
        self.settings = settings

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        response = self.client.embeddings.create(
            model=self.settings.embed_model,
            input=texts,
        )
        return [item.embedding for item in response.data]
