from openai import OpenAI

from news_rag.config import Settings


class OpenAIEmbedder:
    """Creates text embeddings using the configured OpenAI embedding model."""

    def __init__(self, client: OpenAI, settings: Settings):
        self.client = client
        self.settings = settings

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.settings.embed_model,
            input=texts,
        )
        return [item.embedding for item in response.data]
