import time

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

from news_rag.config import Settings


def build_openai_client(settings: Settings) -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


def build_pinecone_client(settings: Settings) -> Pinecone:
    return Pinecone(api_key=settings.pinecone_api_key)


def ensure_pinecone_index(pc: Pinecone, settings: Settings):
    """Create or reuse the configured Pinecone index."""
    index_names = [idx.name for idx in pc.list_indexes()]

    if settings.reset_index and settings.pinecone_index_name in index_names:
        pc.delete_index(settings.pinecone_index_name)
        time.sleep(10)
        index_names = [idx.name for idx in pc.list_indexes()]

    if settings.pinecone_index_name not in index_names:
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.embed_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )
        time.sleep(10)

    return pc.Index(settings.pinecone_index_name)
