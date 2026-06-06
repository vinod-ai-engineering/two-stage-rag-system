import time
from typing import List

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

from news_rag.config import Settings


def create_openai_client(settings: Settings) -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


def create_pinecone_client(settings: Settings) -> Pinecone:
    return Pinecone(api_key=settings.pinecone_api_key)


def list_index_names(pc: Pinecone) -> List[str]:
    return [idx.name for idx in pc.list_indexes()]


def get_or_create_index(settings: Settings):
    """Create or reuse Pinecone index with the configured embedding dimension."""
    pc = create_pinecone_client(settings)
    names = list_index_names(pc)

    if settings.reset_index and settings.pinecone_index_name in names:
        pc.delete_index(settings.pinecone_index_name)
        time.sleep(10)
        names = list_index_names(pc)

    if settings.pinecone_index_name not in names:
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
