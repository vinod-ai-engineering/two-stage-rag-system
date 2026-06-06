from news_rag.clients import build_openai_client, build_pinecone_client, ensure_pinecone_index
from news_rag.config import get_settings
from news_rag.context_builder import ContextBuilder
from news_rag.data_loader import NewsDataLoader
from news_rag.embedder import OpenAIEmbedder
from news_rag.generator import AnswerGenerator
from news_rag.indexer import NewsIndexer
from news_rag.pipeline import NewsRAGPipeline
from news_rag.retriever import NewsRetriever


def build_components():
    settings = get_settings()
    openai_client = build_openai_client(settings)
    pinecone_client = build_pinecone_client(settings)
    index = ensure_pinecone_index(pinecone_client, settings)
    embedder = OpenAIEmbedder(openai_client, settings)
    return settings, openai_client, index, embedder


def build_indexer() -> NewsIndexer:
    settings, _openai_client, index, embedder = build_components()
    loader = NewsDataLoader(settings)
    return NewsIndexer(index, embedder, loader, settings)


def build_pipeline() -> NewsRAGPipeline:
    settings, openai_client, index, embedder = build_components()
    retriever = NewsRetriever(index, embedder, settings)
    context_builder = ContextBuilder()
    generator = AnswerGenerator(openai_client, settings)
    return NewsRAGPipeline(retriever, context_builder, generator)
