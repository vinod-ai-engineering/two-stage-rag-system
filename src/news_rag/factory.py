from news_rag.clients import create_openai_client, get_or_create_index
from news_rag.config import get_settings
from news_rag.context_builder import ContextBuilder
from news_rag.data_loader import NewsDataLoader
from news_rag.embedder import OpenAIEmbedder
from news_rag.generator import AnswerGenerator
from news_rag.indexer import NewsIndexer
from news_rag.pipeline import NewsRAGPipeline
from news_rag.retriever import NewsRetriever


def build_pipeline() -> NewsRAGPipeline:
    settings = get_settings()
    openai_client = create_openai_client(settings)
    index = get_or_create_index(settings)
    embedder = OpenAIEmbedder(openai_client, settings)
    retriever = NewsRetriever(index, embedder, settings)
    context_builder = ContextBuilder()
    generator = AnswerGenerator(openai_client, settings)
    return NewsRAGPipeline(retriever, context_builder, generator)


def build_indexer() -> NewsIndexer:
    settings = get_settings()
    openai_client = create_openai_client(settings)
    index = get_or_create_index(settings)
    embedder = OpenAIEmbedder(openai_client, settings)
    loader = NewsDataLoader(settings)
    return NewsIndexer(index, embedder, loader, settings)
