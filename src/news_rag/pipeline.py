from news_rag.context_builder import ContextBuilder
from news_rag.generator import AnswerGenerator
from news_rag.retriever import NewsRetriever


class NewsRAGPipeline:
    """Coordinates retrieval, context construction, and answer generation."""

    def __init__(
        self,
        retriever: NewsRetriever,
        context_builder: ContextBuilder,
        generator: AnswerGenerator,
    ):
        self.retriever = retriever
        self.context_builder = context_builder
        self.generator = generator

    def answer(self, query: str, top_k: int = 3) -> str:
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        matches = self.retriever.search(query=query, top_k=top_k)

        if not matches:
            return "No relevant articles found."

        context = self.context_builder.build(matches)
        return self.generator.generate(query=query, context=context)
