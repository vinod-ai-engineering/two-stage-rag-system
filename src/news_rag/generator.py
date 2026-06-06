from openai import OpenAI

from news_rag.config import Settings


class AnswerGenerator:
    """Generates grounded answers using retrieved context."""

    SYSTEM_PROMPT = """
You are a careful news research assistant.
Answer the user's question using only the provided context.
If the context is not enough, say that the available articles do not provide enough information.
Mention article titles or sources when useful.
""".strip()

    def __init__(self, client: OpenAI, settings: Settings):
        self.client = client
        self.settings = settings

    def generate(self, query: str, context: str) -> str:
        response = self.client.chat.completions.create(
            model=self.settings.chat_model,
            temperature=0,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"},
            ],
        )
        return response.choices[0].message.content or "No answer generated."
