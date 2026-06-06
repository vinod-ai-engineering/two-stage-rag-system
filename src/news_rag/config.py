from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env."""

    openai_api_key: str
    pinecone_api_key: str
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_index_name: str = "global-news-researcher"
    pinecone_namespace: str = "__default__"
    news_csv_path: str = "./data/all-the-news-3.csv"

    embed_model: str = "text-embedding-3-small"
    embed_dimension: int = 1536
    chat_model: str = "gpt-4o"

    total_rows: int = 1000
    read_chunk_size: int = 100
    upsert_batch_size: int = 50

    max_embed_tokens: int = 8000
    max_metadata_bytes: int = 40960
    max_article_chars: int = 5000
    reset_index: bool = False

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


def get_settings() -> Settings:
    return Settings()
