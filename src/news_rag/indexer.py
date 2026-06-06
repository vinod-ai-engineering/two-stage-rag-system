from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from news_rag.config import Settings
from news_rag.data_loader import NewsDataLoader
from news_rag.embedder import OpenAIEmbedder
from news_rag.utils import fit_metadata, safe_text, truncate_by_tokens


class NewsIndexer:
    """Builds Pinecone records and upserts title embeddings with article metadata."""

    def __init__(
        self,
        index,
        embedder: OpenAIEmbedder,
        loader: NewsDataLoader,
        settings: Settings,
    ):
        self.index = index
        self.embedder = embedder
        self.loader = loader
        self.settings = settings

    def build_record(
        self,
        row_id: int,
        row: pd.Series,
        title_vector: list[float],
    ) -> dict[str, Any]:
        title = safe_text(row.get("title"))
        article = safe_text(row.get("article"))
        article = truncate_by_tokens(article, self.settings.max_embed_tokens)

        metadata = {
            "title": title,
            "article_content": article[: self.settings.max_article_chars],
            "url": safe_text(row.get("url")),
            "publication": safe_text(row.get("publication")),
            "date": safe_text(row.get("date")),
            "section": safe_text(row.get("section")),
            "author": safe_text(row.get("author")),
        }
        metadata = fit_metadata(metadata, self.settings.max_metadata_bytes)

        return {
            "id": f"news-{row_id}",
            "values": title_vector,
            "metadata": metadata,
        }

    def ingest(self) -> int:
        total_indexed = 0

        for chunk in tqdm(self.loader.iter_chunks(), desc="Reading CSV batches"):
            prepared_rows = []

            for row_id, row in chunk.iterrows():
                title = safe_text(row.get("title"))
                article = safe_text(row.get("article"))

                if not title or not article:
                    continue

                prepared_rows.append({"row_id": int(row_id), "title": title, "row": row})

            if not prepared_rows:
                continue

            titles = [item["title"] for item in prepared_rows]
            title_vectors = self.embedder.embed_texts(titles)

            records = [
                self.build_record(item["row_id"], item["row"], vector)
                for item, vector in zip(prepared_rows, title_vectors, strict=False)
            ]

            for start in range(0, len(records), self.settings.upsert_batch_size):
                batch = records[start : start + self.settings.upsert_batch_size]
                self.index.upsert(
                    vectors=batch,
                    namespace=self.settings.pinecone_namespace,
                )
                total_indexed += len(batch)

        return total_indexed
