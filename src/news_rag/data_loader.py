from collections.abc import Iterator

import pandas as pd

from news_rag.config import Settings


class NewsDataLoader:
    """Reads news CSV data in chunks to keep ingestion memory-friendly."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def iter_chunks(self) -> Iterator[pd.DataFrame]:
        return pd.read_csv(
            self.settings.news_csv_path,
            chunksize=self.settings.read_chunk_size,
            nrows=self.settings.total_rows,
        )
