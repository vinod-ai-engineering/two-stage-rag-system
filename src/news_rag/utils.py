import json
from typing import Any, Dict

import pandas as pd
import tiktoken


def safe_text(value: Any) -> str:
    """Convert null/NaN values into safe stripped strings."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def truncate_by_tokens(text: str, max_tokens: int, encoding_name: str = "cl100k_base") -> str:
    """Trim text to stay within a token budget."""
    text = safe_text(text)
    tokenizer = tiktoken.get_encoding(encoding_name)
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.decode(tokens[:max_tokens])


def metadata_size_bytes(metadata: Dict[str, Any]) -> int:
    """Calculate metadata size in bytes before sending to Pinecone."""
    return len(json.dumps(metadata, ensure_ascii=False).encode("utf-8"))


def fit_metadata(metadata: Dict[str, Any], max_metadata_bytes: int) -> Dict[str, Any]:
    """Trim article_content until the metadata payload fits within Pinecone limits."""
    article = metadata.get("article_content", "")

    while metadata_size_bytes(metadata) > max_metadata_bytes and len(article) > 0:
        article = article[:-500]
        metadata["article_content"] = article

    if metadata_size_bytes(metadata) > max_metadata_bytes:
        metadata["article_content"] = ""

    return metadata
