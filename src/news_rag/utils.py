import json
from typing import Any

import pandas as pd
import tiktoken


def safe_text(value: Any) -> str:
    """Convert null/NaN values into clean strings."""
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def truncate_by_tokens(text: str, max_tokens: int) -> str:
    """Trim text to a maximum token count."""
    clean_text = safe_text(text)
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(clean_text)

    if len(tokens) <= max_tokens:
        return clean_text

    return tokenizer.decode(tokens[:max_tokens])


def metadata_size_bytes(metadata: dict[str, Any]) -> int:
    """Return metadata size in bytes for vector database safety checks."""
    return len(json.dumps(metadata, ensure_ascii=False).encode("utf-8"))


def fit_metadata(metadata: dict[str, Any], max_metadata_bytes: int) -> dict[str, Any]:
    """Trim article content until metadata fits within the configured byte limit."""
    fitted = dict(metadata)
    article = fitted.get("article_content", "")

    while metadata_size_bytes(fitted) > max_metadata_bytes and len(article) > 0:
        article = article[:-500]
        fitted["article_content"] = article

    if metadata_size_bytes(fitted) > max_metadata_bytes:
        fitted["article_content"] = ""

    return fitted
