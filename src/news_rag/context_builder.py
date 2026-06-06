class ContextBuilder:
    """Builds grounded context blocks from retrieved Pinecone matches."""

    def build(self, matches) -> str:
        context_blocks = []

        for rank, match in enumerate(matches, start=1):
            metadata = match.metadata or {}
            block = f"""
ARTICLE {rank}
Similarity Score: {match.score}
Title: {metadata.get("title", "")}
Publication: {metadata.get("publication", "")}
Date: {metadata.get("date", "")}
URL: {metadata.get("url", "")}

Content:
{metadata.get("article_content", "")}
"""
            context_blocks.append(block.strip())

        return "\n\n---\n\n".join(context_blocks)
