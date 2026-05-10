```python
# ============================================================
# TWO STAGE RAG SYSTEM - SIMPLIFIED VERSION
# ============================================================

# -----------------------------
# STEP 1 - INSTALL PACKAGES
# -----------------------------
# Run once if needed

# !pip install pandas tiktoken openai pinecone tqdm


# -----------------------------
# STEP 2 - DOWNLOAD DATASET
# -----------------------------

# !wget -q --show-progress -O all-the-news-3.zip "https://www.dropbox.com/scl/fi/wruzj2bwyg743d0jzd7ku/all-the-news-3.zip?rlkey=rgwtwpeznbdadpv3f01sznwxa&dl=1"
# !unzip -o all-the-news-3.zip


# -----------------------------
# STEP 3 - IMPORT LIBRARIES
# -----------------------------

import os
import time
import pandas as pd

from tqdm.auto import tqdm
from getpass import getpass

import tiktoken

from openai import OpenAI

from pinecone import Pinecone, ServerlessSpec


# -----------------------------
# STEP 4 - CONFIGURATION
# -----------------------------

INDEX_NAME = "global-news-researcher"
NAMESPACE = "__default__"

CSV_PATH = "./all-the-news-3.csv"

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"

EMBED_DIMENSION = 1536

TOTAL_ROWS = 1000
READ_CHUNK_SIZE = 100
UPSERT_BATCH_SIZE = 50

MAX_ARTICLE_CHARS = 5000

RESET_INDEX = False


# -----------------------------
# STEP 5 - API KEYS
# -----------------------------

OPENAI_API_KEY = getpass("Enter OPENAI_API_KEY: ")
PINECONE_API_KEY = getpass("Enter PINECONE_API_KEY: ")


# -----------------------------
# STEP 6 - INITIALIZE CLIENTS
# -----------------------------

openai_client = OpenAI(
    api_key=OPENAI_API_KEY
)

pc = Pinecone(
    api_key=PINECONE_API_KEY
)

tokenizer = tiktoken.get_encoding("cl100k_base")


# -----------------------------
# STEP 7 - CREATE PINECONE INDEX
# -----------------------------

existing_indexes = [
    idx.name
    for idx in pc.list_indexes()
]

if RESET_INDEX and INDEX_NAME in existing_indexes:
    pc.delete_index(INDEX_NAME)
    time.sleep(10)

if INDEX_NAME not in existing_indexes:

    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    time.sleep(10)

index = pc.Index(INDEX_NAME)

print("Pinecone index ready")


# -----------------------------
# STEP 8 - HELPER FUNCTIONS
# -----------------------------

def clean_text(value):

    if pd.isna(value):
        return ""

    return str(value).strip()


def embed_texts(texts):

    response = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )

    return [
        item.embedding
        for item in response.data
    ]


# -----------------------------
# STEP 9 - LOAD DATA INTO PINECONE
# -----------------------------

def index_news_articles():

    total_indexed = 0

    chunks = pd.read_csv(
        CSV_PATH,
        chunksize=READ_CHUNK_SIZE,
        nrows=TOTAL_ROWS
    )

    for chunk in tqdm(
        chunks,
        desc="Indexing News Articles"
    ):

        records = []

        titles = []

        rows = []

        for row_id, row in chunk.iterrows():

            title = clean_text(
                row.get("title")
            )

            article = clean_text(
                row.get("article")
            )

            if not title or not article:
                continue

            titles.append(title)

            rows.append(
                (row_id, row)
            )

        if not titles:
            continue

        title_vectors = embed_texts(
            titles
        )

        for (row_id, row), vector in zip(
            rows,
            title_vectors
        ):

            metadata = {

                "title": clean_text(
                    row.get("title")
                ),

                "article_content": clean_text(
                    row.get("article")
                )[:MAX_ARTICLE_CHARS],

                "url": clean_text(
                    row.get("url")
                ),

                "publication": clean_text(
                    row.get("publication")
                ),

                "date": clean_text(
                    row.get("date")
                ),

                "author": clean_text(
                    row.get("author")
                ),
            }

            records.append({

                "id": f"news-{row_id}",

                "values": vector,

                "metadata": metadata
            })

        for start in range(
            0,
            len(records),
            UPSERT_BATCH_SIZE
        ):

            batch = records[
                start:start + UPSERT_BATCH_SIZE
            ]

            index.upsert(
                vectors=batch,
                namespace=NAMESPACE
            )

            total_indexed += len(batch)

    return total_indexed


# -----------------------------
# STEP 10 - START INDEXING
# -----------------------------

total = index_news_articles()

print(
    f"Total records indexed: {total}"
)


# -----------------------------
# STEP 11 - CHECK INDEX STATUS
# -----------------------------

print(
    index.describe_index_stats()
)


# -----------------------------
# STEP 12 - STAGE 1 RETRIEVAL
# -----------------------------

def search_news_titles(
    query,
    top_k=3
):

    query_vector = embed_texts(
        [query]
    )[0]

    results = index.query(

        vector=query_vector,

        top_k=top_k,

        include_metadata=True,

        namespace=NAMESPACE
    )

    return results


# -----------------------------
# STEP 13 - TEST RETRIEVAL
# -----------------------------

query = "What happened with Obama?"

results = search_news_titles(
    query=query,
    top_k=3
)

for rank, match in enumerate(
    results.matches,
    start=1
):

    metadata = match.metadata

    print("\n")
    print("=" * 60)

    print(f"Rank: {rank}")

    print(f"Score: {match.score}")

    print(
        f"Title: {metadata.get('title')}"
    )

    print(
        f"Publication: {metadata.get('publication')}"
    )

    print(
        f"Date: {metadata.get('date')}"
    )

    print(
        f"URL: {metadata.get('url')}"
    )


# -----------------------------
# STEP 14 - BUILD CONTEXT
# -----------------------------

def build_context(matches):

    context = []

    for rank, match in enumerate(
        matches,
        start=1
    ):

        metadata = match.metadata

        article_block = f"""

ARTICLE {rank}

Title:
{metadata.get("title")}

Publication:
{metadata.get("publication")}

Date:
{metadata.get("date")}

URL:
{metadata.get("url")}

Content:
{metadata.get("article_content")}

"""

        context.append(
            article_block
        )

    return "\n\n---\n\n".join(
        context
    )


# -----------------------------
# STEP 15 - STAGE 2 RAG
# -----------------------------

def ask_news_bot(
    query,
    top_k=3
):

    results = search_news_titles(
        query=query,
        top_k=top_k
    )

    if not results.matches:
        return "No relevant articles found."

    context = build_context(
        results.matches
    )

    response = openai_client.chat.completions.create(

        model=CHAT_MODEL,

        temperature=0,

        messages=[

            {
                "role": "system",

                "content": """
You are a careful news research assistant.

Answer ONLY using the provided context.
"""
            },

            {
                "role": "user",

                "content": f"""

Context:
{context}

Question:
{query}

"""
            }
        ]
    )

    return response.choices[0].message.content


# -----------------------------
# STEP 16 - FINAL TEST
# -----------------------------

question = "What happened with Obama?"

answer = ask_news_bot(
    question,
    top_k=3
)

print("\n")
print("=" * 80)
print("FINAL ANSWER")
print("=" * 80)
print("\n")

print(answer)

# ============================================================
# END
# ============================================================
```
