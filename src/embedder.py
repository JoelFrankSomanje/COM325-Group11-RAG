from typing import List
from langchain_community.embeddings import (
    HuggingFaceEmbeddings,
    OllamaEmbeddings
)
from langchain_openai import OpenAIEmbeddings


def get_embedder(
    provider: str = "huggingface",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
):
    """
    Get embedding model with flexibility.

    Supports:
    - huggingface (default)
    - ollama (local)
    - openai (API-based)
    """

    if provider == "huggingface":
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

    elif provider == "ollama":
        return OllamaEmbeddings(model=model_name)

    elif provider == "openai":
        return OpenAIEmbeddings(model=model_name)

    else:
        raise ValueError(f"Unknown provider: {provider}")


def embed_documents(embedder, documents: List, batch_size: int = 32):
    """
    Embed documents with batching for efficiency.
    """
    texts = [doc.page_content for doc in documents]
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings.extend(embedder.embed_documents(batch))

    return embeddings


def embed_query(embedder, query: str):
    """
    Embed a single query.
    """
    return embedder.embed_query(query)