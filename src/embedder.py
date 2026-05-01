"""
Ollama Embedding Model
======================
Local embedding helpers for the RAG pipeline.
"""

from typing import List

try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    from langchain_community.embeddings import OllamaEmbeddings


DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"


def get_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL, **kwargs) -> OllamaEmbeddings:
    """
    Create an Ollama embedding model.

    Use an embedding-focused Ollama model such as:
    - nomic-embed-text
    - mxbai-embed-large

    Pull one first with, for example:
        ollama pull nomic-embed-text
    """
    return OllamaEmbeddings(model=model_name, **kwargs)


def embed_documents(embedder, documents: List, batch_size: int = 32) -> List[List[float]]:
    """
    Embed document page content in batches.

    Batching keeps large document sets friendlier to a local Ollama server.
    """
    texts = [doc.page_content for doc in documents if doc.page_content.strip()]
    embeddings: List[List[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        embeddings.extend(embedder.embed_documents(batch))

    return embeddings


def embed_query(embedder, query: str) -> List[float]:
    """Embed a single query string with Ollama."""
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    return embedder.embed_query(query)


if __name__ == "__main__":
    embedder = get_embedder()
    test_text = "What is Retrieval-Augmented Generation?"
    embedding = embed_query(embedder, test_text)
    print(f"Embedding dimension: {len(embedding)}")
