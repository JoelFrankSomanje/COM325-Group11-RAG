"""
Retriever Module
================
Chroma-based retrieval strategies for the Ollama RAG pipeline.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from langchain_community.vectorstores import Chroma


_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")


def create_vectorstore(
    documents: List[Document],
    embedder,
    persist_dir: Optional[str] = None,
) -> Chroma:
    """
    Create a Chroma vector store from documents using Ollama embeddings.
    """
    if not documents:
        raise ValueError("No documents were provided to index.")

    return Chroma.from_documents(
        documents=documents,
        embedding=embedder,
        persist_directory=persist_dir,
    )


def load_vectorstore(embedder, persist_dir: str) -> Chroma:
    """Load an existing Chroma vector store from disk."""
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embedder,
    )


def get_retriever(
    vectorstore: Chroma,
    search_type: str = "similarity",
    k: int = 4,
    score_threshold: Optional[float] = None,
    filter_criteria: Optional[Dict] = None,
):
    """
    Create a retriever with configurable Chroma search parameters.
    """
    if k < 1:
        raise ValueError("k must be at least 1.")

    search_kwargs = {"k": k}

    if score_threshold is not None:
        search_type = "similarity_score_threshold"
        search_kwargs["score_threshold"] = score_threshold

    if filter_criteria is not None:
        search_kwargs["filter"] = filter_criteria

    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )


def retrieve_with_hybrid_search(
    vectorstore: Chroma,
    query: str,
    k: int = 4,
    alpha: float = 0.7,
) -> List[Document]:
    """
    Retrieve documents with a lightweight hybrid search.

    The score combines Chroma vector relevance with a local lexical-overlap score.
    alpha controls the vector contribution; 1.0 is pure vector search and 0.0 is
    pure lexical matching.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    vector_hits = vectorstore.similarity_search_with_relevance_scores(query, k=max(k * 4, k))
    vector_scores = {_document_key(doc): score for doc, score in vector_hits}

    candidates = _get_all_documents(vectorstore)
    if not candidates:
        return []

    lexical_scores = {
        _document_key(doc): _lexical_overlap_score(query, doc.page_content)
        for doc in candidates
    }

    candidate_map = {_document_key(doc): doc for doc in candidates}
    for doc, _score in vector_hits:
        candidate_map.setdefault(_document_key(doc), doc)

    ranked: List[Tuple[float, Document]] = []
    for key, doc in candidate_map.items():
        vector_score = vector_scores.get(key, 0.0)
        lexical_score = lexical_scores.get(key, 0.0)
        combined_score = (alpha * vector_score) + ((1 - alpha) * lexical_score)
        ranked.append((combined_score, doc))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [doc for _score, doc in ranked[:k]]


def retrieve_with_reranking(
    retriever,
    query: str,
    k: int = 4,
) -> List[Document]:
    """
    Retrieve extra candidates and rerank them by lexical relevance.

    This keeps the project local and dependency-light while still improving the
    order for queries where exact terms matter.
    """
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    docs = retriever.invoke(query)
    ranked = sorted(
        docs,
        key=lambda doc: _lexical_overlap_score(query, doc.page_content),
        reverse=True,
    )
    return ranked[:k]


def _get_all_documents(vectorstore: Chroma) -> List[Document]:
    """Read stored documents back from Chroma for local lexical scoring."""
    data = vectorstore.get(include=["documents", "metadatas"])
    texts = data.get("documents", []) or []
    metadatas = data.get("metadatas", []) or []

    return [
        Document(page_content=text, metadata=metadata or {})
        for text, metadata in zip(texts, metadatas)
        if text
    ]


def _lexical_overlap_score(query: str, text: str) -> float:
    query_terms = set(_tokenize(query))
    if not query_terms:
        return 0.0

    text_terms = set(_tokenize(text))
    if not text_terms:
        return 0.0

    return len(query_terms & text_terms) / len(query_terms)


def _tokenize(text: str) -> List[str]:
    return [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text)]


def _document_key(document: Document) -> Tuple[str, str, int]:
    metadata = document.metadata or {}
    return (
        str(metadata.get("source", "")),
        str(metadata.get("page", "")),
        int(metadata.get("chunk_id", -1)),
    )


if __name__ == "__main__":
    try:
        from .embedder import get_embedder
        from .loader import chunk_documents, load_documents
    except ImportError:
        from embedder import get_embedder
        from loader import chunk_documents, load_documents

    docs = load_documents()
    chunks = chunk_documents(docs)
    embedder = get_embedder()

    vectorstore = create_vectorstore(chunks, embedder)
    retriever = get_retriever(vectorstore, k=3)

    results = retriever.invoke("What is RAG?")
    print(f"Retrieved {len(results)} documents")
