"""
Document Loader and Chunker
===========================
Local document loading and chunking utilities for the RAG pipeline.
"""

from pathlib import Path
from typing import List

try:
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader, TextLoader


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def load_documents(data_dir: str = "data/") -> List:
    """
    Load supported documents from the data directory.

    Supported file types:
    - .txt
    - .md
    - .pdf
    """
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    documents = []

    for file_path in sorted(path.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if file_path.suffix.lower() in {".txt", ".md"}:
            loader = TextLoader(str(file_path), encoding="utf-8")
        elif file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
        else:
            continue

        loaded_docs = loader.load()
        for doc in loaded_docs:
            doc.metadata["source"] = str(file_path)
            doc.metadata["file_name"] = file_path.name
            doc.metadata["file_type"] = file_path.suffix.lower()
        documents.extend(loaded_docs)

    return documents


def chunk_documents(
    documents: List,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    chunking_strategy: str = "recursive",
) -> List:
    """
    Split documents into retrieval-friendly chunks with useful metadata.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    if chunking_strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(documents)
    elif chunking_strategy == "markdown":
        headers_to_split_on = [
            ("#", "header_1"),
            ("##", "header_2"),
            ("###", "header_3"),
        ]
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        chunks = []
        for document in documents:
            markdown_chunks = splitter.split_text(document.page_content)
            for chunk in markdown_chunks:
                chunk.metadata.update(document.metadata)
            chunks.extend(markdown_chunks)
    else:
        raise ValueError(f"Unknown chunking strategy: {chunking_strategy}")

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index
        chunk.metadata["chunk_size"] = len(chunk.page_content)

    return chunks


if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)
    print(f"Loaded {len(docs)} documents, created {len(chunks)} chunks")
