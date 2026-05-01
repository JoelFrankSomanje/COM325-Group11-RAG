"""
Document Loader and Chunker
===========================
"""

from pathlib import Path
from typing import List, Optional
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,      # ← UNCOMMENTED
    WebBaseLoader,
    DirectoryLoader
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    TokenTextSplitter
)

def load_documents(data_dir: str = "data/") -> List:
    """
    Load documents from the data directory.
    """
    path = Path(data_dir)
    documents = []

    # Load text files
    for file_path in path.glob("*.txt"):
        loader = TextLoader(str(file_path), encoding="utf-8")
        documents.extend(loader.load())

    # Load PDF files (UNCOMMENTED AND ADDED)
    for pdf_path in path.glob("*.pdf"):
        try:
            loader = PyPDFLoader(str(pdf_path))
            documents.extend(loader.load())
            print(f"✓ Loaded PDF: {pdf_path.name}")
        except Exception as e:
            print(f"✗ Error loading {pdf_path.name}: {e}")

    return documents


def chunk_documents(
    documents: List,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    chunking_strategy: str = "recursive"
) -> List:
    """
    Split documents into chunks.
    """
    if chunking_strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    elif chunking_strategy == "markdown":
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    elif chunking_strategy == "token":
        splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    else:
        raise ValueError(f"Unknown chunking strategy: {chunking_strategy}")

    chunks = splitter.split_documents(documents)

    return chunks


if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)
    print(f"Loaded {len(docs)} documents, created {len(chunks)} chunks")