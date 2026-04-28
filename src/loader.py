"""
Document Loader and Chunker
============================
UNIMA University Handbook Assistant - Group 11
COM325 - RAG Implementation
 
This module handles loading and chunking of UNIMA policy
documents and handbooks from the data/ directory.
Supports both PDF and TXT file formats.
"""

from pathlib import Path
from typing import List
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)


def load_documents(data_dir: str = "data/") -> List:
    """
    Load all documents from the data directory.
 
    This function loads both PDF and TXT documents from the
    specified directory. For the UNIMA University Handbook
    Assistant, documents include:
        - UNIMA Code of Ethics and Conduct
        - A Compendium of Rules and Regulations on Student Conduct
        - UNIMA Sexual Harassment Policy
        - Any other university policy documents in PDF or TXT format
 
    Args:
        data_dir: Path to the directory containing documents.
                  Defaults to "data/"
 
    Returns:
        List of loaded LangChain Document objects
    """
    path = Path(data_dir)
    documents = []

    if not path.exists():
        print(f"[WARNING] Data directory '{data_dir}' does not exist.")
        return documents
    

    # Load PDF documents
    txt_files = list(path.glob("*.txt"))
    if txt_files: 
        print(f"[INFO] found {len(txt_files)} TXT files(s)...")
        for file_path in txt_files:
            try:
                loader = TextLoader(str(file_path), encoding="utf-8")
                loaded = loader.load()
                # Add source metadata to each document
                for doc in loaded:
                    doc.metadata["source"] = file_path.name
                    doc.metadata["file_type"] = "txt"
                documents.extend(loaded)
                print(f"  ✔ Loaded: {file_path.name}") 
            except Exception as e:     
                print(f"  ✘ Failed to load {file_path.name}: {e}")
    else:
        print(f"[INFO] No TXT files found.")

    # Load PDF files
    pdf_files = list(path.glob("*.pdf"))
    if pdf_files:
        print(f"[INFO] found {len(pdf_files)} PDF file(s)...")
        for pdf_path in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_path))
                loaded = loader.load()
                # Add source metadata to each page
                for doc in loaded:
                    doc.metadata["source"] = pdf_path.name
                    doc.metadata["file_type"] = "pdf"
                    # page number ia automatically added by PyPDFLoader

                documents.extend(loaded)
                print(f"  ✔ Loaded: {pdf_path.name} "
                      f"({len(loaded)} page(s))") 
            except Exception as e:     
                print(f"  ✘ Failed to load {pdf_path.name}: {e}")

    else:
        print(f"[INFO] No PDF files found.")

    print(f"\n[INFO] Total documents loaded: {len(documents)}")
    return documents


def chunk_documents(
        documents: List,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        chunking_strategy: str = "recursive",
) -> List:
    """
    Split documents into smaller chunks for embedding and retrieval.
 
    For the UNIMA University Handbook Assistant, we use a chunk size
    of 500 characters with an overlap of 100 characters. This ensures
    that policy sections are not cut off abruptly, and that context
    is preserved across chunk boundaries — which is critical when
    answering questions about multi-paragraph university policies.
 
    Args:
        documents: List of loaded LangChain Document objects
        chunk_size: Maximum size of each chunk in characters.
                    Default is 500 — suitable for policy documents.
        chunk_overlap: Number of overlapping characters between
                       consecutive chunks. Default is 100 to preserve
                       context across boundaries.
        chunking_strategy: Strategy to use for chunking.
                           Options: "recursive" (default), "token"
 
    Returns:
        List of chunked LangChain Document objects with metadata
    """
    if not documents:
        print("[WARNING] No documents to chunk.")
        return [] 

    if chunking_strategy == "recursive":
        # RecursiveCharacterTextSplitter is the best choice for
        # university policy documents because it respects natural
        # text boundaries like paragraphs, sentences, and words
        # before falling back to character splitting.
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
    elif chunking_strategy == "token":
        # Token-based splitting is useful when working with
        # LLMs that have strict token limits
        splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    else:
        raise ValueError(
            f"Unknown chunking strategy: '{chunking_strategy}'. "
            f"Choose 'recursive' or 'token'."
        )

    print(f"[INFO] Chunking {len(documents)} document(s) using "
          f"'{chunking_strategy}' strategy...")
    print(f"[INFO] Chunk size: {chunk_size} | Overlap: {chunk_overlap}")

    chunks = splitter.split_documents(documents)

    # Add custom metadata to each chunk
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)
        # preserve source from original document metadata
        if "source" not in chunk.metadata:
            chunk.metadata["source"] = "unknown"

    print(f"[INFO] Total chunks created: {len(chunks)}")
    return chunks

def preview_chunks(chunks: List, num_chunks: int = 3) -> None:
    """
    Preview a sample of chunks to verify loading and chunking.
 
    This is useful during development and testing to confirm
    that the documents have been correctly loaded and split.
 
    Args:
        chunks: List of chunked Document objects
        num_chunks: Number of chunks to preview. Default is 3.
    """ 
    print(f"\n{'='*50}")
    print(f"CHUNK PREVIEW - showing {num_chunks} of {len(chunks)} chunks")
    print(f"{'='*50}")

    for i, chunk in enumerate(chunks[:num_chunks]):
        print(f"\n--- Chunk {i + 1} ---")
        print(f"Source  : {chunk.metadata.get('source', 'N/A')}")
        print(f"Chunk ID: {chunk.metadata.get('chunk_id', 'N/A')}")
        print(f"Size    : {chunk.metadata.get('chunk_size', 'N/A')} chars")
        if "page" in chunk.metadata:
            print(f"Page    : {chunk.metadata.get('page', 'N/A')}")
        print(f"Content : {chunk.page_content[:200]}...") # Showing the first 200 characters
    print(f"\n{'='*50}\n")


if __name__ == "__main__":
    #  Quick test of the loader
    print("Testing UNIMA Handbook Document Loader...\n")

    # Load documents from the data directory
    docs = load_documents("data/")

    if docs:
        # chunk the documents
        chunks = chunk_documents(
            docs,
            chunk_size=500,
            chunk_overlap=100,
            chunking_strategy="recursive"
        )

        # Preview sample chunks
        preview_chunks(chunks, num_chunks=3)

        print(f"✔ Loader test complete!")
        print(f" Documents loaded : {len(docs)}")
    else:
        print("✘ No documents found in data/ directory.")
        print(" Make sure your UNIMA PDF files are in the data/ folder.")


        