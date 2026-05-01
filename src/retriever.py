"""
Retriever Module
================
Students must implement and customize retrieval strategies.
"""

from typing import List, Optional, Dict
from langchain_community.vectorstores import Chroma, FAISS
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_community.retrievers import BM25Retriever
from langchain.openai import ChatOpenAI

def create_vectorstore(
    documents: List[Document],
    embedder,
    db_type: str = "chroma",
    persist_dir: Optional[str] = None
):
    """
    Create a vector store from documents.

    Students MUST modify:
    - Choose between ChromaDB, FAISS, or other vector DBs
    - Add metadata filtering fields
    """
    if db_type == "chroma":
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedder,
            persist_directory=persist_dir
        )
    elif db_type == "faiss":
        vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=embedder
        )
        if persist_dir:
            vectorstore.save_local(persist_dir)
    else:
        raise ValueError(f"Unknown DB type: {db_type}")

    return vectorstore


def get_retriever(
    vectorstore,
    search_type: str = "similarity",
    k: int = 4,
    score_threshold: Optional[float] = None,
    filter_criteria: Optional[Dict] = None
):
    """
   enhanced retriever
    """
    search_kwargs = {"k": k}

    #optional tuning
    if score_threshold is not None:
        search_kwargs["score_threshold"] = score_threshold

    if filter_criteria is not None:
        search_kwargs["filter"] = filter_criteria

    if search_type == "mmr":
        search_kwargs["fetch_k"] = 20
        search_kwargs["lambda_mult"] = 0.5


        if search_type == "similarity_score_threshold":
            if score_threshold is None:
                search_kwargs["score_threshold"] = 0.7 # the de

    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )

    return retriever


def retrieve_with_hybrid_search(
    vectorstore,
    query: str,
    k: int = 4,
    alpha: float = 0.5
) -> List[Document]:
   
   #the vector retriver
   vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
   vectore_docs = vector_retriever.invoke(query)


#the BM25 retriever which is keyword based
   bm25_retriever = BM25Retriever.from_documents(vectorstore.similarity_search("", k=100))
   bm25_retriever.k = k
   bm25_docs = bm25_retriever.invoke(query)

   combined = vectore_docs + bm25_docs

   seen = set()
   unique_docs = []
   for doc in combined:
       if doc.page_content not in seen:
           seen.add(doc.page_content)
           unique_docs.append(doc)

   return unique_docs[:k]   



def retrieve_with_reranking(
    retriever,
    query: str,
    k: int = 4
) -> List[Document]:
    llm = ChatOpenAI(temperature=0)
    compressor = LLMChainExtractor.from_llm(llm)

    compression_retriever = ContextualCompressionRetriever(
        base_retriever=retriever,
        base_compressor=compressor
    )

    results = compression_retriever.invoke(query)

    return results[:k]




if __name__ == "__main__":
    # Basic test
    from embedder import get_embedder
    from loader import load_documents, chunk_documents

    docs = load_documents()
    chunks = chunk_documents(docs)
    embedder = get_embedder()

    vectorstore = create_vectorstore(chunks, embedder)
    filter_criteria = {"source": "HandBook"}
    retriever = get_retriever(vectorstore, 
                              search_type="similarity",
                              k=5,
                              filter_criteria=filter_criteria)

    results = retriever.invoke("What is RAG?")
    print(f"Retrieved {len(results)} documents")
