"""
RAG Pipeline Orchestration
===========================
Main pipeline that ties together loading, Ollama embeddings, retrieval, and generation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

try:
    from .embedder import DEFAULT_EMBEDDING_MODEL, get_embedder
    from .generator import DEFAULT_LLM_MODEL, create_qa_chain, create_rag_prompt, generate_response, get_llm
    from .loader import chunk_documents, load_documents
    from .retriever import create_vectorstore, get_retriever, load_vectorstore
except ImportError:
    from embedder import DEFAULT_EMBEDDING_MODEL, get_embedder
    from generator import DEFAULT_LLM_MODEL, create_qa_chain, create_rag_prompt, generate_response, get_llm
    from loader import chunk_documents, load_documents
    from retriever import create_vectorstore, get_retriever, load_vectorstore


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Ollama-powered RAG pipeline.

    The pipeline keeps all model work local through Ollama and stores vectors in
    Chroma so the index can be reused between runs.
    """

    def __init__(
        self,
        data_dir: str = "data/",
        embedder_model: str = DEFAULT_EMBEDDING_MODEL,
        llm_model: str = DEFAULT_LLM_MODEL,
        temperature: float = 0.2,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        retrieval_k: int = 4,
        persist_dir: Optional[str] = "vectorstore",
    ):
        self.data_dir = data_dir
        self.embedder_model = embedder_model
        self.llm_model = llm_model
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retrieval_k = retrieval_k
        self.persist_dir = persist_dir

        logger.info("Initializing Ollama embedder: %s", embedder_model)
        self.embedder = get_embedder(model_name=embedder_model)

        logger.info("Initializing Ollama LLM: %s", llm_model)
        self.llm = get_llm(model_name=llm_model, temperature=temperature)

        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None

    def load_and_index(self, force_rebuild: bool = False):
        """
        Load documents and create or reuse a Chroma vector index.
        """
        persist_path = Path(self.persist_dir) if self.persist_dir else None

        if persist_path and persist_path.exists() and not force_rebuild:
            logger.info("Loading existing Chroma vectorstore from %s", self.persist_dir)
            self.vectorstore = load_vectorstore(self.embedder, self.persist_dir)
        else:
            logger.info("Loading documents from %s", self.data_dir)
            documents = load_documents(self.data_dir)
            if not documents:
                raise RuntimeError(
                    f"No supported documents found in {self.data_dir}. "
                    "Add .txt, .md, or .pdf files and try again."
                )

            chunks = chunk_documents(
                documents,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            logger.info("Created %s chunks from %s documents", len(chunks), len(documents))

            logger.info("Creating Chroma vector index")
            self.vectorstore = create_vectorstore(
                chunks,
                self.embedder,
                persist_dir=self.persist_dir,
            )

        logger.info("Setting up retriever with k=%s", self.retrieval_k)
        self.retriever = get_retriever(self.vectorstore, k=self.retrieval_k)

        logger.info("Creating QA chain")
        prompt = create_rag_prompt()
        self.qa_chain = create_qa_chain(self.llm, self.retriever, prompt)

    def query(self, question: str, return_sources: bool = True) -> Dict:
        """
        Query the RAG pipeline and return an answer plus optional sources.
        """
        if self.qa_chain is None:
            raise RuntimeError("Pipeline not initialized. Call load_and_index() first.")

        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("Question cannot be empty.")

        logger.info("Querying: %s", cleaned_question)
        response = generate_response(self.qa_chain, cleaned_question, return_sources)
        response["confidence"] = self._estimate_confidence(response)
        return response

    def evaluate(self, test_queries: List[Dict]) -> Dict:
        """
        Evaluate the pipeline on simple test queries.

        This lightweight evaluator records whether answers are produced, how many
        sources were returned, and whether an expected answer string appears in
        the generated answer when one is provided.
        """
        results = []
        exact_matches = 0
        answered = 0

        for item in test_queries:
            query = item["question"]
            expected = item.get("expected_answer", "")
            response = self.query(query, return_sources=True)
            answer = response["answer"]
            sources = response.get("sources", [])

            has_answer = bool(answer.strip()) and "do not know" not in answer.lower()
            expected_match = bool(expected) and expected.lower() in answer.lower()

            answered += int(has_answer)
            exact_matches += int(expected_match)

            results.append(
                {
                    "query": query,
                    "expected": expected,
                    "answer": answer,
                    "sources": sources,
                    "source_count": len(sources),
                    "expected_match": expected_match,
                }
            )

        total = len(test_queries)
        return {
            "total_queries": total,
            "answered_rate": answered / total if total else 0.0,
            "expected_match_rate": exact_matches / total if total else 0.0,
            "results": results,
        }

    def _estimate_confidence(self, response: Dict) -> float:
        sources = response.get("sources", [])
        answer = response.get("answer", "")
        if not answer or "do not know" in answer.lower():
            return 0.0
        return min(1.0, len(sources) / max(1, self.retrieval_k))


def main():
    """Main entry point for quick testing."""
    pipeline = RAGPipeline(
        data_dir="data/",
        embedder_model=DEFAULT_EMBEDDING_MODEL,
        llm_model=DEFAULT_LLM_MODEL,
        retrieval_k=3,
    )

    pipeline.load_and_index()

    response = pipeline.query("What is Retrieval-Augmented Generation?")
    print("\nAnswer:", response["answer"])
    if "sources" in response:
        print("\nSources:", response["sources"])


if __name__ == "__main__":
    main()
