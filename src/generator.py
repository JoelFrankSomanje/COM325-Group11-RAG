"""
Ollama LLM Generator Module
===========================
Prompt and generation helpers for local Ollama RAG.
"""

from dataclasses import dataclass
from typing import Dict, Optional

try:
    from langchain_ollama import OllamaLLM as Ollama
except ImportError:
    from langchain_community.llms import Ollama


DEFAULT_LLM_MODEL = "phi3"


@dataclass
class RAGPrompt:
    """Small prompt formatter that avoids version-specific LangChain prompt APIs."""

    template: str
    system_message: str

    def format(self, context: str, question: str) -> str:
        return self.template.format(
            system_message=self.system_message,
            context=context,
            question=question,
        )


@dataclass
class SimpleRAGChain:
    """Minimal retrieval + generation chain compatible with LangChain retrievers."""

    llm: Ollama
    retriever: object
    prompt: RAGPrompt

    def invoke(self, inputs: Dict[str, str]) -> Dict:
        query = inputs.get("query") or inputs.get("question") or ""
        source_documents = self.retriever.invoke(query)
        context = _format_documents(source_documents)
        prompt_text = self.prompt.format(context=context, question=query)
        answer = self.llm.invoke(prompt_text)

        return {
            "result": str(answer).strip(),
            "source_documents": source_documents,
        }


def get_llm(
    model_name: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.2,
    **kwargs,
) -> Ollama:
    """
    Create an Ollama LLM for generation.

    Lower temperatures are better for grounded RAG answers. Good local models:
    - phi3: small and fast
    - llama3: stronger answers if your machine can run it
    - mistral: balanced speed and quality
    """
    return Ollama(model=model_name, temperature=temperature, **kwargs)


def create_rag_prompt(
    system_message: Optional[str] = None,
    template: Optional[str] = None,
) -> RAGPrompt:
    """
    Create a RAG prompt template with citation-friendly instructions.
    """
    if system_message is None:
        system_message = (
            "You are a careful RAG assistant. Answer only from the provided context. "
            "If the context does not contain the answer, say you do not know. "
            "Use concise language and mention the source names when they are available."
        )

    if template is None:
        template = """{system_message}

Context:
{context}

Question: {question}

Answer:"""

    return RAGPrompt(template=template, system_message=system_message)


def create_qa_chain(llm, retriever, prompt: Optional[RAGPrompt] = None) -> SimpleRAGChain:
    """
    Create a simple local RAG chain that returns source documents.
    """
    if prompt is None:
        prompt = create_rag_prompt()

    return SimpleRAGChain(llm=llm, retriever=retriever, prompt=prompt)


def generate_response(
    qa_chain,
    query: str,
    return_sources: bool = True,
) -> Dict:
    """
    Generate a response using the RAG chain.

    Returns a dictionary with an answer and, when requested, compact source data.
    """
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    result = qa_chain.invoke({"query": query})
    response = {"answer": result.get("result", "").strip()}

    if return_sources:
        source_documents = result.get("source_documents", [])
        response["sources"] = [
            {
                "content": doc.page_content[:300].strip(),
                "metadata": doc.metadata,
            }
            for doc in source_documents
        ]

    return response


def _format_documents(documents) -> str:
    """Format retrieved documents into a compact context block."""
    if not documents:
        return "No relevant context was retrieved."

    formatted_docs = []
    for index, doc in enumerate(documents, 1):
        metadata = doc.metadata or {}
        source = metadata.get("file_name") or metadata.get("source") or "Unknown source"
        page = metadata.get("page")
        source_label = f"{source}, page {page}" if page is not None else source
        formatted_docs.append(f"[Source {index}: {source_label}]\n{doc.page_content}")

    return "\n\n".join(formatted_docs)


if __name__ == "__main__":
    prompt = create_rag_prompt()
    print("Default prompt template:")
    print(prompt.template)
