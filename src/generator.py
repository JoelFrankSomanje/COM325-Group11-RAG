"""
LLM GENERATOR MODULE
Scenario: University Academic Policy Q&A

Modifications & Justifications:
1. Custom prompt template tailored for academic regulation answers
   - Instructs LLM to cite sections (e.g., Section 3.4)
   - Tells LLM to say not found if context is insufficient
   - Prevents hallucination of fake university policies

2. Temperature set to 0.1 (from 0.7):
   - Lower temperature = more factual, deterministic answers
   - Critical for policy Q&A where accuracy matters more than creativity

3. chain_type set to stuff:
   - Appropriate for our chunk sizes
   - Faster than map_reduce for small document sets

4. return_source_documents=True:
   - Shows students which document the answer came from
"""

from typing import List, Optional, Dict
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


def get_llm(
    provider: str = "ollama",
    model_name: str = "phi3",
    temperature: float = 0.1,
    **kwargs
):
    """
    Get an LLM for generation.

    Justification for temperature=0.1:
    Academic policy answers must be precise and consistent.
    High temperature risks the model inventing rules that do not exist.
    """
    if provider == "ollama":
        return Ollama(model=model_name, temperature=temperature)
    elif provider == "openai":
        return ChatOpenAI(model=model_name, temperature=temperature, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def create_rag_prompt(
    system_message: Optional[str] = None,
    template: Optional[str] = None
) -> PromptTemplate:
    """
    Custom prompt for University Academic Policy Q&A.

    Justification:
    - Default scaffold prompt is too generic
    - University assistant must cite specific sections
    - Must refuse to guess when information is not available
    """
    if template is None:
        template = """You are an academic advisor assistant for the University of Malawi.
Use ONLY the context below to answer the student's question.
Always cite the relevant section (e.g., According to Section 3.4...).
If the answer is not found in the context, respond with:
I could not find this information in the university regulations.
Please contact the Registrars Office.
Do NOT make up rules or policies.

Context:
{context}

Student Question: {question}

Answer:"""

    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )


def create_qa_chain(llm, retriever, prompt: Optional[PromptTemplate] = None):
    """
    Create RetrievalQA chain.

    Justification for chain_type=stuff:
    - Chunks are 600 characters each, k=4 chunks retrieved
    - Total context fits in phi3 context window
    - Faster than map_reduce for small document sets
    """
    if prompt is None:
        prompt = create_rag_prompt()

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )


def generate_response(
    qa_chain,
    query: str,
    return_sources: bool = True
) -> Dict:
    """
    Generate a response and include source citations.

    Modification from scaffold:
    - Added source document formatting for clear citation
    - Students can see which document answered their question
    """
    result = qa_chain.invoke({"query": query})

    response = {"answer": result["result"]}

    if return_sources and "source_documents" in result:
        response["sources"] = [
            {
                "content": doc.page_content[:200] + "...",
                "metadata": doc.metadata
            }
            for doc in result["source_documents"]
        ]

    return response


if __name__ == "__main__":
    prompt = create_rag_prompt()
    print("Custom prompt template created successfully")
    print(prompt.template)