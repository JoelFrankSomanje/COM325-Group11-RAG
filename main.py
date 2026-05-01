"""
RAG Assistant - CLI Interface
==============================
Interactive CLI for the Ollama-powered RAG pipeline.
"""

import argparse

from src.embedder import DEFAULT_EMBEDDING_MODEL
from src.generator import DEFAULT_LLM_MODEL
from src.pipeline import RAGPipeline


def interactive_mode(pipeline: RAGPipeline):
    """Run the RAG assistant in interactive mode."""
    print("=" * 50)
    print("RAG Assistant - Interactive Mode")
    print("=" * 50)
    print("Type 'quit' or 'exit' to stop\n")

    while True:
        try:
            query = input("You: ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not query:
                continue

            response = pipeline.query(query, return_sources=True)

            print(f"\nAssistant: {response['answer']}")
            print(f"Confidence: {response.get('confidence', 0):.2f}")

            if response.get("sources"):
                print("\nSources:")
                for index, source in enumerate(response["sources"], 1):
                    metadata = source.get("metadata", {})
                    source_name = metadata.get("file_name") or metadata.get("source", "Unknown")
                    print(f"  {index}. {source_name}")
                    print(f"     {source['content'][:120]}...")

            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as error:
            print(f"Error: {error}\n")


def demo_mode(pipeline: RAGPipeline):
    """Run with predefined demo questions."""
    demo_questions = [
        "What is the main topic of the documents?",
        "Summarize the key points.",
        "What are the main conclusions?",
    ]

    print("=" * 50)
    print("RAG Assistant - Demo Mode")
    print("=" * 50)

    for question in demo_questions:
        print(f"\nQ: {question}")
        response = pipeline.query(question)
        print(f"A: {response['answer']}")


def main():
    parser = argparse.ArgumentParser(description="Ollama RAG Assistant CLI")
    parser.add_argument(
        "--mode",
        choices=["interactive", "demo"],
        default="interactive",
        help="Mode to run the assistant",
    )
    parser.add_argument(
        "--data-dir",
        default="data/",
        help="Directory containing .txt, .md, or .pdf documents",
    )
    parser.add_argument(
        "--embedder-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Ollama embedding model name",
    )
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_LLM_MODEL,
        help="Ollama LLM model name",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM temperature; lower is more grounded",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of documents to retrieve",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size for text splitting",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk overlap for text splitting",
    )
    parser.add_argument(
        "--persist-dir",
        default="vectorstore",
        help="Directory for the Chroma vectorstore",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild the vectorstore instead of reusing it",
    )

    args = parser.parse_args()

    print("Initializing Ollama RAG Pipeline...")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Embedder model: {args.embedder_model}")
    print(f"  LLM model: {args.llm_model}")
    print(f"  Retrieval K: {args.k}")
    print(f"  Chunk size: {args.chunk_size}")

    pipeline = RAGPipeline(
        data_dir=args.data_dir,
        embedder_model=args.embedder_model,
        llm_model=args.llm_model,
        temperature=args.temperature,
        retrieval_k=args.k,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        persist_dir=args.persist_dir,
    )

    pipeline.load_and_index(force_rebuild=args.force_rebuild)

    if args.mode == "interactive":
        interactive_mode(pipeline)
    else:
        demo_mode(pipeline)


if __name__ == "__main__":
    main()
