# Ollama RAG Starter

A local Retrieval-Augmented Generation (RAG) starter project powered by Ollama, LangChain, and Chroma.

## Project Structure

```text
rag-starter/
├── data/                 # Place your documents here
├── notebooks/            # For experimentation
├── src/
│   ├── loader.py         # Document loading and chunking
│   ├── embedder.py       # Ollama embeddings
│   ├── retriever.py      # Chroma retrieval and local reranking
│   ├── generator.py      # Ollama generation and RAG prompt
│   └── pipeline.py       # Main orchestration
├── main.py               # CLI interface
├── requirements.txt
└── README.md
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install and Start Ollama

```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama if it is not already running
ollama serve
```

For Windows, download Ollama from https://ollama.com/download/windows.

### 3. Pull Local Models

You need one embedding model and one chat/generation model:

```bash
ollama pull nomic-embed-text
ollama pull phi3
```

Optional alternatives:

```bash
ollama pull mxbai-embed-large
ollama pull llama3
ollama pull mistral
```

### 4. Add Your Documents

Place documents in the `data/` directory.

Supported file types:
- `.txt`
- `.md`
- `.pdf`

### 5. Run the Pipeline

```bash
# Interactive mode
python3 main.py --mode interactive

# Demo mode
python3 main.py --mode demo
```

## Useful Commands

```bash
# Use a stronger LLM and retrieve more chunks
python3 main.py --llm-model llama3 --k 5 --chunk-size 300

# Use a different Ollama embedding model
python3 main.py --embedder-model mxbai-embed-large --force-rebuild

# Rebuild the Chroma index after changing documents or embedding model
python3 main.py --force-rebuild
```

## What Is Implemented

| File | Implementation |
|------|----------------|
| `src/loader.py` | Loads `.txt`, `.md`, and `.pdf` files, then adds chunk metadata |
| `src/embedder.py` | Uses Ollama embeddings only, with batched document embedding |
| `src/retriever.py` | Uses Chroma, supports similarity search, hybrid search, and local reranking |
| `src/generator.py` | Uses Ollama generation only and returns source documents |
| `src/pipeline.py` | Orchestrates indexing, retrieval, querying, confidence scoring, and basic evaluation |

## Troubleshooting

- **No documents loaded**: Check that `data/` contains `.txt`, `.md`, or `.pdf` files.
- **Ollama connection errors**: Ensure Ollama is running with `ollama serve`.
- **Missing model errors**: Pull the model first, for example `ollama pull phi3` and `ollama pull nomic-embed-text`.
- **Old answers after changing documents**: Run with `--force-rebuild` to recreate the vector index.

## Resources

- Ollama: https://ollama.com
- LangChain RAG: https://github.com/langchain-ai/rag-from-scratch
- Chroma: https://www.trychroma.com/
