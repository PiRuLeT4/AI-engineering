# Retrieval-Augmented Generation (RAG)

> Hands-on RAG pipelines in Python, moving from a pure-Python first-principles implementation to a persistent vector database backed by OpenAI.

This module explores how Retrieval-Augmented Generation works under the hood before introducing production tooling. It is split into two implementations: a zero-framework pipeline built from scratch and a real end-to-end setup using ChromaDB and the OpenAI API.

## Structure

- [`from_scratch`](from_scratch) - Pure-Python RAG pipeline implementing document loading, overlapping chunking, TF-IDF embeddings, cosine similarity search, and extractive generation.
- [`real_rag`](real_rag) - End-to-end RAG pipeline using OpenAI embeddings (`text-embedding-3-small`), persistent ChromaDB storage, source metadata, and grounded LLM answers over sample documents in [`real_rag/docs`](real_rag/docs).

## Usage

Run either pipeline from the repository root using `uv`:

```bash
uv run python src/R_A_G/from_scratch/pipeline.py
uv run python src/R_A_G/real_rag/pipeline.py
```

`real_rag` requires `OPENAI_API_KEY` in the root `.env` file and persists its local vector index under `real_rag/chroma_db/`.
