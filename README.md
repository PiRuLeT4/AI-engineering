# AI Engineering

> A personal sandbox for learning and practicing applied Artificial Intelligence concepts with Python.

This repository documents my self-taught journey into AI engineering. Each folder under `src/` explores a specific topic—such as prompt engineering, evaluation harnesses, and retrieval-augmented generation (RAG)—through hands-on experiments and its own dedicated notes.

## Structure

- [`src/prompt_engineering`](src/prompt_engineering) — Prompt patterns, injection tests, and multi-model comparisons.
- [`src/R_A_G`](src/R_A_G) — Retrieval-Augmented Generation experiments.

## Setup

Built with Python `3.12+` and managed with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

Add your model provider API keys to a `.env` file in the root directory before running the scripts.
