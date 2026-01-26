# Project Summary

## What This Project Does

This project implements **Speculative Actions** - a lossless framework for faster agentic systems, based on the paper "Speculative Actions: A Lossless Framework for Faster Agentic Systems" (arXiv:2510.04371).

**Core Concept**: Use two AI models in parallel without changing the final outcome:
- **Actor** (slow, authoritative): decides the real move using GPT-4o.
- **Speculator** (fast, cheap): predicts likely moves using GPT-4o-mini.

**Why it’s faster**: While the Actor thinks (2–5 seconds), the Speculator predicts and pre-launches future Actor calls. Correct predictions become instant cache hits; incorrect ones are canceled **after the Actor finishes**.

## Key Features
- **Lossless speculation**: final trajectory matches sequential play.
- **Parallel execution**: Actor + Speculator run concurrently.
- **Intelligent caching**: FEN‑keyed cache of in‑flight futures.
- **Validation + rollback**: commit correct branch, cancel the rest.
- **Chess integration**: full legality checks via python‑chess.

## Core Architecture
```
Chess Game Loop
    ↓
White/Black turns (strict order)
    ↓
Per turn:
    Actor (authoritative) + Speculator (predictive)
    ↓
Speculative Framework:
    cache → parallel calls → validate → commit/cancel
```

## Key Files (LangChain/LangGraph)
- `langchain/actor_langchain.py`: Actor model + move parsing.
- `langchain/speculator_langchain.py`: Speculator model + top‑K parsing.
- `langchain/speculative_framework_langchain.py`: cache + speculation orchestration.
- `langchain/chess_speculation_langchain.py`: LangGraph state machine.
- `langchain/test_chess_speculation_langchain.py`: tests.
- `benchmark_langchain.py`: benchmark harness.

## What to Know
- **LangGraph** handles workflow (nodes, edges, routing).
- **LangChain** handles LLM calls (messages → API → parse).
- **Caching** stores futures keyed by `(handler, state_fen)`.
