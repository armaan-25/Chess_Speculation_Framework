# Project Structure & Architecture (LangChain/LangGraph)

## Overview
This project implements **Speculative Actions** in a chess environment using **LangChain** for LLM calls and **LangGraph** for workflow orchestration.

## Project Structure
```
Speculative Reasoning/
├── langchain/
│   ├── __init__.py
│   ├── actor_langchain.py              # Actor (GPT-4o)
│   ├── speculator_langchain.py         # Speculator (GPT-4o-mini)
│   ├── speculative_framework_langchain.py
│   ├── chess_speculation_langchain.py  # LangGraph game loop
│   ├── example_usage_langchain.py
│   └── test_chess_speculation_langchain.py
│
├── core/
│   ├── __init__.py
│   ├── chess_environment.py
│   └── quick_test.py
│
├── tools/
│   └── benchmark_langchain.py
│
├── requirements.txt
├── PROJECT_STRUCTURE_LANGCHAIN.md
└── project_summary.md
```

## Architecture (LangChain/LangGraph)
```
┌─────────────────────────────────────────────────────────────┐
│              ChessSpeculationGame (LangGraph)               │
│                  (StateGraph State Machine)                 │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
┌───────▼────────┐                    ┌────────▼────────┐
│  Framework     │                    │   Framework     │
│   (White)      │                    │    (Black)      │
└───────┬────────┘                    └────────┬────────┘
        │                                       │
   ┌────┴────┐                            ┌────┴────┐
   │        │                            │        │
┌──▼──┐ ┌──▼──┐                      ┌──▼──┐ ┌──▼──┐
│Actor│ │Spec │                      │Actor│ │Spec │
│(LC) │ │(LC) │                      │(LC) │ │(LC) │
└──┬──┘ └──┬──┘                      └──┬──┘ └──┬──┘
   │       │                            │       │
   └───┬───┘                            └───┬───┘
       │                                      │
┌──────▼──────────────────────────────────────▼──────┐
│         ChessEnvironment (python-chess backend)    │
└─────────────────────────────────────────────────────┘
```

## Component Breakdown

### Shared Components (`core/`)
- `chess_environment.py`: board state + transitions + FEN formatting
- `quick_test.py`: lightweight unit + integration test harness

### LangChain Implementation (`langchain/`)
- `actor_langchain.py`: Actor model wrapper + move parsing
- `speculator_langchain.py`: Speculator model wrapper + top‑K parsing
- `speculative_framework_langchain.py`: cache + speculative execution
- `chess_speculation_langchain.py`: LangGraph state machine
- `test_chess_speculation_langchain.py`: tests

### Benchmarking (`tools/`)
- `benchmark_langchain.py`: latency, cache, overlap, and cost metrics

## Core Flow (One Turn)
1. Actor constructs API call for current state.
2. Cache lookup by `(handler, state_fen)`.
3. Actor call runs while Speculator predicts top‑K.
4. Pre‑launch Actor calls for predicted next states.
5. Wait for Actor result → validate predictions.
6. Commit correct speculative branch, cancel others.
7. Update state and continue.
