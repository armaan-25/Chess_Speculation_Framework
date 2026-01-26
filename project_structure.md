# Project Structure & Architecture (LangChain/LangGraph)

## Overview
This project implements **Speculative Actions** - a lossless framework for faster agentic systems in a chess environment. It uses **LangChain** for LLM calls and **LangGraph** for workflow orchestration, enabling parallel execution of an authoritative Actor model and a fast Speculator model without changing the final outcome.

## Project Structure
```
Speculative Reasoning/
├── langchain/
│   ├── __init__.py
│   ├── actor_langchain.py              # Actor model wrapper (GPT-4o)
│   ├── speculator_langchain.py         # Speculator model wrapper (GPT-4o-mini)
│   ├── speculative_framework_langchain.py  # Core speculation framework
│   └── chess_speculation_langchain.py  # LangGraph state machine & game loop
│
├── core/
│   ├── __init__.py
│   ├── chess_environment.py           # Chess board state & game logic
│   └── quick_test.py                   # Lightweight test utilities
│
├── tools/
│   └── benchmark_langchain.py          # Comprehensive benchmarking tool
│
├── requirements.txt                    # Python dependencies
├── project_structure.md                # This file
├── project_summary.md                  # High-level project overview
└── README.md                           # Getting started guide
```

## Architecture (LangChain/LangGraph)

```
┌─────────────────────────────────────────────────────────────┐
│              ChessSpeculationGame (LangGraph)               │
│                  (StateGraph State Machine)                 │
│                                                             │
│  Manages game state, turn order, and termination           │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
┌───────▼────────┐                    ┌────────▼────────┐
│  Framework     │                    │   Framework     │
│   (White)      │                    │    (Black)      │
│                │                    │                 │
│  - Cache       │                    │  - Cache       │
│  - Actor       │                    │  - Actor       │
│  - Speculator  │                    │  - Speculator  │
└───────┬────────┘                    └────────┬────────┘
        │                                       │
   ┌────┴────┐                            ┌────┴────┐
   │         │                            │         │
┌──▼──┐ ┌──▼──┐                      ┌──▼──┐ ┌──▼──┐
│Actor│ │Spec │                      │Actor│ │Spec │
│GPT-4│ │GPT-4│                      │GPT-4│ │GPT-4│
│  o  │ │mini │                      │  o  │ │mini │
└──┬──┘ └──┬──┘                      └──┬──┘ └──┬──┘
   │       │                            │       │
   └───┬───┘                            └───┬───┘
       │                                      │
┌──────▼──────────────────────────────────────▼──────┐
│         ChessEnvironment (python-chess backend)    │
│                                                     │
│  - Board state management                           │
│  - Move validation & application                   │
│  - FEN string formatting                            │
│  - Terminal state detection                         │
└─────────────────────────────────────────────────────┘
```

## Component Breakdown

### Core Components (`core/`)

#### `chess_environment.py`
- **Purpose**: Chess game state management and transitions
- **Key Classes**:
  - `ChessEnvironment`: Main environment wrapper around python-chess
  - `GameState`: Immutable game state representation
- **Features**:
  - FEN string generation and parsing
  - Legal move generation
  - Move validation and application
  - Terminal state detection (checkmate, stalemate, draw)
  - State formatting for LLM prompts

#### `quick_test.py`
- Lightweight testing utilities for core components
- Unit tests for chess environment functionality

### LangChain Implementation (`langchain/`)

#### `actor_langchain.py`
- **Purpose**: Authoritative Actor model that makes final move decisions
- **Model**: GPT-4o (slow, high-quality reasoning)
- **Key Features**:
  - System prompts optimized for deep chess analysis
  - Move extraction from LLM responses (UCI format)
  - Configurable reasoning effort (high/low)
  - API call construction for speculative framework

#### `speculator_langchain.py`
- **Purpose**: Fast predictive model that guesses likely moves
- **Model**: GPT-4o-mini (fast, cost-effective)
- **Key Features**:
  - Top-K move prediction (configurable k)
  - Optimized prompts for speed
  - Move extraction and validation
  - Next state API call prediction

#### `speculative_framework_langchain.py`
- **Purpose**: Core speculation engine with caching and parallel execution
- **Key Classes**:
  - `SpeculativeFramework`: Main orchestration class
  - `SpeculativeCache`: FEN-keyed cache for in-flight futures
  - `PendingAction`: Represents speculative API calls
- **Features**:
  - Cache lookup by `(handler, state_fen)` tuple
  - Parallel execution of Actor and Speculator
  - Pre-launching speculative Actor calls
  - Validation and rollback of incorrect predictions
  - Cancellation of wasted speculative work

#### `chess_speculation_langchain.py`
- **Purpose**: LangGraph state machine for game orchestration
- **Key Classes**:
  - `ChessSpeculationGame`: Main game controller
  - `GameStateDict`: TypedDict for LangGraph state
- **LangGraph Nodes**:
  - `white_turn`: Execute white's turn with speculation
  - `black_turn`: Execute black's turn with speculation
  - `check_game_over`: Check termination conditions
  - `should_continue`: Routing logic for turn transitions
- **Features**:
  - State machine for turn-based game flow
  - Support for full games or limited moves per player
  - Statistics tracking (accuracy, time saved, predictions)
  - Integration with speculative framework

### Benchmarking (`tools/`)

#### `benchmark_langchain.py`
- **Purpose**: Comprehensive benchmarking and performance analysis
- **Key Features**:
  - Baseline comparison (no speculation vs with speculation)
  - Multiple game modes (full game, limited moves)
  - Token usage tracking (input/output for Actor and Speculator)
  - Latency measurements (end-to-end, per-turn, per-LLM-call)
  - Cache hit rate analysis
  - Wasted compute ratio tracking
  - Visualization generation (pandas/matplotlib)
  - CSV export for further analysis
- **Metrics Tracked**:
  - Speculator accuracy (% predictions matching Actor)
  - End-to-end latency (with baseline comparison)
  - Token cost (Actor + Speculator, with overhead calculation)
  - Cache performance (hit rate, wait times)
  - Parallel overlap analysis
  - Speculative task cancellation rates

## Core Flow (One Turn)

The speculative execution flow for a single turn:

1. **Actor constructs API call** for current game state
   - Creates handler name and parameters
   - Includes FEN string for caching

2. **Cache lookup** by `(handler, state_fen)`
   - If cache hit: return cached future (speculation succeeded!)
   - If cache miss: proceed to parallel execution

3. **Launch Actor API call** (asynchronous task)
   - Slow, authoritative decision (2-5 seconds)
   - Returns move and reasoning

4. **Speculator predicts moves** (in parallel with Actor)
   - Fast prediction of top-K likely moves
   - Returns list of predicted moves

5. **Pre-launch speculative Actor calls** for predicted next states
   - For each predicted move, create next state
   - Launch Actor API call for each predicted state
   - Store as speculative actions

6. **Wait for Actor response** (actual move decision)

7. **Validate predictions** against actual move
   - Check if any predicted move matches actual move
   - Track accuracy metrics

8. **Commit/rollback**:
   - **If prediction hit**: Keep correct speculative action, cancel others
   - **If no hit**: Cancel all speculative actions
   - Calculate time saved from correct predictions

9. **Apply move and update state**
   - Update board state
   - Increment move counts
   - Check termination conditions

10. **Continue to next turn** or end game

## Key Design Decisions

### Why LangGraph?
- **State Machine**: Natural fit for turn-based game flow
- **Type Safety**: TypedDict for state ensures correctness
- **Composability**: Easy to add new nodes/edges
- **Async Support**: Built-in async/await support

### Why FEN-based Caching?
- **State Identity**: FEN uniquely identifies board position
- **Efficiency**: O(1) lookup for identical positions
- **Correctness**: Same position → same move (deterministic)

### Why Parallel Execution?
- **Latency Hiding**: Speculator runs while Actor thinks
- **Cost Efficiency**: Speculator is cheaper than Actor
- **Speedup**: Correct predictions become instant cache hits

### Why Validation + Rollback?
- **Lossless**: Final trajectory matches sequential execution
- **Correctness**: Only commit validated predictions
- **Efficiency**: Cancel wasted speculative work immediately

## Dependencies

- **langgraph**: Workflow orchestration and state machines
- **langchain**: LLM integration and message handling
- **langchain-openai**: OpenAI API integration
- **python-chess**: Chess engine and move validation
- **pandas/matplotlib**: (Optional) Benchmark visualization
