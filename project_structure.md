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
├── shared/
│   ├── __init__.py
│   ├── chess_environment.py
│   └── quick_test.py
│
├── benchmark_langchain.py
├── requirements.txt
├── PROJECT_STRUCTURE.md
└── PROJECT_SUMMARY.md
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

### Shared Components (`shared/`)
- `chess_environment.py`: board state + transitions + FEN formatting
- `quick_test.py`: lightweight unit + integration test harness

### LangChain Implementation (`langchain/`)
- `actor_langchain.py`: Actor model wrapper + move parsing
- `speculator_langchain.py`: Speculator model wrapper + top‑K parsing
- `speculative_framework_langchain.py`: cache + speculative execution
- `chess_speculation_langchain.py`: LangGraph state machine
- `test_chess_speculation_langchain.py`: tests

## Core Flow (One Turn)
1. Actor constructs API call for current state.
2. Cache lookup by `(handler, state_fen)`.
3. Actor call runs while Speculator predicts top‑K.
4. Pre‑launch Actor calls for predicted next states.
5. Wait for Actor result → validate predictions.
6. Commit correct speculative branch, cancel others.
7. Update state and continue.
# Project Structure & Architecture

## Overview

This project implements **Speculative Actions** - a lossless framework for faster agentic systems, as described in the paper "Speculative Actions: A Lossless Framework for Faster Agentic Systems" (arXiv:2510.04371).

The core idea: Use a fast, cheap model (Speculator) to predict likely actions while a slow, authoritative model (Actor) makes the actual decision. Pre-launch API calls for predicted actions, cache results, and commit/rollback based on validation.

## Project Structure

```
Speculative Reasoning/
├── langchain/                    # LangChain/LangGraph implementation
│   ├── __init__.py
│   ├── actor.py                  # Slow, authoritative agent (GPT-4o)
│   ├── speculator.py             # Fast predictive agent (GPT-4o-mini)
│   ├── speculative_framework.py  # Core speculation logic & caching
│   ├── chess_speculation.py      # Main game loop with LangGraph
│   ├── example_usage.py           # Usage examples
│   └── test_chess_speculation.py  # Unit tests
│
├── ag2/                          # AutoGen 2 (AG2) implementation
│   ├── __init__.py
│   ├── actor_ag2.py              # Slow, authoritative agent (GPT-4o)
│   ├── speculator_ag2.py         # Fast predictive agent (GPT-4o-mini)
│   ├── speculative_framework_ag2.py # Core speculation logic & caching
│   ├── chess_speculation_ag2.py   # Main game loop (direct async)
│   └── example_usage_ag2.py        # Usage examples
│
├── shared/                       # Shared components
│   ├── __init__.py
│   ├── chess_environment.py      # Chess board state management
│   └── quick_test.py             # Quick test script
│
├── requirements.txt              # Dependencies
├── PROJECT_STRUCTURE.md          # This file
└── PROJECT_SUMMARY.md            # Project overview
```

## Architecture

### LangChain/LangGraph Implementation

```
┌─────────────────────────────────────────────────────────────┐
│              ChessSpeculationGame (LangGraph)                │
│                  (StateGraph State Machine)                   │
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
│         ChessEnvironment (State Management)         │
│              (python-chess backend)                 │
└─────────────────────────────────────────────────────┘
```

### AutoGen 2 Implementation

```
┌─────────────────────────────────────────────────────────────┐
│           ChessSpeculationGameAG2 (Direct Async)            │
│                  (Simple while loop)                         │
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
│(AG2)│ │(AG2)│                      │(AG2)│ │(AG2)│
└──┬──┘ └──┬──┘                      └──┬──┘ └──┬──┘
   │       │                            │       │
   └───┬───┘                            └───┬───┘
       │                                      │
┌──────▼──────────────────────────────────────▼──────┐
│         ChessEnvironment (State Management)         │
│              (python-chess backend)                 │
└─────────────────────────────────────────────────────┘
```

## Component Breakdown

### Shared Components (`shared/`)

#### `chess_environment.py` - Game State Management

**Purpose**: Manages chess board state and provides state transitions (used by both implementations).

**Key Classes**:
- `GameState`: Immutable game state representation
  - `board`: python-chess Board object
  - `turn`: Current turn number
  - `current_player`: WHITE or BLACK
  - Methods: `apply_move()`, `is_terminal()`, `get_legal_moves()`, `to_fen()`

- `ChessEnvironment`: Environment wrapper
  - `get_state()`: Get current game state
  - `transition()`: Apply move and return new state
  - `parse_move()`: Parse UCI move strings
  - `format_state_for_prompt()`: Format state for LLM prompts

### LangChain Implementation (`langchain/`)

#### `actor.py` - Authoritative Decision Maker

**Purpose**: Slow, high-quality agent that makes the actual decisions using LangChain.

**Key Class**: `Actor`
- Uses GPT-4o with high reasoning effort
- Uses `ChatOpenAI` from LangChain
- `decide_move(state)`: Makes deliberate move decision
- `construct_api_call(state)`: Creates API call specification
- `_extract_move()`: Parses LLM response to extract move

**Configuration**:
- Model: `gpt-4o` (default)
- Temperature: `0.7`
- Max tokens: `2048`
- Reasoning effort: `high`

#### `speculator.py` - Fast Predictor

**Purpose**: Fast, cheap agent that predicts likely moves using LangChain.

**Key Class**: `Speculator`
- Uses GPT-4o-mini with low reasoning effort
- Uses `ChatOpenAI` from LangChain
- `predict_moves(state, k)`: Predicts top-k most likely moves
- `predict_next_api_calls()`: Predicts API calls for next steps
- `_extract_moves()`: Parses LLM response to extract moves

**Configuration**:
- Model: `gpt-4o-mini` (default)
- Temperature: `0.3` (lower for consistency)
- Max tokens: `512` (fewer for speed)
- Reasoning effort: `low`

#### `speculative_framework.py` - Core Framework

**Purpose**: Implements the speculative actions algorithm with caching and parallel execution.

**Key Classes**:

- `PendingAction`: Represents an in-flight API call
  - Fields: `handler`, `params`, `future`, `is_speculative`

- `SpeculativeCache`: Caches pending API calls by (handler, FEN) key
  - `get()`: Retrieve cached pending action
  - `put()`: Cache a pending action
  - **Key insight**: Caches futures, not results - enables parallel execution

- `SpeculativeFramework`: Main framework orchestrating Actor + Speculator
  - `execute_api_call()`: Execute API call (cached or new)
  - `pre_launch_speculative_call()`: Launch speculative call without awaiting
  - `validate_prediction()`: Check if prediction matches actual
  - `cancel_speculative_actions()`: Cancel incorrect speculative branches

#### `chess_speculation.py` - Game Loop

**Purpose**: Orchestrates the full game using LangGraph state machine.

**Key Classes**:

- `GameStateDict`: LangGraph state dictionary (TypedDict)
  - Tracks: game state, current player, turn number, moves history, statistics

- `ChessSpeculationGame`: Main game orchestrator
  - Uses LangGraph `StateGraph` for state machine
  - `_execute_turn()`: Implements Algorithm 1 per turn
  - `play_game()`: Runs full game and returns statistics

**State Machine**:
```
START → white_turn → check_game_over → [continue_white | continue_black | end]
                                    ↓
                              black_turn → check_game_over → ...
```

### AutoGen 2 Implementation (`ag2/`)

#### `actor_ag2.py` - Authoritative Decision Maker

**Purpose**: Slow, high-quality agent using AutoGen 2's `ConversableAgent`.

**Key Class**: `ActorAgentAG2`
- Uses GPT-4o with high reasoning effort
- Uses `ConversableAgent` from AutoGen 2
- Agent-based conversation model (not direct API calls)
- Maintains conversation history
- `decide_move(state)`: Makes deliberate move decision using `initiate_chat()`

#### `speculator_ag2.py` - Fast Predictor

**Purpose**: Fast, cheap agent using AutoGen 2's `ConversableAgent`.

**Key Class**: `SpeculatorAgentAG2`
- Uses GPT-4o-mini with low reasoning effort
- Uses `ConversableAgent` from AutoGen 2
- Agent-based conversation model
- `predict_moves(state, k)`: Predicts top-k moves using `initiate_chat()`

#### `speculative_framework_ag2.py` - Core Framework

**Purpose**: Same as LangChain version, but adapted for AutoGen 2 agents.

**Key Classes**:
- `PendingActionAG2`: Similar to `PendingAction`, but stores AutoGen chat results
- `SpeculativeCacheAG2`: Same caching strategy (FEN-based)
- `SpeculativeFrameworkAG2`: Same algorithm, works with AutoGen agents

#### `chess_speculation_ag2.py` - Game Loop

**Purpose**: Orchestrates the full game using direct async loop (no LangGraph).

**Key Class**: `ChessSpeculationGameAG2`
- Uses simple `while` loop instead of state machine
- Dictionary-based state (not TypedDict)
- Same `_execute_turn()` algorithm
- `play_game()`: Runs full game and returns statistics

## Algorithm Flow (Both Implementations)

**Turn Execution Flow** (`_execute_turn`):
1. **Policy Decision**: Actor constructs API call for current state
2. **Cache Check**: If cached, use cached result (cache hit!)
3. **Parallel Launch**:
   - Launch Actor API call (slow, authoritative)
   - Launch Speculator predictions (fast, parallel)
4. **Speculative Pre-launch**: For each predicted move, pre-launch next API calls
5. **Wait & Validate**: Wait for Actor, validate predictions
6. **Commit/Rollback**: Keep correct predictions, cancel incorrect ones
7. **State Update**: Apply move, advance turn

## Data Flow

### Single Turn Flow

```
Current State (FEN)
    ↓
[Policy: Actor constructs API call]
    ↓
[Cache Check]
    ├─ Cache Hit → Use cached result → Apply move → Next turn
    └─ Cache Miss
        ↓
[Parallel Execution]
    ├─ Actor: decide_move(state) [slow, 2-5s]
    └─ Speculator: predict_moves(state, k=3) [fast, 0.5-1s]
        ↓
[Speculative Pre-launch]
    For each predicted move:
        - Compute next_state = apply_move(predicted_move)
        - Pre-launch: decide_move(next_state)
        - Cache speculative future
    ↓
[Wait for Actor]
    ↓
[Validation]
    Compare Actor move with predictions
    ↓
[Commit/Rollback]
    ├─ Prediction Hit: Keep cached speculative call
    └─ Prediction Miss: Cancel speculative calls
    ↓
[Apply Move & Update State]
    ↓
Next Turn
```

### Cache Hit Flow (When Prediction Was Correct)

```
Current State
    ↓
[Cache Check] → Cache Hit! (from previous speculation)
    ↓
[Await Cached Future]
    ├─ Already done → Instant result (saved ~3s)
    └─ Still running → Wait less time (saved ~1-2s)
    ↓
[Apply Move]
    ↓
Next Turn (no API call needed!)
```

## Key Design Decisions

### 1. **Caching Futures, Not Results**
- Cache stores `PendingAction` (with `asyncio.Future`)
- Enables parallel execution: multiple consumers can await the same future
- If speculative call completes before needed, instant cache hit

### 2. **Lossless Guarantee**
- Final trajectory identical to sequential Actor execution
- Only difference: some API calls happen in parallel
- If prediction wrong, speculative calls are cancelled **after Actor finishes**

### 3. **Two-Model Architecture**
- Actor: Slow, high-quality (GPT-4o)
- Speculator: Fast, cheaper (GPT-4o-mini)
- Cost-benefit: Speculator predictions are cheap, Actor calls are expensive

### 4. **Dual Implementation**
- **LangChain/LangGraph**: Declarative state machine, type-safe, structured
- **AutoGen 2**: Imperative loop, agent-based conversations, flexible
- Both achieve identical results with same cancellation behavior

## Performance Characteristics

### Time Savings
- **Cache hits**: Save full API call time (~3 seconds)
- **Parallel execution**: Speculator runs while Actor runs (overlap time)
- **Typical savings**: 15-25% reduction in end-to-end latency

### Prediction Accuracy
- **Typical accuracy**: 30-55% (depends on position complexity)
- **Higher k**: More predictions = higher chance of hit, but more wasted compute
- **Sweet spot**: k=3-5 for chess

### Cost Analysis
- **Actor calls**: Expensive (GPT-4o, ~$0.01-0.03 per call)
- **Speculator calls**: Cheap (GPT-4o-mini, ~$0.0001-0.001 per call)
- **Net cost**: Slightly higher due to speculative calls, but much faster

## Usage

### LangChain Implementation

```bash
# Run game
python -m langchain.chess_speculation

# Run examples
python -m langchain.example_usage

# Run tests
pytest langchain/test_chess_speculation.py -v
```

### AutoGen 2 Implementation

```bash
# Run game
python -m ag2.chess_speculation_ag2

# Run examples
python -m ag2.example_usage_ag2
```

## Extensibility

The framework can be extended to:
1. **Multi-step speculation**: Predict N moves ahead (not just 1)
2. **Adaptive k**: Dynamically adjust number of predictions based on confidence
3. **Different environments**: Not just chess - any sequential decision-making task
4. **Uncertainty-aware**: Use prediction confidence to prioritize speculative calls
5. **Batch processing**: Process multiple games in parallel

## References

- Paper: "Speculative Actions: A Lossless Framework for Faster Agentic Systems" (arXiv:2510.04371)
- LangGraph: https://github.com/langchain-ai/langgraph
- AutoGen 2: https://github.com/microsoft/autogen
- Python-Chess: https://python-chess.readthedocs.io/
