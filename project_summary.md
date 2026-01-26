# Project Summary

## What This Project Does

This project implements **Speculative Actions** - a lossless framework for faster agentic systems, based on the paper "Speculative Actions: A Lossless Framework for Faster Agentic Systems" (arXiv:2510.04371).

**Core Concept**: Use two AI models in parallel without changing the final outcome:
- **Actor** (slow, authoritative): decides the real move using GPT-4o with deep reasoning.
- **Speculator** (fast, cheap): predicts likely moves using GPT-4o-mini for speed.

**Why it's faster**: While the Actor thinks (2–5 seconds), the Speculator predicts and pre-launches future Actor calls. Correct predictions become instant cache hits; incorrect ones are canceled **after the Actor finishes**, ensuring the final trajectory matches sequential execution.

## Key Features

### Lossless Speculation
- **Guarantee**: Final game trajectory matches sequential play exactly
- **Validation**: Only commit predictions that match Actor's actual decisions
- **Rollback**: Cancel incorrect speculative work immediately

### Parallel Execution
- **Actor + Speculator**: Run concurrently to hide latency
- **Pre-launching**: Start Actor calls for predicted states before they're needed
- **Overlap**: Speculator typically finishes before Actor, enabling early speculative calls

### Intelligent Caching
- **FEN-keyed**: Cache futures by `(handler, state_fen)` tuple
- **In-flight tracking**: Store pending speculative actions
- **Cache hits**: Correct predictions become instant (0ms wait time)

### Chess Integration
- **Full legality**: All moves validated via python-chess
- **State management**: Immutable game states with FEN representation
- **Terminal detection**: Checkmate, stalemate, and draw detection

## Architecture Overview

```
Chess Game Loop (LangGraph State Machine)
    ↓
White/Black turns (strict alternating order)
    ↓
Per Turn: Speculative Execution
    ├── Actor (authoritative, GPT-4o)
    │   └── Deep analysis → final move decision
    │
    ├── Speculator (predictive, GPT-4o-mini)
    │   └── Fast prediction → top-K likely moves
    │
    └── Speculative Framework
        ├── Cache lookup (FEN-keyed)
        ├── Parallel execution (Actor + Speculator)
        ├── Pre-launch speculative calls
        ├── Validate predictions
        └── Commit correct / Cancel incorrect
```

## Key Components

### Core (`core/`)
- **`chess_environment.py`**: Chess board state, move validation, FEN formatting
- **`quick_test.py`**: Lightweight testing utilities

### LangChain Implementation (`langchain/`)
- **`actor_langchain.py`**: Actor model wrapper with move parsing
  - GPT-4o with configurable reasoning effort
  - System prompts for deep chess analysis
  - UCI format move extraction
  
- **`speculator_langchain.py`**: Speculator model wrapper with top-K parsing
  - GPT-4o-mini for fast, cost-effective predictions
  - Optimized prompts for speed
  - Multiple move prediction and extraction
  
- **`speculative_framework_langchain.py`**: Core speculation engine
  - `SpeculativeCache`: FEN-keyed cache for in-flight futures
  - `SpeculativeFramework`: Orchestrates parallel execution
  - `PendingAction`: Represents speculative API calls
  - Validation and rollback logic
  
- **`chess_speculation_langchain.py`**: LangGraph state machine
  - `ChessSpeculationGame`: Main game controller
  - LangGraph nodes: `white_turn`, `black_turn`, `check_game_over`
  - Turn-based state machine with termination detection
  - Statistics tracking (accuracy, time saved, predictions)

### Benchmarking (`tools/`)
- **`benchmark_langchain.py`**: Comprehensive performance analysis
  - Baseline comparison (no speculation vs with speculation)
  - Token usage tracking (input/output for both models)
  - Latency measurements (end-to-end, per-turn, per-call)
  - Cache performance analysis
  - Visualization generation (pandas/matplotlib)
  - CSV export for further analysis

## How It Works

### Single Turn Execution

1. **Actor constructs API call** for current state
2. **Cache lookup** - check if this state was predicted
3. **If cache miss**: Launch Actor call (async)
4. **In parallel**: Speculator predicts top-K moves
5. **Pre-launch**: Start Actor calls for predicted next states
6. **Wait**: Actor returns actual move
7. **Validate**: Check if any prediction matched
8. **Commit/Rollback**: Keep correct, cancel incorrect
9. **Apply move**: Update game state
10. **Continue**: Next turn or end game

### Performance Characteristics

**Best Case** (high accuracy):
- Speculator predicts correctly → cache hit → instant move
- Time saved: ~2-5 seconds per correct prediction
- Token overhead: Speculator tokens (~1-2K per turn)

**Worst Case** (low accuracy):
- No predictions match → all speculative work wasted
- Time overhead: Minimal (speculator is fast)
- Token overhead: Speculator + wasted speculative Actor tokens

**Typical Case** (30-50% accuracy):
- Some predictions hit → partial time savings
- Net benefit depends on accuracy vs token cost tradeoff

## Benchmarking

The project includes comprehensive benchmarking to measure:
- **Speculator Accuracy**: % of predictions matching Actor
- **End-to-End Latency**: Total time with/without speculation
- **Token Cost**: Actor + Speculator tokens vs baseline
- **Cache Performance**: Hit rates and wait times
- **Wasted Compute**: Ratio of canceled speculative tasks

See `README.md` for detailed benchmarking instructions.

## Technical Stack

- **LangGraph**: Workflow orchestration and state machines
- **LangChain**: LLM integration and message handling
- **LangChain OpenAI**: GPT-4o and GPT-4o-mini integration
- **python-chess**: Chess engine and move validation
- **pandas/matplotlib**: (Optional) Benchmark visualization

## Key Design Principles

1. **Lossless**: Never change the final outcome
2. **Parallel**: Maximize concurrent execution
3. **Efficient**: Minimize wasted compute
4. **Measurable**: Comprehensive benchmarking
5. **Extensible**: Easy to add new models or games

## Use Cases

- **Research**: Study speculative execution in agentic systems
- **Optimization**: Reduce latency in chess-playing agents
- **Benchmarking**: Compare different speculation strategies
- **Education**: Learn about parallel AI execution patterns
