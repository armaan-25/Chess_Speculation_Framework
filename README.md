# Speculative Reasoning (LangChain + LangGraph)

This repo implements speculative actions in a chess environment using LangChain for LLM calls and LangGraph for orchestration.

For architecture details, see `project_structure.md`. For a high-level overview, see `project_summary.md`.

## Prerequisites
- Python 3.10+
- An OpenAI API key

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Create a `.env` file in the repo root:
```
OPENAI_API_KEY=your-key-here
```

## Run the game
```bash
PYTHONPATH=./langchain:./shared python3 -m langchain.chess_speculation_langchain
```

## Run examples
```bash
PYTHONPATH=./langchain:./shared python3 -m langchain.example_usage_langchain
```

## Run tests
```bash
PYTHONPATH=./langchain:./shared python3 -m pytest langchain/test_chess_speculation_langchain.py -v
```

## Benchmark
Quick run:
```bash
python3 benchmark_langchain.py --quick
```

Custom:
```bash
python3 benchmark_langchain.py --k 2,3 --runs 2 --max-turns 5
```
