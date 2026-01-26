# Speculative Reasoning (LangChain + LangGraph)

This repository implements **Speculative Actions** - a lossless framework for faster agentic systems in a chess environment. It uses LangChain for LLM calls and LangGraph for workflow orchestration, enabling parallel execution of an authoritative Actor model (GPT-4o) and a fast Speculator model (GPT-4o-mini) without changing the final outcome.

For detailed architecture, see [`project_structure.md`](project_structure.md). For a high-level overview, see [`project_summary.md`](project_summary.md).

## Prerequisites

- Python 3.10+
- OpenAI API key

## Setup

1. **Create virtual environment**:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. **Install dependencies**:
```bash
python3 -m pip install -r requirements.txt
```

3. **Set up API key**:
Create a `.env` file in the repo root:
```
OPENAI_API_KEY=your-key-here
```

## Quick Start: Run a Game

```bash
PYTHONPATH=./langchain:./core python3 -m langchain.chess_speculation_langchain
```

This runs a single game with default settings (k=3, max 30 turns).

## Benchmarking

The benchmark tool is the primary way to evaluate speculative execution performance. It compares baseline execution (Actor only, no speculation) against speculative execution (Actor + Speculator) across multiple metrics.

### Key Metrics

The benchmark measures three critical metrics:

1. **Speculator Accuracy**: Percentage of predictions that match the Actor's actual moves
2. **End-to-End Latency**: Total execution time (with baseline comparison)
3. **Token Cost**: Total tokens used by Actor and Speculator (with overhead calculation)

### Basic Usage

**Quick test** (1 move per player, fastest):
```bash
python3 tools/benchmark_langchain.py --quick
```

**Limited moves mode** (recommended for testing):
```bash
python3 tools/benchmark_langchain.py --moves-per-player 3 --k 2 --runs 1
```

**Full game mode** (until checkmate/stalemate):
```bash
python3 tools/benchmark_langchain.py --full-game --k 2,3 --runs 2
```

### Game Modes

#### Limited Moves Mode (Default)
Plays until each player makes N moves. Good for controlled testing.

```bash
python3 tools/benchmark_langchain.py --moves-per-player 5 --k 2,3 --runs 2
```

**When to use**: Testing, development, quick iterations

#### Full Game Mode
Plays until checkmate, stalemate, or draw. More realistic but slower.

```bash
python3 tools/benchmark_langchain.py --full-game --k 2,3 --runs 1
```

**When to use**: Final evaluation, realistic performance measurement

### Benchmark Options

| Option | Description | Default |
|--------|-------------|---------|
| `--k` | Comma-separated k values (number of predictions) | `3` |
| `--runs` | Number of runs per configuration | `1` |
| `--moves-per-player` | Moves per player (limited mode) | `3` |
| `--full-game` | Play until checkmate/stalemate | `False` |
| `--no-baseline` | Skip baseline comparison | `False` |
| `--quick` | Quick test (k=2, runs=1, moves=1) | `False` |
| `--visualize` | Generate visualization plots | `False` |
| `--output-dir` | Directory for outputs | `benchmark_results/` |

### Understanding Results

The benchmark outputs three sections:

#### 1. KEY METRICS
Shows the three critical metrics with baseline comparison:
- **Speculator Accuracy**: How often predictions matched
- **End-to-End Latency**: Total time vs baseline (improvement/overhead)
- **Token Cost**: Total tokens vs baseline (overhead percentage)

#### 2. Summary Table
Quick comparison table with key metrics for each k value.

#### 3. Detailed Metrics
Comprehensive breakdown including:
- Per-component latencies (Actor, Speculator)
- Cache performance (hit rate, wait times)
- Speculative task statistics (launched, canceled, wasted ratio)
- Token breakdown (input/output for each model)
- Parallel overlap analysis

### Visualization

Generate visual charts comparing baseline vs speculative execution:

```bash
python3 tools/benchmark_langchain.py --moves-per-player 3 --k 2 --runs 1 --visualize
```

**Requirements**: Install visualization dependencies first:
```bash
pip install pandas matplotlib
```

**Outputs** (in `benchmark_results/`):
- `benchmark_visualization.png`: 4-panel chart showing:
  - Latency comparison (baseline vs speculative)
  - Token cost comparison
  - Prediction accuracy
  - Performance metrics summary
- `benchmark_results.csv`: Detailed data in CSV format

### Example Benchmark Session

```bash
# 1. Quick sanity check
python3 tools/benchmark_langchain.py --quick

# 2. Test different k values
python3 tools/benchmark_langchain.py --moves-per-player 3 --k 2,3,5 --runs 2

# 3. Generate visualizations
python3 tools/benchmark_langchain.py --moves-per-player 5 --k 2,3 --runs 3 --visualize

# 4. Full game evaluation (takes longer)
python3 tools/benchmark_langchain.py --full-game --k 2 --runs 1
```

### Interpreting Results

**Good Performance Indicators**:
- ✅ Speculator accuracy > 40%
- ✅ Latency improvement > 10% vs baseline
- ✅ Token overhead < 30% vs baseline
- ✅ Cache hit rate > 20%

**Poor Performance Indicators**:
- ❌ Speculator accuracy < 20%
- ❌ Latency overhead (slower than baseline)
- ❌ Token overhead > 50%
- ❌ Wasted compute ratio > 70%

**Note**: Performance depends on:
- Speculator model quality (GPT-4o-mini vs alternatives)
- Actor model behavior (consistency vs creativity)
- Game phase (opening vs endgame)
- k value (more predictions = higher cost, potentially higher accuracy)

### Advanced Usage

**Skip baseline** (faster, but no comparison):
```bash
python3 tools/benchmark_langchain.py --moves-per-player 3 --k 2 --no-baseline
```

**Multiple k values** (compare different speculation levels):
```bash
python3 tools/benchmark_langchain.py --moves-per-player 3 --k 2,3,5 --runs 2
```

**Custom output directory**:
```bash
python3 tools/benchmark_langchain.py --moves-per-player 3 --k 2 --output-dir my_results
```

## Project Structure

- `langchain/`: Core implementation (Actor, Speculator, Framework, Game)
- `core/`: Chess environment and game logic
- `tools/`: Benchmarking tool
- `project_structure.md`: Detailed architecture documentation
- `project_summary.md`: High-level overview

## Key Concepts

- **Actor**: Slow, authoritative model (GPT-4o) that makes final decisions
- **Speculator**: Fast, predictive model (GPT-4o-mini) that guesses likely moves
- **Speculative Execution**: Running Actor and Speculator in parallel
- **Cache**: FEN-keyed storage of in-flight speculative calls
- **Validation**: Checking if predictions match actual moves
- **Rollback**: Canceling incorrect speculative work

## Performance Notes

- **API Costs**: Each run makes multiple API calls. Monitor your OpenAI usage.
- **Execution Time**: Full games can take 5-15 minutes depending on moves and API latency.
- **Accuracy**: Speculator accuracy typically ranges from 20-50% depending on game phase.
- **Optimization**: Higher k values increase cost but may improve accuracy.

## Troubleshooting

**"OPENAI_API_KEY not set"**:
- Ensure `.env` file exists in repo root with `OPENAI_API_KEY=your-key`

**"ModuleNotFoundError: No module named 'langchain'"**:
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt`

**Visualization fails**:
- Install dependencies: `pip install pandas matplotlib`

**Benchmark takes too long**:
- Use `--quick` for fast testing
- Use `--moves-per-player 1` for minimal moves
- Use `--no-baseline` to skip baseline comparison

## License

See LICENSE file for details.
