#!/usr/bin/env python3
"""
LangChain-only benchmark (standalone).
"""
from __future__ import annotations

import argparse
import asyncio
import contextvars
import hashlib
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False


TURN_ID: contextvars.ContextVar[int | None] = contextvars.ContextVar("TURN_ID", default=None)
CALL_KIND: contextvars.ContextVar[str] = contextvars.ContextVar("CALL_KIND", default="actual")


class RunMetrics:
    def __init__(self) -> None:
        self.actor_latencies: List[float] = []
        self.actor_actual_latencies: List[float] = []
        self.actor_actual_latency_by_turn: Dict[int, float] = {}
        self.speculator_latencies: List[float] = []
        self.speculator_latency_by_turn: Dict[int, float] = {}
        self.cache_hit_wait_times: List[float] = []
        self.cache_hit_wait_by_turn: Dict[int, float] = {}
        self.cache_hit_turns: set[int] = set()
        self.cache_lookups: int = 0
        self.cache_hits: int = 0
        self.speculative_tasks_launched: int = 0
        self.speculative_tasks_canceled: int = 0
        self.actor_calls: int = 0
        self.speculative_actor_calls: int = 0
        self.speculator_calls: int = 0
        self.turn_times: List[float] = []
        self.next_turn_id: int = 0
        self.move_sequence: List[str] = []
        # Token tracking
        self.actor_tokens_input: List[int] = []
        self.actor_tokens_output: List[int] = []
        self.speculator_tokens_input: List[int] = []
        self.speculator_tokens_output: List[int] = []
        self.speculative_actor_tokens_input: List[int] = []
        self.speculative_actor_tokens_output: List[int] = []


class InstrumentedFuture:
    def __init__(self, future: asyncio.Future, metrics: RunMetrics, turn_id: int | None):
        self._future = future
        self._metrics = metrics
        self._turn_id = turn_id

    def done(self) -> bool:
        return self._future.done()

    def cancel(self) -> bool:
        return self._future.cancel()

    def add_done_callback(self, fn) -> None:
        self._future.add_done_callback(fn)

    def __await__(self):
        start = time.time()
        result = yield from self._future.__await__()
        wait_time = time.time() - start
        self._metrics.cache_hit_wait_times.append(wait_time)
        if self._turn_id is not None:
            self._metrics.cache_hit_wait_by_turn[self._turn_id] = wait_time
        return result


class InstrumentedCache:
    def __init__(self, metrics: RunMetrics):
        self._cache: Dict[Tuple[str, str], Any] = {}
        self._metrics = metrics

    def _make_key(self, handler: str, params: dict) -> Tuple[str, str]:
        fen = params.get("state_fen", "")
        return (handler, fen)

    def get(self, handler: str, params: dict) -> Any:
        self._metrics.cache_lookups += 1
        key = self._make_key(handler, params)
        cached = self._cache.get(key)
        if cached:
            self._metrics.cache_hits += 1
            turn_id = TURN_ID.get()
            if turn_id is not None:
                self._metrics.cache_hit_turns.add(turn_id)
            if not isinstance(cached.future, InstrumentedFuture):
                cached.future = InstrumentedFuture(cached.future, self._metrics, turn_id)
        return cached

    def put(self, handler: str, params: dict, pending_action: Any) -> None:
        key = self._make_key(handler, params)
        self._cache[key] = pending_action

    def clear(self) -> None:
        self._cache.clear()


class InstrumentedModel:
    """Wrapper for ChatOpenAI that tracks token usage."""
    def __init__(self, model: Any, metrics: RunMetrics, is_actor: bool = True):
        self._model = model
        self._metrics = metrics
        self._is_actor = is_actor
    
    async def ainvoke(self, messages, **kwargs):
        response = await self._model.ainvoke(messages, **kwargs)
        # Extract token usage
        metadata = getattr(response, 'response_metadata', {})
        token_usage = metadata.get('token_usage', {})
        prompt_tokens = token_usage.get('prompt_tokens', 0)
        completion_tokens = token_usage.get('completion_tokens', 0)
        
        if self._is_actor:
            kind = CALL_KIND.get()
            if kind == "speculative":
                self._metrics.speculative_actor_tokens_input.append(prompt_tokens)
                self._metrics.speculative_actor_tokens_output.append(completion_tokens)
            else:
                self._metrics.actor_tokens_input.append(prompt_tokens)
                self._metrics.actor_tokens_output.append(completion_tokens)
        else:
            self._metrics.speculator_tokens_input.append(prompt_tokens)
            self._metrics.speculator_tokens_output.append(completion_tokens)
        
        return response
    
    def __getattr__(self, name: str):
        return getattr(self._model, name)


class ActorWrapper:
    def __init__(self, actor: Any, metrics: RunMetrics):
        self._actor = actor
        self._metrics = metrics
        # Replace the model with an instrumented version
        original_model = actor.model
        actor.model = InstrumentedModel(original_model, metrics, is_actor=True)

    async def decide_move(self, state):
        start = time.time()
        result = await self._actor.decide_move(state)
        latency = time.time() - start
        self._metrics.actor_calls += 1
        self._metrics.actor_latencies.append(latency)

        kind = CALL_KIND.get()
        turn_id = TURN_ID.get()
        if kind == "speculative":
            self._metrics.speculative_actor_calls += 1
        else:
            self._metrics.actor_actual_latencies.append(latency)
            if turn_id is not None:
                self._metrics.actor_actual_latency_by_turn[turn_id] = latency

        return result

    def __getattr__(self, name: str):
        return getattr(self._actor, name)


class SpeculatorWrapper:
    def __init__(self, speculator: Any, metrics: RunMetrics):
        self._speculator = speculator
        self._metrics = metrics
        # Replace the model with an instrumented version
        original_model = speculator.model
        speculator.model = InstrumentedModel(original_model, metrics, is_actor=False)

    async def predict_moves(self, state, k: int = 3):
        start = time.time()
        result = await self._speculator.predict_moves(state, k=k)
        latency = time.time() - start
        self._metrics.speculator_calls += 1
        self._metrics.speculator_latencies.append(latency)
        turn_id = TURN_ID.get()
        if turn_id is not None:
            self._metrics.speculator_latency_by_turn[turn_id] = latency
        return result

    def __getattr__(self, name: str):
        return getattr(self._speculator, name)


def _load_dotenv_if_present() -> None:
    repo_root = Path(__file__).parent.parent  # Go up from tools/ to repo root
    env_paths = [repo_root / ".env", repo_root / "core" / ".env"]
    for env_path in env_paths:
        if env_path.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_path)
            except Exception:
                with env_path.open() as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            key, value = line.split("=", 1)
                            os.environ.setdefault(key, value)


def _parse_k_list(raw: str) -> List[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return [int(p) for p in parts]


def _avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stddev(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def _run_langchain_once(
    k: int, 
    max_moves_per_player: Optional[int] = None,
    full_game: bool = False
) -> Tuple[Dict[str, Any], RunMetrics]:
    # Add repo root to path (same pattern as other files)
    repo_root = Path(__file__).parent.parent
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    
    from langchain.chess_speculation_langchain import ChessSpeculationGame
    from langchain.speculative_framework_langchain import SpeculativeFramework, PendingAction
    from langchain.actor_langchain import Actor
    from langchain.speculator_langchain import Speculator

    metrics = RunMetrics()

    class InstrumentedSpeculativeFramework(SpeculativeFramework):
        def __init__(self, actor, speculator, k_val: int):
            super().__init__(actor, speculator, k_val)
            self.cache = InstrumentedCache(metrics)

        async def execute_api_call(self, handler: str, params: dict, is_speculative: bool = False) -> Any:
            cached = self.cache.get(handler, params)
            if cached:
                return await cached.future

            token = CALL_KIND.set("actual")
            try:
                future = asyncio.create_task(self._call_handler(handler, params))
            finally:
                CALL_KIND.reset(token)

            pending_action = PendingAction(
                handler=handler,
                params=params,
                future=future,
                is_speculative=is_speculative
            )
            self.cache.put(handler, params, pending_action)
            return await future

        def pre_launch_speculative_call(self, handler: str, params: dict) -> Any:
            cached = self.cache.get(handler, params)
            if cached:
                return cached

            token = CALL_KIND.set("speculative")
            try:
                future = asyncio.create_task(self._call_handler(handler, params))
            finally:
                CALL_KIND.reset(token)

            pending_action = PendingAction(
                handler=handler,
                params=params,
                future=future,
                is_speculative=True
            )
            self.cache.put(handler, params, pending_action)
            metrics.speculative_tasks_launched += 1
            return pending_action

        async def cancel_speculative_actions(self, actions: List[PendingAction]):
            to_cancel = [a for a in actions if a.is_speculative]
            metrics.speculative_tasks_canceled += len(to_cancel)
            for action in actions:
                if action.is_speculative and not action.future.done():
                    action.future.cancel()
                    try:
                        await action.future
                    except asyncio.CancelledError:
                        pass

    actor = ActorWrapper(Actor(model_name="gpt-4o", reasoning_effort="high"), metrics)
    speculator = SpeculatorWrapper(Speculator(model_name="gpt-4o-mini", reasoning_effort="low"), metrics)

    game = ChessSpeculationGame(
        actor_white=actor,
        actor_black=actor,
        speculator_white=speculator,
        speculator_black=speculator,
        k=k
    )
    game.framework_white = InstrumentedSpeculativeFramework(actor, speculator, k)
    game.framework_black = InstrumentedSpeculativeFramework(actor, speculator, k)

    original_execute = game._execute_turn

    async def instrumented_execute(state, framework, spec, player_color):
        turn_id = metrics.next_turn_id
        metrics.next_turn_id += 1
        token = TURN_ID.set(turn_id)
        start = time.time()
        try:
            result_state = await original_execute(state, framework, spec, player_color)
        finally:
            TURN_ID.reset(token)
        metrics.turn_times.append(time.time() - start)
        last_move = result_state.get("last_move")
        if last_move:
            metrics.move_sequence.append(last_move.uci())
        return result_state

    game._execute_turn = instrumented_execute  # type: ignore[assignment]
    
    # Determine game mode
    if full_game:
        result = await game.play_game(max_moves_per_player=None)
    else:
        result = await game.play_game(max_moves_per_player=max_moves_per_player)
    
    return result, metrics


def _summarize(results: List[Dict[str, Any]]) -> Dict[str, float]:
    return {
        "total_time": _avg([r.get("total_time", 0.0) for r in results]),
        "time_saved_percent": _avg([r.get("time_saved_percent", 0.0) for r in results]),
        "prediction_accuracy": _avg([r.get("prediction_accuracy", 0.0) for r in results]),
        "predictions_made": _avg([r.get("predictions_made", 0.0) for r in results]),
        "predictions_hit": _avg([r.get("predictions_hit", 0.0) for r in results]),
        "moves_played": _avg([r.get("moves_played", 0.0) for r in results]),
    }


def _summarize_detailed(
    results: List[Dict[str, Any]],
    metrics_list: List[RunMetrics],
    baseline_results: List[Dict[str, Any]] | None
) -> Dict[str, Any]:
    per_run = []
    for idx, metrics in enumerate(metrics_list):
        baseline_avg_turn = 0.0
        baseline_final_fen = None
        if baseline_results and idx < len(baseline_results):
            base = baseline_results[idx]
            moves = base.get("moves_played", 0) or 0
            if moves > 0:
                baseline_avg_turn = base.get("total_time", 0.0) / moves
            baseline_final_fen = base.get("final_fen")

        cache_hit_rate = (
            metrics.cache_hits / metrics.cache_lookups
            if metrics.cache_lookups > 0 else 0.0
        )
        avg_cache_hit_latency = _avg(metrics.cache_hit_wait_times)
        avg_cache_miss_latency = _avg([
            v for k, v in metrics.actor_actual_latency_by_turn.items()
            if k not in metrics.cache_hit_turns
        ])

        overlaps = []
        overlap_turns = 0
        for turn_id, actor_latency in metrics.actor_actual_latency_by_turn.items():
            spec_latency = metrics.speculator_latency_by_turn.get(turn_id)
            if spec_latency is not None:
                overlap = min(actor_latency, spec_latency)
                overlaps.append(overlap)
                if overlap > 0:
                    overlap_turns += 1

        avg_overlap = _avg(overlaps)
        percent_turns_with_overlap = (
            (overlap_turns / len(overlaps)) * 100.0 if overlaps else 0.0
        )

        max_turn_speedup = 0.0
        turns_with_speedup = 0
        if baseline_avg_turn > 0:
            for turn_time in metrics.turn_times:
                speedup = (baseline_avg_turn - turn_time) / baseline_avg_turn * 100.0
                if speedup > 0:
                    turns_with_speedup += 1
                max_turn_speedup = max(max_turn_speedup, speedup)

        total_llm_calls = metrics.actor_calls + metrics.speculator_calls
        latency_per_llm_call = (
            results[idx].get("total_time", 0.0) / total_llm_calls
            if total_llm_calls > 0 else 0.0
        )

        final_fen = results[idx].get("final_fen", "")
        baseline_match = (
            baseline_final_fen == final_fen
            if baseline_final_fen is not None else None
        )

        # Calculate token totals
        actor_tokens_input_total = sum(metrics.actor_tokens_input)
        actor_tokens_output_total = sum(metrics.actor_tokens_output)
        speculator_tokens_input_total = sum(metrics.speculator_tokens_input)
        speculator_tokens_output_total = sum(metrics.speculator_tokens_output)
        speculative_actor_tokens_input_total = sum(metrics.speculative_actor_tokens_input)
        speculative_actor_tokens_output_total = sum(metrics.speculative_actor_tokens_output)
        
        # Total tokens for actor (actual + speculative)
        total_actor_tokens_input = actor_tokens_input_total + speculative_actor_tokens_input_total
        total_actor_tokens_output = actor_tokens_output_total + speculative_actor_tokens_output_total
        
        per_run.append({
            "avg_actor_latency": _avg(metrics.actor_actual_latencies),
            "avg_speculator_latency": _avg(metrics.speculator_latencies),
            "avg_wait_time_on_cache_hit": _avg(metrics.cache_hit_wait_times),
            "cache_hit_rate": cache_hit_rate,
            "avg_cache_hit_latency": avg_cache_hit_latency,
            "avg_cache_miss_latency": avg_cache_miss_latency,
            "speculative_tasks_launched": metrics.speculative_tasks_launched,
            "speculative_tasks_canceled": metrics.speculative_tasks_canceled,
            "wasted_compute_ratio": (
                metrics.speculative_tasks_canceled / metrics.speculative_tasks_launched
                if metrics.speculative_tasks_launched > 0 else 0.0
            ),
            "avg_parallel_overlap_time": avg_overlap,
            "percent_turns_with_overlap": percent_turns_with_overlap,
            "avg_turn_time_speculative": _avg(metrics.turn_times),
            "max_turn_speedup": max_turn_speedup,
            "turns_with_speedup": turns_with_speedup,
            "actor_calls": metrics.actor_calls,
            "speculator_calls": metrics.speculator_calls,
            "speculative_actor_calls": metrics.speculative_actor_calls,
            "total_llm_calls": total_llm_calls,
            "latency_per_llm_call": latency_per_llm_call,
            "final_board_hash": _hash_text(final_fen),
            "move_sequence_hash": _hash_text(",".join(metrics.move_sequence)),
            "baseline_match": baseline_match,
            # Token metrics
            "actor_tokens_input": actor_tokens_input_total,
            "actor_tokens_output": actor_tokens_output_total,
            "speculator_tokens_input": speculator_tokens_input_total,
            "speculator_tokens_output": speculator_tokens_output_total,
            "speculative_actor_tokens_input": speculative_actor_tokens_input_total,
            "speculative_actor_tokens_output": speculative_actor_tokens_output_total,
            "total_actor_tokens_input": total_actor_tokens_input,
            "total_actor_tokens_output": total_actor_tokens_output,
            "total_actor_tokens": total_actor_tokens_input + total_actor_tokens_output,
            "total_speculator_tokens": speculator_tokens_input_total + speculator_tokens_output_total,
        })

    return {
        "avg_actor_latency": _avg([r["avg_actor_latency"] for r in per_run]),
        "avg_speculator_latency": _avg([r["avg_speculator_latency"] for r in per_run]),
        "avg_wait_time_on_cache_hit": _avg([r["avg_wait_time_on_cache_hit"] for r in per_run]),
        "cache_hit_rate": _avg([r["cache_hit_rate"] for r in per_run]),
        "avg_cache_hit_latency": _avg([r["avg_cache_hit_latency"] for r in per_run]),
        "avg_cache_miss_latency": _avg([r["avg_cache_miss_latency"] for r in per_run]),
        "speculative_tasks_launched": _avg([r["speculative_tasks_launched"] for r in per_run]),
        "speculative_tasks_canceled": _avg([r["speculative_tasks_canceled"] for r in per_run]),
        "wasted_compute_ratio": _avg([r["wasted_compute_ratio"] for r in per_run]),
        "avg_parallel_overlap_time": _avg([r["avg_parallel_overlap_time"] for r in per_run]),
        "percent_turns_with_overlap": _avg([r["percent_turns_with_overlap"] for r in per_run]),
        "avg_turn_time_speculative": _avg([r["avg_turn_time_speculative"] for r in per_run]),
        "max_turn_speedup": _avg([r["max_turn_speedup"] for r in per_run]),
        "turns_with_speedup": _avg([r["turns_with_speedup"] for r in per_run]),
        "actor_calls": _avg([r["actor_calls"] for r in per_run]),
        "speculator_calls": _avg([r["speculator_calls"] for r in per_run]),
        "speculative_actor_calls": _avg([r["speculative_actor_calls"] for r in per_run]),
        "total_llm_calls": _avg([r["total_llm_calls"] for r in per_run]),
        "latency_per_llm_call": _avg([r["latency_per_llm_call"] for r in per_run]),
        "final_board_hash": per_run[0]["final_board_hash"] if per_run else "",
        "move_sequence_hash": per_run[0]["move_sequence_hash"] if per_run else "",
        "baseline_match_rate": _avg([
            1.0 if r["baseline_match"] is True else 0.0
            for r in per_run if r["baseline_match"] is not None
        ]),
        "stddev_total_time": _stddev([r.get("total_time", 0.0) for r in results]),
        "stddev_actor_latency": _stddev(
            [lat for m in metrics_list for lat in m.actor_actual_latencies]
        ),
        # Token metrics (sum across runs)
        "actor_tokens_input": sum([r["actor_tokens_input"] for r in per_run]),
        "actor_tokens_output": sum([r["actor_tokens_output"] for r in per_run]),
        "speculator_tokens_input": sum([r["speculator_tokens_input"] for r in per_run]),
        "speculator_tokens_output": sum([r["speculator_tokens_output"] for r in per_run]),
        "speculative_actor_tokens_input": sum([r["speculative_actor_tokens_input"] for r in per_run]),
        "speculative_actor_tokens_output": sum([r["speculative_actor_tokens_output"] for r in per_run]),
        "total_actor_tokens_input": sum([r["total_actor_tokens_input"] for r in per_run]),
        "total_actor_tokens_output": sum([r["total_actor_tokens_output"] for r in per_run]),
        "total_actor_tokens": sum([r["total_actor_tokens"] for r in per_run]),
        "total_speculator_tokens": sum([r["total_speculator_tokens"] for r in per_run]),
    }


async def _run_impl(
    k_values: List[int],
    runs: int,
    max_moves_per_player: Optional[int],
    full_game: bool,
    include_baseline: bool
) -> List[Tuple[int, Dict[str, float], Dict[str, float] | None, Dict[str, Any]]]:
    rows = []

    baseline_summary = None
    baseline_results: List[Dict[str, Any]] | None = None
    baseline_metrics_list: List[RunMetrics] | None = None
    if include_baseline:
        baseline_results = []
        baseline_metrics_list = []
        for _ in range(runs):
            result, metrics = await _run_langchain_once(0, max_moves_per_player, full_game)
            baseline_results.append(result)
            baseline_metrics_list.append(metrics)
        baseline_summary = _summarize(baseline_results)
        # Create baseline detailed summary for token comparison
        baseline_detailed = _summarize_detailed(baseline_results, baseline_metrics_list, None)
        baseline_summary["baseline_detailed"] = baseline_detailed

    for k in k_values:
        results: List[Dict[str, Any]] = []
        metrics_list: List[RunMetrics] = []
        for _ in range(runs):
            result, metrics = await _run_langchain_once(k, max_moves_per_player, full_game)
            results.append(result)
            metrics_list.append(metrics)
        summary = _summarize(results)
        detailed = _summarize_detailed(results, metrics_list, baseline_results)
        rows.append((k, summary, baseline_summary, detailed))

    return rows


def _print_key_metrics(
    rows: List[Tuple[int, Dict[str, float], Dict[str, float] | None, Dict[str, Any]]]
) -> None:
    """Print the three key metrics: Accuracy, Latency, and Cost."""
    print("\n" + "=" * 80)
    print("KEY METRICS: Speculator Accuracy, End-to-End Latency, Token Cost")
    print("=" * 80)
    
    for k, summary, baseline, detailed in rows:
        print(f"\nConfiguration: k={k}")
        print("-" * 80)
        
        # 1. Accuracy of speculator vs actor
        accuracy = summary["prediction_accuracy"]
        print(f"1. SPECULATOR ACCURACY: {accuracy:.2f}%")
        print(f"   (Predictions matched: {summary['predictions_hit']:.1f} / {summary['predictions_made']:.1f})")
        
        # 2. End-to-end latency
        total_time = summary["total_time"]
        print(f"\n2. END-TO-END LATENCY: {total_time:.3f}s")
        if baseline:
            baseline_time = baseline["total_time"]
            latency_improvement = ((baseline_time - total_time) / baseline_time * 100.0) if baseline_time > 0 else 0.0
            print(f"   Baseline (no speculation): {baseline_time:.3f}s")
            if latency_improvement > 0:
                print(f"   Improvement: {latency_improvement:.2f}% faster")
            else:
                print(f"   Overhead: {abs(latency_improvement):.2f}% slower")
        print(f"   Avg turn time: {detailed['avg_turn_time_speculative']:.3f}s")
        
        # 3. Token cost
        actor_tokens = detailed["total_actor_tokens"]
        speculator_tokens = detailed["total_speculator_tokens"]
        total_tokens = actor_tokens + speculator_tokens
        print(f"\n3. TOKEN COST:")
        print(f"   Actor tokens: {actor_tokens:,} (input: {detailed['total_actor_tokens_input']:,}, output: {detailed['total_actor_tokens_output']:,})")
        print(f"   Speculator tokens: {speculator_tokens:,} (input: {detailed['speculator_tokens_input']:,}, output: {detailed['speculator_tokens_output']:,})")
        print(f"   Total tokens: {total_tokens:,}")
        
        if baseline and "baseline_detailed" in baseline:
            baseline_detailed = baseline["baseline_detailed"]
            baseline_actor_tokens = baseline_detailed.get("total_actor_tokens", 0)
            baseline_speculator_tokens = baseline_detailed.get("total_speculator_tokens", 0)
            baseline_total_tokens = baseline_actor_tokens + baseline_speculator_tokens
            
            token_overhead = total_tokens - baseline_total_tokens
            token_overhead_percent = (token_overhead / baseline_total_tokens * 100.0) if baseline_total_tokens > 0 else 0.0
            
            print(f"   Baseline tokens: {baseline_total_tokens:,} (actor: {baseline_actor_tokens:,}, speculator: {baseline_speculator_tokens:,})")
            if token_overhead > 0:
                print(f"   Token overhead: +{token_overhead:,.0f} ({token_overhead_percent:.2f}%)")
            else:
                print(f"   Token savings: {abs(token_overhead):,.0f} ({abs(token_overhead_percent):.2f}%)")
        
        print("-" * 80)
    
    print("=" * 80 + "\n")


def _print_table(rows: List[Tuple[int, Dict[str, float], Dict[str, float] | None, Dict[str, Any]]]) -> None:
    header = (
        "impl | k | avg_total_time_s | avg_time_saved_% | avg_accuracy_% | "
        "avg_pred_made | avg_pred_hit | speedup_vs_baseline_%"
    )
    print(header)
    print("-" * len(header))
    for k, summary, baseline, _ in rows:
        total_time = summary["total_time"]
        time_saved = summary["time_saved_percent"]
        accuracy = summary["prediction_accuracy"]
        pred_made = summary["predictions_made"]
        pred_hit = summary["predictions_hit"]
        speedup = 0.0
        if baseline and baseline["total_time"] > 0:
            speedup = (baseline["total_time"] - total_time) / baseline["total_time"] * 100.0
        print(
            f"langchain | {k} | {total_time:.2f} | {time_saved:.2f} | "
            f"{accuracy:.2f} | {pred_made:.2f} | {pred_hit:.2f} | {speedup:.2f}"
        )


def _visualize_results(
    rows: List[Tuple[int, Dict[str, float], Dict[str, float] | None, Dict[str, Any]]],
    output_dir: Path
) -> None:
    """Create visualizations using pandas and matplotlib."""
    if not HAS_VISUALIZATION:
        print("\nNote: Install pandas and matplotlib for visualizations: pip install pandas matplotlib")
        return
    
    output_dir.mkdir(exist_ok=True)
    
    # Prepare data for DataFrame
    data = []
    for k, summary, baseline, detailed in rows:
        row_data = {
            'k': k,
            'total_time': summary['total_time'],
            'prediction_accuracy': summary['prediction_accuracy'],
            'actor_tokens': detailed['total_actor_tokens'],
            'speculator_tokens': detailed['total_speculator_tokens'],
            'total_tokens': detailed['total_actor_tokens'] + detailed['total_speculator_tokens'],
            'cache_hit_rate': detailed['cache_hit_rate'],
            'wasted_compute_ratio': detailed['wasted_compute_ratio'],
        }
        
        if baseline:
            row_data['baseline_time'] = baseline['total_time']
            baseline_detailed = baseline.get('baseline_detailed', {})
            baseline_actor_tokens = baseline_detailed.get('total_actor_tokens', 0) if baseline_detailed else 0
            row_data['baseline_tokens'] = baseline_actor_tokens
            row_data['latency_improvement_pct'] = (
                (baseline['total_time'] - summary['total_time']) / baseline['total_time'] * 100.0
                if baseline['total_time'] > 0 else 0.0
            )
            row_data['token_overhead_pct'] = (
                (row_data['total_tokens'] - row_data['baseline_tokens']) / row_data['baseline_tokens'] * 100.0
                if row_data['baseline_tokens'] > 0 else 0.0
            )
        
        data.append(row_data)
    
    df = pd.DataFrame(data)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Benchmark Results: Speculation vs Baseline', fontsize=16, fontweight='bold')
    
    # 1. Latency Comparison
    ax1 = axes[0, 0]
    if 'baseline_time' in df.columns:
        x = range(len(df))
        width = 0.35
        ax1.bar([i - width/2 for i in x], df['baseline_time'], width, label='Baseline (no spec)', alpha=0.8, color='#3498db')
        ax1.bar([i + width/2 for i in x], df['total_time'], width, label=f'With Speculation (k={df["k"].iloc[0]})', alpha=0.8, color='#e74c3c')
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('End-to-End Latency Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'k={k}' for k in df['k']])
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
    else:
        ax1.bar(df['k'], df['total_time'], alpha=0.8, color='#3498db')
        ax1.set_xlabel('k value')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('End-to-End Latency')
        ax1.grid(axis='y', alpha=0.3)
    
    # 2. Token Cost Comparison
    ax2 = axes[0, 1]
    if 'baseline_tokens' in df.columns:
        x = range(len(df))
        width = 0.35
        ax2.bar([i - width/2 for i in x], df['baseline_tokens'], width, label='Baseline (actor only)', alpha=0.8, color='#3498db')
        ax2.bar([i + width/2 for i in x], df['total_tokens'], width, label='With Speculation', alpha=0.8, color='#e74c3c')
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('Total Tokens')
        ax2.set_title('Token Cost Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'k={k}' for k in df['k']])
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
    else:
        ax2.bar(df['k'], df['total_tokens'], alpha=0.8, color='#3498db')
        ax2.set_xlabel('k value')
        ax2.set_ylabel('Total Tokens')
        ax2.set_title('Token Cost')
        ax2.grid(axis='y', alpha=0.3)
    
    # 3. Prediction Accuracy
    ax3 = axes[1, 0]
    ax3.bar(df['k'], df['prediction_accuracy'], alpha=0.8, color='#2ecc71')
    ax3.set_xlabel('k value')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Speculator Prediction Accuracy')
    ax3.set_ylim(0, 100)
    ax3.grid(axis='y', alpha=0.3)
    for i, v in enumerate(df['prediction_accuracy']):
        ax3.text(df['k'].iloc[i], v + 1, f'{v:.1f}%', ha='center', va='bottom')
    
    # 4. Performance Metrics
    ax4 = axes[1, 1]
    if 'latency_improvement_pct' in df.columns:
        metrics_data = {
            'Latency\nImprovement': df['latency_improvement_pct'].mean(),
            'Token\nOverhead': df['token_overhead_pct'].mean(),
            'Cache Hit\nRate': df['cache_hit_rate'].mean() * 100,
            'Wasted\nCompute': df['wasted_compute_ratio'].mean() * 100,
        }
        colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in [
            metrics_data['Latency\nImprovement'],
            metrics_data['Token\nOverhead'],
            metrics_data['Cache Hit\nRate'],
            metrics_data['Wasted\nCompute'],
        ]]
        bars = ax4.bar(metrics_data.keys(), metrics_data.values(), alpha=0.8, color=colors)
        ax4.set_ylabel('Percentage (%)')
        ax4.set_title('Performance Metrics Summary')
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, metrics_data.values()):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
    else:
        ax4.text(0.5, 0.5, 'No baseline comparison\navailable', 
                 ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / 'benchmark_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Also save CSV
    csv_path = output_dir / 'benchmark_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"Results CSV saved to: {csv_path}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)


def _print_details(rows: List[Tuple[int, Dict[str, float], Dict[str, float] | None, Dict[str, Any]]]) -> None:
    print("\nDetailed Metrics")
    print("-" * 80)
    for k, _, baseline, detailed in rows:
        baseline_turn = 0.0
        if baseline and baseline.get("moves_played", 0) > 0:
            baseline_turn = baseline["total_time"] / baseline["moves_played"]
        print(f"langchain | k={k}")
        print(f"  avg_actor_latency: {detailed['avg_actor_latency']:.3f}s")
        print(f"  avg_speculator_latency: {detailed['avg_speculator_latency']:.3f}s")
        print(f"  avg_wait_time_on_cache_hit: {detailed['avg_wait_time_on_cache_hit']:.3f}s")
        print(f"  cache_hit_rate: {detailed['cache_hit_rate']:.3f}")
        print(f"  avg_cache_hit_latency: {detailed['avg_cache_hit_latency']:.3f}s")
        print(f"  avg_cache_miss_latency: {detailed['avg_cache_miss_latency']:.3f}s")
        print(f"  speculative_tasks_launched: {detailed['speculative_tasks_launched']:.1f}")
        print(f"  speculative_tasks_canceled: {detailed['speculative_tasks_canceled']:.1f}")
        print(f"  wasted_compute_ratio: {detailed['wasted_compute_ratio']:.3f}")
        print(f"  avg_parallel_overlap_time: {detailed['avg_parallel_overlap_time']:.3f}s")
        print(f"  percent_turns_with_overlap: {detailed['percent_turns_with_overlap']:.1f}%")
        print(f"  avg_turn_time_baseline: {baseline_turn:.3f}s")
        print(f"  avg_turn_time_speculative: {detailed['avg_turn_time_speculative']:.3f}s")
        print(f"  max_turn_speedup: {detailed['max_turn_speedup']:.1f}%")
        print(f"  turns_with_speedup: {detailed['turns_with_speedup']:.1f}")
        print(f"  stddev_total_time: {detailed['stddev_total_time']:.3f}s")
        print(f"  stddev_actor_latency: {detailed['stddev_actor_latency']:.3f}s")
        print(f"  actor_calls: {detailed['actor_calls']:.1f}")
        print(f"  speculator_calls: {detailed['speculator_calls']:.1f}")
        print(f"  speculative_actor_calls: {detailed['speculative_actor_calls']:.1f}")
        print(f"  total_llm_calls: {detailed['total_llm_calls']:.1f}")
        print(f"  latency_per_llm_call: {detailed['latency_per_llm_call']:.3f}s")
        print(f"  actor_tokens: {detailed['total_actor_tokens']:,} (input: {detailed['total_actor_tokens_input']:,}, output: {detailed['total_actor_tokens_output']:,})")
        print(f"  speculator_tokens: {detailed['total_speculator_tokens']:,} (input: {detailed['speculator_tokens_input']:,}, output: {detailed['speculator_tokens_output']:,})")
        print(f"  total_tokens: {detailed['total_actor_tokens'] + detailed['total_speculator_tokens']:,}")
        print(f"  final_board_hash: {detailed['final_board_hash']}")
        print(f"  move_sequence_hash: {detailed['move_sequence_hash']}")
        print(f"  baseline_match_rate: {detailed['baseline_match_rate']:.2f}")
        print("-" * 80)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark speculative vs sequential runs (LangChain only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Game Modes:
  --full-game              Play until checkmate/stalemate (full game)
  --moves-per-player N     Play until each player makes N moves (default: 3)

Examples:
  # Full game mode
  python3 tools/benchmark_langchain.py --full-game --k 2,3

  # Limited moves mode (5 moves per player)
  python3 tools/benchmark_langchain.py --moves-per-player 5 --k 2,3

  # Quick test (1 move per player, k=2, 1 run)
  python3 tools/benchmark_langchain.py --quick
        """
    )
    parser.add_argument("--k", default="3", help="Comma-separated k values, e.g. 2,3,5")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per configuration")
    parser.add_argument("--max-turns", type=int, help="DEPRECATED: Use --moves-per-player instead")
    parser.add_argument("--moves-per-player", type=int, help="Number of moves per player (ignored if --full-game is set)")
    parser.add_argument("--full-game", action="store_true", help="Play full game until checkmate/stalemate")
    parser.add_argument("--no-baseline", action="store_true", help="Skip baseline (no speculation) run")
    parser.add_argument("--quick", action="store_true", help="Quick test: k=2, runs=1, moves-per-player=1")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization plots (requires pandas and matplotlib)")
    parser.add_argument("--output-dir", type=str, default="benchmark_results", help="Directory to save outputs (default: benchmark_results/)")
    args = parser.parse_args()

    _load_dotenv_if_present()
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set. Set it in .env or export it.")
        return 1

    # Handle quick mode
    if args.quick:
        args.k = "2"
        args.runs = 1
        args.moves_per_player = 1
        args.full_game = False
    
    # Handle deprecated --max-turns
    if args.max_turns is not None:
        print("Warning: --max-turns is deprecated. Use --moves-per-player instead.")
        if args.moves_per_player is None:
            args.moves_per_player = args.max_turns
    
    # Determine game mode
    if args.full_game:
        max_moves_per_player = None
        print("Mode: Full game (until checkmate/stalemate)")
    else:
        max_moves_per_player = args.moves_per_player if args.moves_per_player is not None else 3
        print(f"Mode: Limited moves ({max_moves_per_player} moves per player)")

    k_values = _parse_k_list(args.k)
    include_baseline = not args.no_baseline

    rows: List[Tuple[int, Dict[str, float], Dict[str, float] | None, Dict[str, Any]]] = []
    rows.extend(
        asyncio.run(_run_impl(k_values, args.runs, max_moves_per_player, args.full_game, include_baseline))
    )

    _print_key_metrics(rows)
    _print_table(rows)
    _print_details(rows)
    
    # Generate visualizations if requested
    if args.visualize:
        output_dir = Path(args.output_dir)
        _visualize_results(rows, output_dir)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
