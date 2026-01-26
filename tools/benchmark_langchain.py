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
from typing import Any, Dict, List, Tuple


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


class ActorWrapper:
    def __init__(self, actor: Any, metrics: RunMetrics):
        self._actor = actor
        self._metrics = metrics

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
    repo_root = Path(__file__).parent
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


async def _run_langchain_once(k: int, max_turns: int) -> Tuple[Dict[str, Any], RunMetrics]:
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
    return await game.play_game(max_turns=max_turns), metrics


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
    }


async def _run_impl(
    k_values: List[int],
    runs: int,
    max_turns: int,
    include_baseline: bool
) -> List[Tuple[int, Dict[str, float], Dict[str, float] | None, Dict[str, Any]]]:
    rows = []

    baseline_summary = None
    baseline_results: List[Dict[str, Any]] | None = None
    if include_baseline:
        baseline_results = []
        for _ in range(runs):
            result, _ = await _run_langchain_once(0, max_turns)
            baseline_results.append(result)
        baseline_summary = _summarize(baseline_results)

    for k in k_values:
        results: List[Dict[str, Any]] = []
        metrics_list: List[RunMetrics] = []
        for _ in range(runs):
            result, metrics = await _run_langchain_once(k, max_turns)
            results.append(result)
            metrics_list.append(metrics)
        summary = _summarize(results)
        detailed = _summarize_detailed(results, metrics_list, baseline_results)
        rows.append((k, summary, baseline_summary, detailed))

    return rows


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
        print(f"  final_board_hash: {detailed['final_board_hash']}")
        print(f"  move_sequence_hash: {detailed['move_sequence_hash']}")
        print(f"  baseline_match_rate: {detailed['baseline_match_rate']:.2f}")
        print("-" * 80)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark speculative vs sequential runs (LangChain only).")
    parser.add_argument("--k", default="3", help="Comma-separated k values, e.g. 2,3,5")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Fast run: k=2, runs=1, max-turns=1")
    args = parser.parse_args()

    _load_dotenv_if_present()
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set. Set it in .env or export it.")
        return 1

    if args.quick:
        args.k = "2"
        args.runs = 1
        args.max_turns = 1

    k_values = _parse_k_list(args.k)
    include_baseline = not args.no_baseline

    rows: List[Tuple[int, Dict[str, float], Dict[str, float] | None, Dict[str, Any]]] = []
    rows.extend(
        asyncio.run(_run_impl(k_values, args.runs, args.max_turns, include_baseline))
    )

    _print_table(rows)
    _print_details(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
