"""
Example usage of the chess speculation framework.
"""
import asyncio
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from langchain.chess_speculation_langchain import ChessSpeculationGame
from langchain.actor_langchain import Actor
from langchain.speculator_langchain import Speculator


async def example_basic_game():
    """Basic example: play a game with default settings."""
    print("Example 1: Basic Game")
    print("-" * 50)
    
    game = ChessSpeculationGame(k=3)
    results = await game.play_game(max_turns=10)
    
    print(f"Prediction Accuracy: {results['prediction_accuracy']:.1f}%")
    print(f"Time Saved: {results['time_saved_percent']:.1f}%")
    print()


async def example_custom_models():
    """Example with custom model configurations."""
    print("Example 2: Custom Models")
    print("-" * 50)
    
    # Use different models for actor and speculator
    actor = Actor(
        model_name="gpt-4o",
        reasoning_effort="high",
        max_tokens=2048
    )
    
    speculator = Speculator(
        model_name="gpt-4o-mini",
        reasoning_effort="low",
        max_tokens=512,
        temperature=0.3
    )
    
    game = ChessSpeculationGame(
        actor_white=actor,
        actor_black=actor,
        speculator_white=speculator,
        speculator_black=speculator,
        k=5  # More predictions
    )
    
    results = await game.play_game(max_turns=15)
    
    print(f"Predictions Made: {results['predictions_made']}")
    print(f"Predictions Hit: {results['predictions_hit']}")
    print(f"Prediction Accuracy: {results['prediction_accuracy']:.1f}%")
    print()


async def example_comparison():
    """Compare speculative vs sequential execution."""
    print("Example 3: Speculation Benefits")
    print("-" * 50)
    
    # Run with speculation
    game_spec = ChessSpeculationGame(k=3)
    results_spec = await game_spec.play_game(max_turns=10)
    
    print("With Speculation:")
    print(f"  Total Time: {results_spec['total_time']:.2f}s")
    print(f"  Time Saved: {results_spec['time_saved_percent']:.1f}%")
    print(f"  Prediction Accuracy: {results_spec['prediction_accuracy']:.1f}%")
    print()


async def main():
    """Run all examples."""
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it before running examples.")
        return
    
    print("=" * 50)
    print("Chess Speculation Framework - Examples")
    print("=" * 50)
    print()
    
    await example_basic_game()
    await example_custom_models()
    await example_comparison()
    
    print("=" * 50)
    print("Examples completed!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())

