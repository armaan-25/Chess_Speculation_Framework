#!/usr/bin/env python3
"""
Quick test script to verify the framework is working.
Run this to test without needing pytest.
"""
import os
import sys
import asyncio

def test_unit_tests():
    """Run unit tests that don't require API calls."""
    print("=" * 60)
    print("Running Unit Tests (No API Key Required)")
    print("=" * 60)
    
    import chess
    from core.chess_environment import ChessEnvironment
    from langchain.speculative_framework_langchain import SpeculativeCache, PendingAction, SpeculativeFramework
    from langchain.actor_langchain import Actor
    from langchain.speculator_langchain import Speculator
    
    # Test 1: Chess Environment
    print("\n1. Testing Chess Environment...")
    env = ChessEnvironment()
    state = env.get_state()
    assert state.turn == 0
    assert state.current_player == chess.WHITE
    print("   ✓ Initial state correct")
    
    move = chess.Move.from_uci("e2e4")
    new_state = env.transition(state, move)
    assert new_state.turn == 1
    assert new_state.current_player == chess.BLACK
    print("   ✓ Move application works")
    
    # Test 2: Cache
    print("\n2. Testing Cache...")
    cache = SpeculativeCache()
    handler = "decide_move"
    params = {"state_fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"}
    
    assert cache.get(handler, params) is None
    print("   ✓ Cache initially empty")
    
    future = asyncio.Future()
    pending = PendingAction(handler=handler, params=params, future=future)
    cache.put(handler, params, pending)
    retrieved = cache.get(handler, params)
    assert retrieved == pending
    print("   ✓ Cache put/get works")
    
    # Test 3: Validation (without instantiating models)
    print("\n3. Testing Validation...")
    # Create a minimal framework just for validation testing
    # We'll use mock objects to avoid needing API key
    class MockActor:
        pass
    class MockSpeculator:
        pass
    
    framework = SpeculativeFramework(MockActor(), MockSpeculator(), k=3)
    
    move1 = chess.Move.from_uci("e2e4")
    move2 = chess.Move.from_uci("e2e4")
    move3 = chess.Move.from_uci("d2d4")
    
    assert framework.validate_prediction(move1, move2) == True
    assert framework.validate_prediction(move1, move3) == False
    print("   ✓ Prediction validation works")
    
    print("\n" + "=" * 60)
    print("✓ All unit tests passed!")
    print("=" * 60)
    return True


async def test_integration():
    """Test with actual API calls."""
    print("\n" + "=" * 60)
    print("Running Integration Test (API Key Required)")
    print("=" * 60)
    
    # Try to load .env file if it exists
    from pathlib import Path
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
        except ImportError:
            # If python-dotenv not installed, manually parse .env
            with open(env_file) as f:
                for line in f:
                    if line.strip() and not line.startswith("#"):
                        key, value = line.strip().split("=", 1)
                        os.environ[key] = value
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  OPENAI_API_KEY not set. Skipping integration test.")
        print("   Set it with: export OPENAI_API_KEY='your-key-here'")
        return False
    
    from langchain.chess_speculation_langchain import ChessSpeculationGame
    
    print("\nPlaying a short game (3 turns)...")
    print("This will make API calls and may take 10-30 seconds.")
    
    try:
        game = ChessSpeculationGame(k=2)  # Small k for faster testing
        results = await game.play_game(max_turns=3)
        
        print("\n" + "=" * 60)
        print("Game Results:")
        print("=" * 60)
        print(f"Total Time: {results['total_time']:.2f} seconds")
        print(f"Time Saved: {results['time_saved']:.2f} seconds ({results['time_saved_percent']:.1f}%)")
        print(f"Predictions Made: {results['predictions_made']}")
        print(f"Predictions Hit: {results['predictions_hit']}")
        print(f"Prediction Accuracy: {results['prediction_accuracy']:.1f}%")
        print(f"Moves Played: {results['moves_played']}")
        print("=" * 60)
        print("✓ Integration test passed!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("\n🧪 Speculative Reasoning Framework - Quick Test")
    print("=" * 60)
    
    # Run unit tests
    try:
        unit_passed = test_unit_tests()
    except Exception as e:
        print(f"\n❌ Unit tests failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Run integration test
    integration_passed = await test_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    print(f"Unit Tests: {'✓ PASSED' if unit_passed else '❌ FAILED'}")
    if os.getenv("OPENAI_API_KEY"):
        print(f"Integration Test: {'✓ PASSED' if integration_passed else '❌ FAILED'}")
    else:
        print("Integration Test: ⚠️  SKIPPED (no API key)")
    print("=" * 60)
    
    if unit_passed and (not os.getenv("OPENAI_API_KEY") or integration_passed):
        print("\n✅ All tests completed successfully!")
        return 0
    else:
        print("\n⚠️  Some tests failed or were skipped.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

