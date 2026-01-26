"""
Tests for chess speculation framework.
"""
import pytest
import chess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.chess_environment import ChessEnvironment, GameState
from langchain.actor_langchain import Actor
from langchain.speculator_langchain import Speculator
from langchain.speculative_framework_langchain import SpeculativeFramework, SpeculativeCache, PendingAction


class TestChessEnvironment:
    """Tests for chess environment."""
    
    def test_initial_state(self):
        """Test initial game state."""
        env = ChessEnvironment()
        state = env.get_state()
        
        assert state.turn == 0
        assert state.current_player == chess.WHITE
        assert not state.is_terminal()
    
    def test_move_application(self):
        """Test applying moves."""
        env = ChessEnvironment()
        state = env.get_state()
        
        move = chess.Move.from_uci("e2e4")
        new_state = env.transition(state, move)
        
        assert new_state.turn == 1
        assert new_state.current_player == chess.BLACK
        assert move in state.get_legal_moves()
    
    def test_parse_move(self):
        """Test move parsing."""
        env = ChessEnvironment()
        state = env.get_state()
        
        move = env.parse_move("e2e4", state)
        assert move is not None
        assert move.uci() == "e2e4"
        
        # Invalid move
        invalid = env.parse_move("e2e5", state)  # Illegal move
        assert invalid is None


class TestSpeculativeCache:
    """Tests for speculative cache."""
    
    def test_cache_operations(self):
        """Test cache get/put operations."""
        cache = SpeculativeCache()
        
        handler = "decide_move"
        params1 = {"state_fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"}
        params2 = {"state_fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"}
        
        # Initially empty
        assert cache.get(handler, params1) is None
        
        # Create mock pending action (ensure we have an event loop)
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            future = loop.create_future()
            pending = PendingAction(handler=handler, params=params1, future=future)
        finally:
            loop.close()
            asyncio.set_event_loop(None)
        
        # Put and get
        cache.put(handler, params1, pending)
        retrieved = cache.get(handler, params1)
        assert retrieved == pending
        
        # Different params should not match
        assert cache.get(handler, params2) is None


class TestSpeculativeFramework:
    """Tests for speculative framework."""
    
    def test_validation(self):
        """Test prediction validation."""
        # Use dummy actor/speculator to avoid API key requirements
        class DummyActor:
            pass

        class DummySpeculator:
            pass

        framework = SpeculativeFramework(DummyActor(), DummySpeculator(), k=3)
        
        move1 = chess.Move.from_uci("e2e4")
        move2 = chess.Move.from_uci("e2e4")
        move3 = chess.Move.from_uci("d2d4")
        
        # Same moves should validate
        assert framework.validate_prediction(move1, move2) == True
        
        # Different moves should not validate
        assert framework.validate_prediction(move1, move3) == False


def test_integration():
    """Integration test (requires API key)."""
    import os
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set, skipping integration test")
    
    # Verify components can be instantiated
    actor = Actor(model_name="gpt-4o-mini")
    speculator = Speculator(model_name="gpt-4o-mini")
    framework = SpeculativeFramework(actor, speculator, k=3)
    
    assert framework.actor is not None
    assert framework.speculator is not None
    assert framework.k == 3


@pytest.mark.asyncio
async def test_full_game_short():
    """Test a short game (requires API key)."""
    import os
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set, skipping integration test")
    
    from langchain.chess_speculation_langchain import ChessSpeculationGame
    
    # Run a very short game (3 turns)
    game = ChessSpeculationGame(k=2)  # Small k for faster testing
    results = await game.play_game(max_turns=3)
    
    # Verify results structure
    assert "total_time" in results
    assert "prediction_accuracy" in results
    assert "time_saved_percent" in results
    assert "moves_played" in results
    
    # Verify reasonable values
    assert results["moves_played"] >= 0
    assert 0 <= results["prediction_accuracy"] <= 100
    assert results["total_time"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

