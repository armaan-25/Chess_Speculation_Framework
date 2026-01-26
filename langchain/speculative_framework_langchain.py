"""
Core speculative actions framework with caching and parallel execution.
"""
from typing import Dict, List, Optional, Tuple, Any
import asyncio
import chess
from dataclasses import dataclass
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.chess_environment import GameState
from langchain.actor_langchain import Actor
from langchain.speculator_langchain import Speculator


@dataclass
class PendingAction:
    """Represents a pending API call."""
    handler: str
    params: dict
    future: asyncio.Future
    is_speculative: bool = False


class SpeculativeCache:
    """Cache for speculative API calls."""
    
    def __init__(self):
        self._cache: Dict[Tuple[str, str], PendingAction] = {}
    
    def _make_key(self, handler: str, params: dict) -> Tuple[str, str]:
        """Create cache key from handler and params."""
        # Use FEN as key component for chess moves
        fen = params.get("state_fen", "")
        return (handler, fen)
    
    def get(self, handler: str, params: dict) -> Optional[PendingAction]:
        """Get cached pending action if exists."""
        key = self._make_key(handler, params)
        return self._cache.get(key)
    
    def put(self, handler: str, params: dict, pending_action: PendingAction):
        """Cache a pending action."""
        key = self._make_key(handler, params)
        self._cache[key] = pending_action
    
    def clear(self):
        """Clear all cached entries."""
        self._cache.clear()


class SpeculativeFramework:
    """Main framework for speculative actions."""
    
    def __init__(
        self,
        actor: Actor,
        speculator: Speculator,
        k: int = 3  # Number of speculative predictions
    ):
        """Initialize speculative framework.
        
        Args:
            actor: Slow, authoritative actor
            speculator: Fast predictive speculator
            k: Number of parallel speculative branches
        """
        self.actor = actor
        self.speculator = speculator
        self.k = k
        self.cache = SpeculativeCache()
    
    async def execute_api_call(self, handler: str, params: dict, is_speculative: bool = False) -> Any:
        """Execute an API call (either cached or new).
        
        Args:
            handler: API handler name
            params: Parameters for the API call
            is_speculative: Whether this is a speculative call
            
        Returns:
            Result of the API call
        """
        # Check cache first
        cached = self.cache.get(handler, params)
        if cached:
            # Wait for cached future if not already done
            return await cached.future
        
        # Create new API call
        future = asyncio.create_task(self._call_handler(handler, params))
        pending_action = PendingAction(
            handler=handler,
            params=params,
            future=future,
            is_speculative=is_speculative
        )
        
        self.cache.put(handler, params, pending_action)
        return await future
    
    def pre_launch_speculative_call(self, handler: str, params: dict) -> Optional[PendingAction]:
        """Pre-launch a speculative API call without awaiting it.
        
        Args:
            handler: API handler name
            params: Parameters for the API call
            
        Returns:
            PendingAction if launched, None if already cached
        """
        # Check cache first
        cached = self.cache.get(handler, params)
        if cached:
            return cached
        
        # Create speculative API call
        future = asyncio.create_task(self._call_handler(handler, params))
        pending_action = PendingAction(
            handler=handler,
            params=params,
            future=future,
            is_speculative=True
        )
        
        self.cache.put(handler, params, pending_action)
        return pending_action
    
    async def _call_handler(self, handler: str, params: dict) -> Any:
        """Call the actual handler."""
        if handler == "decide_move":
            # Reconstruct state from FEN
            from core.chess_environment import ChessEnvironment
            env = ChessEnvironment(initial_fen=params["state_fen"])
            state = env.get_state()
            
            move, reasoning = await self.actor.decide_move(state)
            return {
                "move": move,
                "reasoning": reasoning,
                "move_uci": move.uci()
            }
        else:
            raise ValueError(f"Unknown handler: {handler}")
    
    def validate_prediction(
        self, 
        predicted_move: chess.Move, 
        actual_move: chess.Move
    ) -> bool:
        """Validate if predicted move matches actual move.
        
        Args:
            predicted_move: Move predicted by speculator
            actual_move: Move returned by actor
            
        Returns:
            True if prediction matches actual move
        """
        return predicted_move == actual_move
    
    async def cancel_speculative_actions(self, actions: List[PendingAction]):
        """Cancel speculative actions that didn't match.
        
        Args:
            actions: List of speculative actions to cancel
        """
        for action in actions:
            if action.is_speculative and not action.future.done():
                action.future.cancel()
                try:
                    await action.future
                except asyncio.CancelledError:
                    pass

