"""
Speculator: Fast predictive agent for chess moves.
"""
from typing import List, Optional
import chess
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.chess_environment import GameState, ChessEnvironment


class Speculator:
    """Fast predictive agent that guesses likely chess moves."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",  # Faster, cheaper model
        temperature: float = 0.3,  # Lower temperature for more predictable outputs
        max_tokens: int = 512,  # Fewer tokens for speed
        reasoning_effort: str = "low"
    ):
        """Initialize Speculator.
        
        Args:
            model_name: Faster LLM model to use
            temperature: Lower temperature for more consistent predictions
            max_tokens: Fewer tokens for faster responses
            reasoning_effort: "low" for quick predictions
        """
        self.model = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        self.reasoning_effort = reasoning_effort
        self.environment = ChessEnvironment()
    
    def get_system_prompt(self) -> str:
        """Get system prompt optimized for fast prediction."""
        return """You are a chess move predictor. Your goal is to quickly predict 
the most likely moves a strong player would make in this position.

Respond with the top K most likely moves in UCI format, one per line, 
ordered by likelihood. For example:
e2e4
d2d4
g1f3

Be fast and concise. Focus on common opening moves and natural continuations."""
    
    async def predict_moves(self, state: GameState, k: int = 3) -> List[chess.Move]:
        """Predict top-k most likely moves.
        
        Args:
            state: Current game state
            k: Number of predictions to return
            
        Returns:
            List of predicted moves, ordered by confidence
        """
        prompt = self.environment.format_state_for_prompt(state)
        prompt += f"\nPredict the top {k} most likely moves:"
        
        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=prompt)
        ]
        
        response = await self.model.ainvoke(messages)
        text = response.content
        
        # Extract moves from response
        moves = self._extract_moves(text, state, k)
        
        return moves
    
    def _extract_moves(self, text: str, state: GameState, k: int) -> List[chess.Move]:
        """Extract up to k moves from response text."""
        import re
        
        # Pattern for UCI moves
        pattern = r'\b([a-h][1-8][a-h][1-8])\b'
        matches = re.findall(pattern, text.lower())
        
        moves = []
        legal_moves = set(state.get_legal_moves())
        
        for match in matches:
            move = self.environment.parse_move(match, state)
            if move and move in legal_moves and move not in moves:
                moves.append(move)
                if len(moves) >= k:
                    break
        
        # If we don't have enough moves, fill with legal moves
        if len(moves) < k:
            for move in legal_moves:
                if move not in moves:
                    moves.append(move)
                    if len(moves) >= k:
                        break
        
        return moves[:k]
    
    async def predict_next_api_calls(
        self, 
        state: GameState, 
        predicted_move: chess.Move,
        k: int = 3
    ) -> List[tuple[str, dict]]:
        """Predict next API calls after a predicted move.
        
        Args:
            state: Current game state
            predicted_move: The move we're speculating on
            k: Number of next moves to predict
            
        Returns:
            List of (handler_name, parameters) tuples for next API calls
        """
        # Apply predicted move to get next state
        next_state = state.apply_move(predicted_move)
        
        # Predict moves from the next state
        next_moves = await self.predict_moves(next_state, k)
        
        # Construct API calls for each predicted next move
        api_calls = []
        for move in next_moves:
            handler_name = "decide_move"
            params = {
                "state_fen": next_state.to_fen(),
                "prompt": self.environment.format_state_for_prompt(next_state),
                "reasoning_effort": "low"
            }
            api_calls.append((handler_name, params))
        
        return api_calls

