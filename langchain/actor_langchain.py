"""
Actor: Slow, authoritative executor for chess moves.
"""
from typing import Optional
import chess
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.chess_environment import GameState, ChessEnvironment


class Actor:
    """Slow, authoritative agent that makes deliberate chess moves."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        reasoning_effort: str = "high"
    ):
        """Initialize Actor.
        
        Args:
            model_name: LLM model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens for response
            reasoning_effort: "high" for deep analysis, "low" for quick moves
        """
        self.model = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        self.reasoning_effort = reasoning_effort
        self.environment = ChessEnvironment()
    
    def get_system_prompt(self) -> str:
        """Get system prompt for the Actor."""
        if self.reasoning_effort == "high":
            return """You are an expert chess player analyzing positions deeply. 
You must carefully evaluate the position, consider multiple candidate moves, 
analyze tactics, and provide a well-reasoned move choice.

When responding, provide:
1. Brief analysis of the position
2. Candidate moves considered
3. Your chosen move in UCI format (e.g., "e2e4")

Respond ONLY with the move in UCI format at the end, preceded by your analysis."""
        else:
            return """You are a chess player making a move. 
Respond with your move in UCI format (e.g., "e2e4")."""
    
    async def decide_move(self, state: GameState) -> tuple[chess.Move, str]:
        """Decide on a move given the current state.
        
        Args:
            state: Current game state
            
        Returns:
            Tuple of (move, reasoning) where move is the chosen move
        """
        prompt = self.environment.format_state_for_prompt(state)
        
        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=prompt)
        ]
        
        response = await self.model.ainvoke(messages)
        reasoning = response.content
        
        # Extract move from response (look for UCI format)
        move = self._extract_move(reasoning, state)
        
        if move is None:
            # Fallback: use first legal move
            legal_moves = state.get_legal_moves()
            if legal_moves:
                move = legal_moves[0]
            else:
                raise ValueError("No legal moves available")
        
        return move, reasoning
    
    def _extract_move(self, text: str, state: GameState) -> Optional[chess.Move]:
        """Extract move from LLM response text."""
        # Try to find UCI format move (e.g., "e2e4", "e2-e4", "e2 e4")
        import re
        
        # Pattern for UCI moves: two squares (e.g., e2e4)
        patterns = [
            r'\b([a-h][1-8][a-h][1-8])\b',  # e2e4
            r'\b([a-h][1-8])[- ]([a-h][1-8])\b',  # e2-e4 or e2 e4
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                if isinstance(match, tuple):
                    move_str = ''.join(match)
                else:
                    move_str = match
                
                move = self.environment.parse_move(move_str, state)
                if move:
                    return move
        
        return None
    
    def construct_api_call(self, state: GameState) -> tuple[str, dict]:
        """Construct API call specification for the move decision.
        
        Args:
            state: Current game state
            
        Returns:
            Tuple of (handler_name, parameters) representing the API call
        """
        prompt = self.environment.format_state_for_prompt(state)
        
        return ("decide_move", {
            "state_fen": state.to_fen(),
            "prompt": prompt,
            "reasoning_effort": self.reasoning_effort
        })

