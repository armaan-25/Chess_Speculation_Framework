"""
Chess environment for managing game state and moves.
"""
import chess
import chess.engine
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class GameState:
    """Represents the current state of a chess game."""
    board: chess.Board
    turn: int  # Current turn number
    current_player: chess.Color  # Current player to move
    
    def copy(self) -> 'GameState':
        """Create a deep copy of the game state."""
        return GameState(
            board=self.board.copy(),
            turn=self.turn,
            current_player=self.current_player
        )
    
    def apply_move(self, move: chess.Move) -> 'GameState':
        """Apply a move and return new state."""
        new_board = self.board.copy()
        new_board.push(move)
        return GameState(
            board=new_board,
            turn=self.turn + 1,
            current_player=not self.current_player
        )
    
    def is_terminal(self) -> bool:
        """Check if the game is over."""
        return self.board.is_game_over()
    
    def get_legal_moves(self) -> list[chess.Move]:
        """Get all legal moves for the current player."""
        return list(self.board.legal_moves)
    
    def to_fen(self) -> str:
        """Get FEN representation of the board."""
        return self.board.fen()
    
    def to_uci(self, move: chess.Move) -> str:
        """Convert move to UCI format."""
        return move.uci()


class ChessEnvironment:
    """Manages chess game environment and state transitions."""
    
    def __init__(self, initial_fen: Optional[str] = None):
        """Initialize chess environment.
        
        Args:
            initial_fen: Optional FEN string for initial position
        """
        if initial_fen:
            self.board = chess.Board(fen=initial_fen)
        else:
            self.board = chess.Board()
        
        self.initial_state = GameState(
            board=self.board.copy(),
            turn=0,
            current_player=chess.WHITE
        )
    
    def get_state(self) -> GameState:
        """Get current game state."""
        return GameState(
            board=self.board.copy(),
            turn=len(self.board.move_stack),
            current_player=self.board.turn
        )
    
    def transition(self, state: GameState, action: chess.Move) -> GameState:
        """Apply action to state and return new state.
        
        Args:
            state: Current game state
            action: Move to apply
            
        Returns:
            New game state after applying move
        """
        if action not in state.get_legal_moves():
            raise ValueError(f"Illegal move: {action}")
        
        return state.apply_move(action)
    
    def parse_move(self, move_str: str, state: Optional[GameState] = None) -> Optional[chess.Move]:
        """Parse move string to chess.Move.
        
        Args:
            move_str: Move in UCI format (e.g., "e2e4")
            state: Optional state to validate move against
            
        Returns:
            chess.Move if valid, None otherwise
        """
        try:
            move = chess.Move.from_uci(move_str)
            if state:
                if move in state.get_legal_moves():
                    return move
                return None
            return move
        except ValueError:
            return None
    
    def format_state_for_prompt(self, state: GameState) -> str:
        """Format game state for LLM prompt."""
        board_str = str(state.board)
        turn_info = "White" if state.current_player == chess.WHITE else "Black"
        move_num = state.turn + 1
        
        prompt = f"""Current position (Move {move_num}, {turn_info} to move):

{board_str}

FEN: {state.to_fen()}

Legal moves: {', '.join([m.uci() for m in state.get_legal_moves()[:10]])}
"""
        return prompt

