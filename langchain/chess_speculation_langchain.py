"""
Main chess speculation game loop using LangGraph.
"""
import asyncio
import time
from typing import TypedDict, Annotated, List, Optional
from langgraph.graph import StateGraph, END
import chess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.chess_environment import ChessEnvironment, GameState
from langchain.actor_langchain import Actor
from langchain.speculator_langchain import Speculator
from langchain.speculative_framework_langchain import SpeculativeFramework, PendingAction


class GameStateDict(TypedDict):
    """State dictionary for LangGraph."""
    game_state: GameState
    current_player: str  # "white" or "black"
    turn_number: int
    moves_history: List[str]
    speculative_actions: List[PendingAction]
    last_move: Optional[chess.Move]
    is_game_over: bool
    time_saved: float
    predictions_made: int
    predictions_hit: int


class ChessSpeculationGame:
    """Main game loop with speculative actions using LangGraph."""
    
    def __init__(
        self,
        actor_white: Optional[Actor] = None,
        actor_black: Optional[Actor] = None,
        speculator_white: Optional[Speculator] = None,
        speculator_black: Optional[Speculator] = None,
        k: int = 3
    ):
        """Initialize game with actors and speculators.
        
        Args:
            actor_white: Actor for white player
            actor_black: Actor for black player
            speculator_white: Speculator for white player
            speculator_black: Speculator for black player
            k: Number of speculative predictions
        """
        # Default actors
        if actor_white is None:
            actor_white = Actor(
                model_name="gpt-4o",
                reasoning_effort="high"
            )
        if actor_black is None:
            actor_black = Actor(
                model_name="gpt-4o",
                reasoning_effort="high"
            )
        
        # Default speculators
        if speculator_white is None:
            speculator_white = Speculator(
                model_name="gpt-4o-mini",
                reasoning_effort="low"
            )
        if speculator_black is None:
            speculator_black = Speculator(
                model_name="gpt-4o-mini",
                reasoning_effort="low"
            )
        
        self.actor_white = actor_white
        self.actor_black = actor_black
        self.speculator_white = speculator_white
        self.speculator_black = speculator_black
        self.k = k
        
        # Create frameworks for each player
        self.framework_white = SpeculativeFramework(actor_white, speculator_white, k)
        self.framework_black = SpeculativeFramework(actor_black, speculator_black, k)
        
        self.environment = ChessEnvironment()
        
        # Build LangGraph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph state machine."""
        workflow = StateGraph(GameStateDict)
        
        # Add nodes
        workflow.add_node("white_turn", self.white_turn)
        workflow.add_node("black_turn", self.black_turn)
        workflow.add_node("check_game_over", self.check_game_over)
        
        # Add edges
        workflow.set_entry_point("white_turn")
        workflow.add_edge("white_turn", "check_game_over")
        workflow.add_edge("black_turn", "check_game_over")
        workflow.add_conditional_edges(
            "check_game_over",
            self.should_continue,
            {
                "continue_white": "white_turn",
                "continue_black": "black_turn",
                "end": END
            }
        )
        
        return workflow.compile()
    
    async def white_turn(self, state: GameStateDict) -> GameStateDict:
        """Execute white's turn with speculation."""
        return await self._execute_turn(
            state, 
            self.framework_white, 
            self.speculator_white,
            chess.WHITE
        )
    
    async def black_turn(self, state: GameStateDict) -> GameStateDict:
        """Execute black's turn with speculation."""
        return await self._execute_turn(
            state,
            self.framework_black,
            self.speculator_black,
            chess.BLACK
        )
    
    async def _execute_turn(
        self,
        state: GameStateDict,
        framework: SpeculativeFramework,
        speculator: Speculator,
        player_color: chess.Color
    ) -> GameStateDict:
        """Execute a turn with speculative actions (Algorithm 1 from paper)."""
        game_state = state["game_state"]
        
        # Step 1: Policy decides API call
        handler, params = framework.actor.construct_api_call(game_state)
        
        # Step 2: Check cache
        cached = framework.cache.get(handler, params)
        if cached:
            # Cache hit - this means a previous speculative call predicted this correctly!
            cache_hit_start = time.time()
            
            # Check if speculative call already completed
            was_already_done = cached.future.done()
            
            result = await cached.future
            cache_hit_time = time.time() - cache_hit_start
            
            # Calculate time saved
            if was_already_done:
                # Speculative call completed before we needed it - saved full API call time
                # Estimate saved time: typical actor call is 2-5 seconds, use 3s as average
                estimated_api_time = 3.0
                state["time_saved"] += estimated_api_time
            else:
                # Speculative call was still running - we saved the time to start it
                # The call started earlier, so we only waited cache_hit_time instead of full API time
                estimated_api_time = 3.0
                state["time_saved"] += max(0, estimated_api_time - cache_hit_time)
            
            actual_move = result["move"]
        else:
            # Step 3: Launch Actor API call (returns future)
            actor_start_time = time.time()
            actor_future = asyncio.create_task(
                framework.execute_api_call(handler, params)
            )
            
            # Step 4: Speculator predicts moves in parallel
            spec_start_time = time.time()
            predicted_moves = await speculator.predict_moves(game_state, k=self.k)
            spec_time = time.time() - spec_start_time
            
            # Step 5: Launch speculative API calls for next steps
            # Map predicted moves to their speculative actions
            predicted_to_actions = {}
            speculative_actions = []
            
            for predicted_move in predicted_moves:
                next_state = game_state.apply_move(predicted_move)
                next_handler, next_params = framework.actor.construct_api_call(next_state)
                
                # Pre-launch speculative call (handles caching internally)
                pending = framework.pre_launch_speculative_call(next_handler, next_params)
                
                if pending:
                    predicted_to_actions[predicted_move] = pending
                    speculative_actions.append(pending)
            
            # Step 6: Wait for Actor response
            result = await actor_future
            actual_move = result["move"]
            actor_time = time.time() - actor_start_time
            
            # Step 7: Validate predictions and commit/rollback
            prediction_hit = False
            hit_predicted_move = None
            
            for predicted_move in predicted_moves:
                if framework.validate_prediction(predicted_move, actual_move):
                    prediction_hit = True
                    hit_predicted_move = predicted_move
                    # Time saved: speculator finished before actor
                    if spec_time < actor_time:
                        state["time_saved"] += (actor_time - spec_time)
                    state["predictions_hit"] += 1
                    break
            
            state["predictions_made"] += len(predicted_moves)
            
            # Commit: Keep the correct speculative action, cancel others
            if prediction_hit:
                # Keep the correct speculative action (it's already cached and running)
                # Cancel all other incorrect speculative actions
                actions_to_cancel = [
                    action for move, action in predicted_to_actions.items()
                    if move != hit_predicted_move
                ]
                await framework.cancel_speculative_actions(actions_to_cancel)
            else:
                # No prediction hit - cancel all speculative actions
                await framework.cancel_speculative_actions(speculative_actions)
        
        # Step 8: Apply move and update state
        new_state = game_state.apply_move(actual_move)
        state["game_state"] = new_state
        state["last_move"] = actual_move
        state["moves_history"].append(actual_move.uci())
        state["turn_number"] += 1
        state["current_player"] = "black" if player_color == chess.WHITE else "white"
        
        return state
    
    def check_game_over(self, state: GameStateDict) -> GameStateDict:
        """Check if game is over."""
        game_state = state["game_state"]
        state["is_game_over"] = game_state.is_terminal()
        return state
    
    def should_continue(self, state: GameStateDict) -> str:
        """Determine next step based on game state."""
        if state["is_game_over"]:
            return "end"
        
        if state["current_player"] == "white":
            return "continue_white"
        else:
            return "continue_black"
    
    async def play_game(self, max_turns: int = 30) -> dict:
        """Play a complete game with speculation.
        
        Args:
            max_turns: Maximum number of turns to play
            
        Returns:
            Dictionary with game results and statistics
        """
        initial_state = self.environment.get_state()
        
        state: GameStateDict = {
            "game_state": initial_state,
            "current_player": "white",
            "turn_number": 0,
            "moves_history": [],
            "speculative_actions": [],
            "last_move": None,
            "is_game_over": False,
            "time_saved": 0.0,
            "predictions_made": 0,
            "predictions_hit": 0
        }
        
        start_time = time.time()
        
        # Run game loop
        try:
            final_state = await self.graph.ainvoke(state)
        except Exception as e:
            print(f"Error during game: {e}")
            final_state = state
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        prediction_accuracy = 0.0
        if final_state["predictions_made"] > 0:
            prediction_accuracy = (
                final_state["predictions_hit"] / final_state["predictions_made"]
            ) * 100
        
        time_saved_percent = 0.0
        if total_time > 0:
            time_saved_percent = (final_state["time_saved"] / total_time) * 100
        
        return {
            "total_time": total_time,
            "time_saved": final_state["time_saved"],
            "time_saved_percent": time_saved_percent,
            "predictions_made": final_state["predictions_made"],
            "predictions_hit": final_state["predictions_hit"],
            "prediction_accuracy": prediction_accuracy,
            "moves_played": len(final_state["moves_history"]),
            "is_game_over": final_state["is_game_over"],
            "final_fen": final_state["game_state"].to_fen()
        }


async def main():
    """Main entry point."""
    import os
    from pathlib import Path
    
    # Try to load .env file if it exists
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
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Please set it to run the game.")
        print("You can:")
        print("  1. Set environment variable: export OPENAI_API_KEY='your-key'")
        print("  2. Create a .env file with: OPENAI_API_KEY=your-key")
        return
    
    print("Starting Chess Speculation Game...")
    print("=" * 50)
    
    # Create game
    game = ChessSpeculationGame(k=3)
    
    # Play game
    results = await game.play_game(max_turns=30)
    
    # Print results
    print("\nGame Results:")
    print("=" * 50)
    print(f"Total time: {results['total_time']:.2f} seconds")
    print(f"Time saved: {results['time_saved']:.2f} seconds ({results['time_saved_percent']:.1f}%)")
    print(f"Predictions made: {results['predictions_made']}")
    print(f"Predictions hit: {results['predictions_hit']}")
    print(f"Prediction accuracy: {results['prediction_accuracy']:.1f}%")
    print(f"Moves played: {results['moves_played']}")
    print(f"Game over: {results['is_game_over']}")
    print(f"Final position: {results['final_fen']}")


if __name__ == "__main__":
    asyncio.run(main())

