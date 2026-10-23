import random
import chess
import chess.polyglot
from typing import Optional

# Model
from modelComponent.moveCommand import MoveCommand, move_command_to_uci
from modelComponent.openingMoveNodeProtocal import OpeningMoveNodeProtocal

# Enum
from appEnums import MoveCommandType

class PolyglotOpeningHandbook(OpeningMoveNodeProtocal):
    def __init__(self, book_path: str, chessboard_model_ref: any, internal_board: Optional[chess.Board] = None):
        self.book_path = book_path
        # Keep a reference to translate UCI strings to your custom MoveCommand object
        self.fe_board_model = chessboard_model_ref 
        # State tracker to match the Polyglot Zobrist hashes
        self.board = internal_board if internal_board is not None else chess.Board()

    def stepForward(self, cmd: MoveCommand) -> Optional['OpeningMoveNodeProtocal']:
        """Applies the executed move to our opening state tracker."""
        if not cmd or cmd.moveType == MoveCommandType.NULL:
            return self

        try:
            # 1. Convert your frontend command into a UCI string format
            uci_str = move_command_to_uci(cmd)
            # 2. Convert UCI to python-chess Move object
            move = chess.Move.from_uci(uci_str)
            
            # 3. If it's a legal book move, advance our internal state
            if move in self.board.legal_moves:
                new_board = self.board.copy()
                new_board.push(move)
                
                # Check if the new position actually exists in our handbook data
                with chess.polyglot.open_reader(self.book_path) as reader:
                    if list(reader.find_all(new_board)):
                        return PolyglotOpeningHandbook(self.book_path, self.fe_board_model, new_board)
            
            # If the move falls out of book theory, dissolve the book tracker
            return None
        except Exception:
            return None

    def hasSubsequentCmd(self) -> bool:
        """Checks if the current position has recorded variations inside the book file."""
        try:
            with chess.polyglot.open_reader(self.book_path) as reader:
                # If entries list is empty, we are out of book theory
                return len(list(reader.find_all(self.board))) > 0
        except FileNotFoundError:
            return False

    def randomSubsequentCmd(self) -> MoveCommand:
        """Selects a subsequent book option weighted by popularity."""
        try:
            with chess.polyglot.open_reader(self.book_path) as reader:
                entries = list(reader.find_all(self.board))
                
                if not entries:
                    raise ValueError("No subsequent book moves available.")
                    
                # Extract the underlying chess.Move object and weight from the polyglot entry
                moves = [e.move for e in entries]
                weights = [e.weight for e in entries]
                
                # Weighted random selection returns a list -> grab index 0
                chosen_move = random.choices(moves, weights=weights, k=1)[0]
                
                # Convert python-chess Move object to UCI string ("e2e4")
                uci_move = chosen_move.uci()
                
                # Translate UCI string into your local frontend MoveCommand format
                return self.fe_board_model.uci_to_move_command(uci_move)
        except FileNotFoundError:
            raise FileNotFoundError(f"Opening handbook file not found at: {self.book_path}")
