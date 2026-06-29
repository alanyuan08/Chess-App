# Model
from modelComponent.chessBoardModel import ChessBoardModel
from modelComponent.moveCommand import MoveCommand
from modelComponent.openingMoveProtocal import OpeningMoveNodeProtocal

# Enum
from appEnums import Player, GameState, MoveCommandType

# Multi Process
import rust_compute

# Controller 
class ChessGameModel():
    def __init__(self, humanPlayers: list[Player], chessBoard: list[list[ChessBoardModel]], 
        openingHandBook: OpeningMoveNodeProtocal):
        self.chessBoard = chessBoard

        # Set Human Player
        self.humanPlayers = humanPlayers

        # Game Turn - Chess Board Turn may be different due to backtracking
        self.gamePlayerTurn = Player.WHITE

        # Set Player Lose
        self.gameState = GameState.PLAYING

        # Opening Handbook - Node Represents Current Move
        self.currOpeningMove = openingHandBook

        # Rust Chess Engine
        self.game_engine = rust_compute.ChessGame()
    
    # Return Moves in UCI for Rust Computations
    def returnChessMoves(self) -> list[str]:
        return self.chessBoard.previousMoves

    # Move Piece
    def movePiece(self, cmd: MoveCommand):
        # Move the Chess Piece
        if cmd and cmd.moveType != MoveCommandType.NULL: 
            self.chessBoard.movePiece(cmd)

        if self.currOpeningMove:
            self.currOpeningMove = self.currOpeningMove.stepForward(cmd)

        # Compute EndGame
        self.gamePlayerTurn = ChessBoardModel.opponent(self.gamePlayerTurn)

        # Three Move Repetition Draw
        if self.chessBoard.checkThreeMoveRepetiton():
            self.gameState = GameState.DRAW
            return

        # No Moves - Determine Checkmate or Draw
        if len(self.chessBoard.allValidMoves()) == 0:
            if self.gamePlayerTurn == Player.WHITE:
                if self.chessBoard.inCheck():
                    self.gameState = GameState.BLACKWIN
                else:
                    self.gameState = GameState.DRAW
            else:
                if self.chessBoard.inCheck():
                    self.gameState = GameState.WHITEWIN
                else:
                    self.gameState = GameState.DRAW

    # Validate Move
    def validateMove(self, initRow: int, initCol: int, targetRow: int, 
        targetCol: int, player: Player) -> MoveCommand:
        # It's not your turn to move
        if player != self.gamePlayerTurn:
            return None

        return self.chessBoard.validateMove(initRow, initCol, targetRow, targetCol, player)

    # Take Opponent Turn
    def computeBestMove(self) -> MoveCommand:
        if self.currOpeningMove and self.currOpeningMove.hasSubsequentCmd():
            return self.currOpeningMove.randomSubsequentCmd()
        
        # Rust Compute Next Move
        uci_move = self.game_engine.compute_next_move(self.returnChessMoves())
        print(uci_move)
        return self.chessBoard.uci_to_move_command(uci_move)
