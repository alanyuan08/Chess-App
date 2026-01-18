# Import Factory
from modelFactory.chessBoardFactory import ChessBoardFactory

# Model
from modelComponent.chessBoardModel import ChessBoardModel
from modelComponent.moveCommand import MoveCommand
from modelComponent.openingMoveBook import OpeningMovebook, rootCmd

# Factory
from modelFactory.chessPieceFactory import ChessPieceFactory

# Enum
from appEnums import PieceType, Player, MoveCommandType, GameState

# Multi Process
import multiprocessing
import concurrent.futures

# Controller 
class ChessGameModel():
    def __init__(self, playersArray):
        self.chessBoard = ChessBoardFactory.createChessBoard(playersArray)

        # Set Human Player
        self.humanPlayers = playersArray

        # Game Turn - Chess Board Turn may be different due to backtracking
        self.gamePlayerTurn = Player.WHITE

        # Set Player Lose
        self.gameState = GameState.PLAYING

        # Opening HandBook
        self.openingHandBook = rootCmd

    # Move Piece
    def movePiece(self, cmd: MoveCommand):
        # Move the Chess Piece
        self.chessBoard.movePiece(cmd)

        if self.openingHandBook:
            self.openingHandBook = self.openingHandBook.stepForward(cmd)

        self.gamePlayerTurn = ChessBoardModel.opponent(self.gamePlayerTurn)

        # Set Player Loss
        if len(self.chessBoard.allValidMoves()) == 0:
            if self.gamePlayerTurn == Player.WHITE:
                self.gameState = GameState.BLACKWIN
            else:
                self.gameState = GameState.WHITEWIN

    # Validate Move
    def validateMove(self, initRow: int, initCol: int, targetRow: int, 
        targetCol: int, player: Player) -> MoveCommand:
        # It's not your turn to move
        if player != self.gamePlayerTurn:
            return None

        return self.chessBoard.validateMove(initRow, initCol, targetRow, targetCol, player)

    # Take Opponent Turn
    def computeBestMove(self) -> MoveCommand:
        if self.openingHandBook and self.openingHandBook.hasSubsequentCmd():
            return self.openingHandBook.randomSubsequentCmd()

        commandList = self.chessBoard.allValidMoves()
        commandList.sort(key=lambda move: self.chessBoard._getMovePriority(move), reverse=True)

        alpha = float('-inf')
        beta = float('inf')

        bestScore = float('-inf')
        bestMove = None

        if len(commandList) == 0:
            return None

        with multiprocessing.Manager() as manager:
            # Compute the most optimal search move
            cmd1 = commandList[0]
            removedPiece, prevEnPassant, prevCastleIndex = self.chessBoard.movePiece(cmd1)
            # Search the first move normally to get a strong alpha value quickly
            score = (-1) * self.chessBoard._negamax(4, (-1) * beta, (-1) * alpha) 
            self.chessBoard.undoMove(cmd1, removedPiece, prevEnPassant, prevCastleIndex)

            if score > bestScore:
                bestScore = score
                bestMove = cmd1
                alpha = max(alpha, score) # Establish the strong alpha

            # Younger Brother Parallel Search
            remaining_moves = commandList[1:]
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count() - 1) as executor:
                futures = [
                    executor.submit(self.chessBoard._negamax_worker, cmd, alpha, beta, 4) 
                    for cmd in remaining_moves
                ]
                
                for future in concurrent.futures.as_completed(futures):
                    move, score = future.result()
                    if score > bestScore:
                        bestScore = score
                        bestMove = move

        return bestMove

