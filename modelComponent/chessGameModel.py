# Import Factory
from modelFactory.chessBoardFactory import ChessBoardFactory

# Model
from modelComponent.chessBoardModel import ChessBoardModel
from modelComponent.moveCommand import MoveCommand

# Factory
from modelFactory.chessPieceFactory import ChessPieceFactory

# Enum
from appEnums import PieceType, Player, MoveCommandType

# Controller 
class ChessGameModel():
    def __init__(self, playersArray):
        self.chessBoard = ChessBoardFactory.createChessBoard(playersArray)

        # Set Human Player
        self.humanPlayers = playersArray

        # Set Game Turn 
        self.gamePlayerTurn = Player.WHITE

    # Move Piece
    def movePiece(self, cmd: MoveCommand):
        self.chessBoard.movePiece(cmd)

        self.gamePlayerTurn = ChessBoardModel.opponent(self.gamePlayerTurn)

    # Validate Move
    def validateMove(self, initRow: int, initCol: int, targetRow: int, 
        targetCol: int, player: Player) -> MoveCommand:
        # It's not your turn to move
        if player != self.gamePlayerTurn:
            return None

        return self.chessBoard.validateMove(initRow, initCol, targetRow, targetCol, player)

    # Take Opponent Turn
    def computeBestMove(self) -> MoveCommand:
        commandList = self.chessBoard.allValidMoves()
        commandList.sort(key=lambda move: self.chessBoard._getMovePriority(move), reverse=True)

        alpha = float('-inf')
        beta = float('inf')

        bestScore = float('-inf')
        bestMove = None

        for cmd in commandList:
            removedPiece, prevEnPassant, prevCastleIndex = self.chessBoard.movePiece(cmd)
            score = (-1) * self.chessBoard._negamax(3, (-1) * beta, (-1) * alpha)
            self.chessBoard.undoMove(cmd, removedPiece, prevEnPassant, prevCastleIndex)

            print(cmd)
            if score > bestScore:
                bestScore = score
                bestMove = cmd

            alpha = max(alpha, score)

        return bestMove

