# Enum
from appEnums import PieceType, Player, MoveCommandType

# Import Factory
from modelFactory.chessBoardFactory import ChessBoardFactory

# Import Models
from modelComponent.openingMoveNode import OpeningMoveCmd
from modelComponent.chessGameModel import ChessGameModel

class ChessGameFactory:

    @staticmethod
    def createChessGame(humanPlayers: list[Player]) -> ChessGameModel:

        # Create Chess Board
        chessBoard = ChessBoardFactory.createChessBoard(humanPlayers)

        # Create New Game Model
        newGameModel = ChessGameModel(humanPlayers, chessBoard, OpeningMoveCmd)

        return newGameModel
