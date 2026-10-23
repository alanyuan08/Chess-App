# Enum
from appEnums import Player

# Import Factory
from modelFactory.chessBoardFactory import ChessBoardFactory

# Import Models
from modelComponent.chessGameModel import ChessGameModel

# Import Opening Handbook Adapter
from modelComponent.polyglotOpeningHandbook import PolyglotOpeningHandbook

class ChessGameFactory:

    @staticmethod
    def createChessGame(humanPlayers: list[Player], book_path: str = "performance.bin") -> ChessGameModel:

        # 1. Create Chess Board Layout
        chessBoard = ChessBoardFactory.createChessBoard(humanPlayers)

        # 2. Initialize the Open-Source Polyglot Opening Rulebook/Handbook
        opening_handbook = PolyglotOpeningHandbook(book_path=book_path, chessboard_model_ref=chessBoard)

        # 3. Create New Game Model with the Handbook Tracker included
        newGameModel = ChessGameModel(
            humanPlayers=humanPlayers, 
            chessBoard=chessBoard, 
            openingHandBook=opening_handbook
        )

        return newGameModel
