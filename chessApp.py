from PySide6.QtWidgets import QGraphicsScene, QGraphicsView, QApplication
import sys

# Import Enums
from appEnums import Player

# Import View
from viewComponent.chessBoardView import ChessBoardView

# Import ViewModel
from viewModelComponent.chessBoardViewModel import ChessBoardViewModel

# Import Model
from modelComponent.chessGameModel import ChessGameModel

if __name__ == '__main__':
	app = QApplication(sys.argv)

	humanPlayer = [Player.BLACK, Player.WHITE]

	# Set Human Player
	if len(sys.argv) > 1:
	    match(sys.argv[1]):
	    	case "white":
	    		humanPlayer = [Player.WHITE]
	    	case "black":
	    		humanPlayer = [Player.BLACK]

	# ChessBoard View
	scene = QGraphicsScene(0, 0, 720, 840)
	view = QGraphicsView(scene)
	view.setFixedSize(720, 840)

	# Create Chess Game Moded / ChessBoard/ Chess Piece Model Components
	chessGameModel = ChessGameModel(humanPlayer)

	# Create View Components
	chessBoardView = ChessBoardView(scene, chessGameModel)

	# Create ChessBoard ViewModel
	chessBoardController = ChessBoardViewModel(chessBoardView, chessGameModel)

	view.show()
	app.exec()
