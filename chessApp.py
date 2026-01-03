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
	sys.setrecursionlimit(3000)

	app = QApplication(sys.argv)

	# Init Scene + View
	scene = QGraphicsScene(0, 0, 720, 720)
	view = QGraphicsView(scene)
	view.setFixedSize(720, 720)

	# Create Chess Game Moded / ChessBoard/ Chess Piece Model Components
	chessGameModel = ChessGameModel([Player.WHITE, Player.BLACK])

	# Create View Components
	chessBoardView = ChessBoardView(scene, chessGameModel)

	# Create ChessBoard ViewModel
	chessBoardController = ChessBoardViewModel(chessBoardView, chessGameModel)

	view.show()
	app.exec()
