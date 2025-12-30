from PySide6.QtWidgets import QGraphicsScene, QGraphicsView, QApplication
import sys

# Import Enums
from appEnums import Player

# Import Factory
from modelFactory.chessBoardFactory import ChessBoardFactory

# Import View
from viewComponent.chessBoardView import ChessBoardView

# Import ViewModel
from viewModelComponent.chessBoardViewModel import ChessBoardViewModel

if __name__ == '__main__':
	sys.setrecursionlimit(3000)

	app = QApplication(sys.argv)

	# Init Scene + View
	scene = QGraphicsScene(0, 0, 720, 720)
	view = QGraphicsView(scene)
	view.setFixedSize(720, 720)

	# Create ChessBoard/ Chess Piece Model Components
	chessBoardModel = ChessBoardFactory.createChessBoard([Player.WHITE])

	# Create View Components
	chessBoardView = ChessBoardView(scene, chessBoardModel)

	# Create ChessBoard ViewModel
	chessBoardController = ChessBoardViewModel(chessBoardView, chessBoardModel);

	view.show()
	app.exec()
