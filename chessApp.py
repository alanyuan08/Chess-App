from PySide6.QtWidgets import QGraphicsScene, QGraphicsView, QApplication
import sys

# Import Enums
from controllerComponent.chessController import ChessBoardController

# Import Model
from modelComponent.chessBoardModel import ChessBoardModel

# Import View
from viewComponent.chessBoardView import ChessBoardView

if __name__ == '__main__':
	app = QApplication(sys.argv)

	# Init Scene + View
	scene = QGraphicsScene(0, 0, 720, 720)
	view = QGraphicsView(scene)
	view.setFixedSize(720, 720)

	# Create ChessBoard/ Chess Piece Model Components
	chessBoardModel = ChessBoardModel()

	# Create View Components
	chessBoardView = ChessBoardView(scene, chessBoardModel)

	# Create ChessBoard Controller
	chessBoardController = ChessBoardController();

	# Add Chess Board to Controller
	chessBoardController.addChessBoard(chessBoardView, chessBoardModel)
	chessBoardView.updateController(chessBoardController)

	view.show()
	app.exec()
