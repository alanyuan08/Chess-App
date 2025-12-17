from PySide6.QtWidgets import QGraphicsScene, QGraphicsView, QApplication
import sys

# Import Enums
from appEnums import Player

# Import Model
from modelComponent.chessBoardModel import ChessBoardModel

# Import View
from viewComponent.chessBoardView import ChessBoardView

# Import ViewModel
from viewModelComponent.chessBoardViewModel import ChessBoardViewModel

if __name__ == '__main__':
	app = QApplication(sys.argv)

	# Init Scene + View
	scene = QGraphicsScene(0, 0, 720, 720)
	view = QGraphicsView(scene)
	view.setFixedSize(720, 720)

	# Create ChessBoard/ Chess Piece Model Components
	chessBoardModel = ChessBoardModel([Player.WHITE])

	# Create View Components
	chessBoardView = ChessBoardView(scene, chessBoardModel)

	# Create ChessBoard ViewMoidel
	chessBoardController = ChessBoardViewModel(chessBoardView, chessBoardModel);

	view.show()
	app.exec()
