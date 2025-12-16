from PySide6.QtWidgets import QGraphicsScene, QGraphicsView, QApplication
import sys

# Import Enums
from controllerComponent.chessController import ChessBoardController

# Import Model
from modelComponent.chessPieceModel import ChessPieceModel

# Import View
from viewComponent.chessPieceView import ChessPieceView
from viewComponent.chessBoardView import ChessBoardView

if __name__ == '__main__':
	app = QApplication(sys.argv)

	# Init Scene + View
	scene = QGraphicsScene(0, 0, 720, 720)
	view = QGraphicsView(scene)
	view.setFixedSize(720, 720)

	# Create Model Componens

	# Create View Components
	chessBoard = ChessBoardView(scene)

	for property in ChessPieceModel.returnChessPieceProperties():
		row = property[0]
		col = property[1]
		player = property[2]
		type = property[3]

		ChessPieceView(player, type, row, col, chessBoard, controller)

	view.show()
	app.exec()
