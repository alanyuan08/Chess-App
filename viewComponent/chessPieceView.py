from PySide6.QtCore import Signal, QObject
from PySide6.QtWidgets import QWidget, QGraphicsScene, QGraphicsView, QApplication, QGraphicsItem, QGraphicsPixmapItem, QGraphicsSceneMouseEvent
from PySide6.QtGui import QPixmap
import sys

# Import Enums
from appEnums import PieceType, Player, MoveCommandType
from communicatorProxy import MoveCommand, CommunicatorProxy

# Import Model
from modelComponent.chessPieceModel import ChessPieceModel

# Import Controller
from controllerComponent.chessController import ChessBoardController


class ChessPieceView(QGraphicsPixmapItem):
	def __init__(self, row, col, chessPieceModel: ChessPieceModel, parent):
		self.row = row
		self.col = col
		self.player = chessPieceModel.player

		pixmap = QPixmap(ChessPieceView.returnImageURL(self.player, chessPieceModel.type))
		super().__init__(pixmap, parent)
		self.setFlag(QGraphicsItem.ItemIsMovable)


		# Set the row / col positions 
		xCoordinate = self.col * 90
		yCoordinate = (7 - self.row) * 90
		self.setPos(xCoordinate, yCoordinate)

		# Signal proxy
		self.communicatorProxy = None

	def updateCommunicateProxy(self, controller: ChessBoardController):
		self.communicatorProxy = CommunicatorProxy()
		self.communicatorProxy.move_request.connect(controller.on_move_executed)

	@staticmethod
	def returnImageURL(player, type):

		url = "img/"
		match player:
			case Player.WHITE:
				url += "white"
			case Player.BLACK:
				url += "black"

		match type:
			case PieceType.KING:
				url += "King"
			case PieceType.QUEEN:
				url += "Queen"
			case PieceType.KNIGHT:
				url += "Knight"
			case PieceType.ROOK:
				url += "Rook"
			case PieceType.PAWN:
				url += "Pawn"
			case PieceType.BISHOP:
				url += "Bishop"

		return url + ".svg"

	def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
		super().mouseReleaseEvent(event)

		computedRow = 7 - int((self.pos().y() + 45) / 90)
		computedCol = int((self.pos().x() + 45) / 90)

		if not (computedCol < 0 or computedCol > 7 or computedRow < 0 or computedRow > 7):
			self.communicatorProxy.signal_move_request(self.row, self.col, computedRow, computedCol, self.player)

		# This is used to capture the move intention, the controller moves the object
		xCoordinate = self.col * 90
		yCoordinate = (7 - self.row) * 90
		self.setPos(xCoordinate, yCoordinate)
		event.accept()
