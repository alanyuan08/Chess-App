from PySide6.QtCore import Signal, QObject
from PySide6.QtWidgets import QWidget, QGraphicsScene, QGraphicsView, \
	QApplication, QGraphicsItem, QGraphicsPixmapItem,QGraphicsTextItem, QGraphicsSceneMouseEvent
from PySide6.QtGui import QPixmap
import sys

# Import Enums
from appEnums import PieceType, Player, MoveCommandType

# Import Model
from modelComponent.chessPieceModel import ChessPieceModel
from modelComponent.chessBoardModel import ChessBoardModel
from modelComponent.moveCommand import MoveCommand

# Import View
from viewComponent.chessPieceView import ChessPieceView

# Import ViewModel
from viewModelComponent.chessBoardViewModel import ChessBoardViewModel

# QGraphics for ChessBoard
class PlayerInfo(QGraphicsPixmapItem):
	def __init__(self, player: Player, playerName: str, parent):
		super().__init__(parent) 

		self.textItem = QGraphicsTextItem(self)
		self.playerTurn = False
		self.player = player
		self.playerName = playerName

		padding = 10
		if player == Player.WHITE:
			self.setPos(0 + padding, 720 + padding)
			self.playerTurn = True
		else:
			self.setPos(0 + padding, -60 + padding)

		newText = self.playerName + " | "
		if self.playerTurn:
			newText += "Player Turn"
		self.textItem.setHtml(
				"<h1 style='color: white;'>" + newText + "</h1>"
			)

	def updateTurn(self, playerTurn: bool):
		self.playerTurn = playerTurn

		newText = self.playerName + " | "
		if self.playerTurn:
			newText += "Player Turn"

		self.textItem.setHtml(
			"<h1 style='color: white;'>" + newText + "</h1>"
		)

