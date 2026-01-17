from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsTextItem

# Import Enums
from appEnums import PieceType, Player

# Import Model
from modelComponent.chessPieceModel import ChessPieceModel
from modelComponent.chessBoardModel import ChessBoardModel
from modelComponent.moveCommand import MoveCommand

# Import View
from viewComponent.chessPieceView import ChessPieceView


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

	def playerLoss(self, playerLoss: bool):
		newText = self.playerName + " | "
		if playerLoss:
			newText += "Player Lose"
		else:
			newText += "Player Win"

		self.textItem.setHtml(
			"<h1 style='color: white;'>" + newText + "</h1>"
		)

