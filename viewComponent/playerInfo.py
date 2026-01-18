from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsTextItem

# Import Enums
from appEnums import PieceType, Player, GameState

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

	def updateGameState(self, gameState: GameState, playerTurn: Player):
		if playerTurn == self.player:
			self.playerTurn = True
		else:
			self.playerTurn = False

		newText = self.playerName + " | "

		if gameState == GameState.PLAYING:
			if self.playerTurn:
				newText += "Active Turn"
		elif (self.player == Player.WHITE and gameState == GameState.WHITEWIN) or \
			(self.player == Player.BLACK and gameState == GameState.BLACKWIN):
			newText += "Player Win"
		elif GameState.DRAW:
			newText += "Draw"

		self.textItem.setHtml(
			"<h1 style='color: white;'>" + newText + "</h1>"
		)

