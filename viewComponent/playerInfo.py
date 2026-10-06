from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsTextItem

# Import Enums
from appEnums import Player, GameState

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
		elif gameState == GameState.DRAW:
			newText += "Draw"
		elif self.player == Player.WHITE:
			if gameState == GameState.WHITEWIN:
				newText += "Player Win"
			else:
				newText += "Player Lose"
		elif self.player == Player.BLACK:
			if gameState == GameState.BLACKWIN:
				newText += "Player Win"
			else:
				newText += "Player Lose"

		self.textItem.setHtml(
			"<h1 style='color: white;'>" + newText + "</h1>"
		)

