from PySide6.QtCore import Signal, QObject
from PySide6.QtWidgets import QWidget, QGraphicsScene, QGraphicsView, \
	QApplication, QGraphicsItem, QGraphicsPixmapItem, QGraphicsSceneMouseEvent
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
from viewComponent.playerInfo import PlayerInfo

# Import ViewModel
from viewModelComponent.chessBoardViewModel import ChessBoardViewModel

# QGraphics for ChessBoard
class ChessBoardView(QGraphicsPixmapItem):
	def __init__(self, scene: QGraphicsScene, chessGameModel: ChessBoardModel):
		pixmap = QPixmap("img/chessBackground.jpg")
		super().__init__(pixmap.scaled(720, 720), None)

		pixmapItem = scene.addItem(self)
		self.setPos(0, 60)
		self.scene = scene

		# Create Chess Piece Views
		for row in range(0, 8):
			for col in range(0, 8):
				chessBoard = chessGameModel.chessBoard
				pieceModel = chessBoard.board[row][col]
				if pieceModel != None:
					humanPlayer = False
					if pieceModel.player in chessBoard.humanPlayers:
						humanPlayer = True
					ChessPieceView(row, col, pieceModel, humanPlayer, self)

		# Setup Player Sidebars
		if len(chessGameModel.humanPlayers) == 2:
			PlayerInfo(Player.WHITE, "Player 1", self)
			PlayerInfo(Player.BLACK, "Player 1", self)

		elif len(chessGameModel.humanPlayers) == 0:
			PlayerInfo(Player.WHITE, "Computer 1", self)
			PlayerInfo(Player.BLACK, "Computer 2", self)

		else:
			for player in [Player.WHITE, Player.BLACK]:
				if player in chessGameModel.humanPlayers:
					PlayerInfo(player, "Player 1", self)
				else:
					PlayerInfo(player, "Computer 1", self)

	def connectViewModel(self, chessBoardViewModel: ChessBoardViewModel):
		for item in self.childItems():
			if isinstance(item, ChessPieceView):
				item.connectViewModel(chessBoardViewModel)

	def deletePieceAtLocation(self, row: int, col: int):
		for item in self.childItems():
			if isinstance(item, ChessPieceView):
				if item.row == row and item.col == col:
					self.scene.removeItem(item)

	def promotePiece(self, row: int, col: int):
		for item in self.childItems():
			if isinstance(item, ChessPieceView):
				if item.row == row and item.col == col:
					item.setPixmap(
						QPixmap(ChessPieceView.returnQueenURL(item.player))
					)

	def movePieceToLocation(self, startRow: int, startCol: int, endRow: int, endCol: int):
		for item in self.childItems():
			if isinstance(item, ChessPieceView):
				if item.row == startRow and item.col == startCol:
					item.row = endRow
					item.col = endCol
					# Set the row / col positions 
					xCoordinate = item.col * 90
					yCoordinate = (7 - item.row) * 90
					item.setPos(xCoordinate, yCoordinate)

	def setPlayerInfoTurn(self, newPlayerTurn: Player):
		for item in self.childItems():
			if isinstance(item, PlayerInfo):
				if item.player == newPlayerTurn:
					item.updateTurn(True)
				else:
					item.updateTurn(False)

	# The backend is responsible for checking game logic
	def updatePosition(self, cmd: MoveCommand, playerTurn: Player):
		self.setPlayerInfoTurn(playerTurn)

		match cmd.moveType:
			# Move Piece
			case MoveCommandType.CAPTURE:
				self.deletePieceAtLocation(cmd.endRow, cmd.endCol)
				self.movePieceToLocation(cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

			# En Passant
			case MoveCommandType.ENPASSANT:
				self.deletePieceAtLocation(cmd.startRow, cmd.endCol)
				self.movePieceToLocation(cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

			# Move Piece
			case MoveCommandType.MOVE | MoveCommandType.PAWNOPENMOVE:
				self.movePieceToLocation(cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

			# Promote
			case MoveCommandType.PROMOTE:
				self.deletePieceAtLocation(cmd.endRow, cmd.endCol)
				self.movePieceToLocation(cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)
				self.promotePiece(cmd.endRow, cmd.endCol)

			# Queen Side Castle
			case MoveCommandType.QUEENSIDECASTLE:
				row = cmd.startRow
				self.movePieceToLocation(row, 0, row, 3)
				self.movePieceToLocation(row, 4, row, 2)

			# King Side Castle
			case MoveCommandType.KINGSIDECASTLE:
				row = cmd.startRow		
				self.movePieceToLocation(row, 7, row, 5)
				self.movePieceToLocation(row, 4, row, 6)
