from PySide6.QtCore import Signal, QObject
from PySide6.QtWidgets import QWidget, QGraphicsScene, QGraphicsView, QApplication, QGraphicsItem, QGraphicsPixmapItem, QGraphicsSceneMouseEvent
from PySide6.QtGui import QPixmap
import sys

# Import Enums
from appEnums import PieceType, Player, MoveCommandType
from communicatorProxy import MoveCommand, CommunicatorProxy

# Import Model
from modelComponent.chessPieceModel import ChessPieceModel
from modelComponent.chessBoardModel import ChessBoardModel

# Import View
from viewComponent.chessPieceView import ChessPieceView

# Import ViewModel
from viewModelComponent.chessBoardViewModel import ChessBoardViewModel

# QGraphics for ChessBoard
class ChessBoardView(QGraphicsPixmapItem):
	def __init__(self, scene: QGraphicsScene, chessBoard: ChessBoardModel):
		pixmap = QPixmap("img/chessBackground.jpg")
		super().__init__(pixmap.scaled(720, 720), None)

		pixmapItem = scene.addItem(self)
		self.setPos(0, 0)
		self.scene = scene

		# Create Chess Piece Views
		for row in range(0, 8):
			for col in range(0, 8):
				pieceModel = chessBoard.board[row][col]
				if pieceModel != None:
					humanPlayer = False
					if pieceModel.player in chessBoard.humanPlayers:
						humanPlayer = True
					print(humanPlayer)
					ChessPieceView(row, col, pieceModel, humanPlayer, self)

	def connectViewModel(self, chessBoardViewModel: ChessBoardViewModel):
		for item in self.childItems():
			item.connectViewModel(chessBoardViewModel)

	def deletePieceAtLocation(self, row: int, col: int):
		for item in self.childItems():
			if item.row == row and item.col == col:
				self.scene.removeItem(item)

	def promotePiece(self, row: int, col: int):
		for item in self.childItems():
			if item.row == row and item.col == col:
				item.setPixmap(QPixmap(ChessPieceView.returnImageURL(item.player, PieceType.QUEEN)))

	def movePieceToLocation(self, startRow: int, startCol: int, endRow: int, endCol: int):
		for item in self.childItems():
			if item.row == startRow and item.col == startCol:
				item.row = endRow
				item.col = endCol
				# Set the row / col positions 
				xCoordinate = item.col * 90
				yCoordinate = (7 - item.row) * 90
				item.setPos(xCoordinate, yCoordinate)

	# The backend is responsible for checking game logic
	def updatePosition(self, cmd: MoveCommand):
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
				if cmd.player == Player.BLACK:
					self.movePieceToLocation(7, 0, 7, 3)
					self.movePieceToLocation(7, 4, 7, 2)
				elif cmd.player == Player.WHITE:
					self.movePieceToLocation(0, 0, 0, 3)
					self.movePieceToLocation(0, 4, 0, 2)

			# King Side Castle
			case MoveCommandType.KINGSIDECASTLE:
				if cmd.player == Player.BLACK:
					self.movePieceToLocation(7, 7, 7, 5)
					self.movePieceToLocation(7, 4, 7, 6)
				elif cmd.player == Player.WHITE:
					self.movePieceToLocation(0, 7, 0, 5)
					self.movePieceToLocation(0, 4, 0, 6)
