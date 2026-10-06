# Import Model
from modelComponent.moveCommand import MoveCommand
from modelComponent.chessPieceModel import ChessPieceModel
from modelComponent.chessBoardProtocal import ChessBoardProtocal

# Enum
from appEnums import Player, MoveCommandType, PieceType

class KnightModel(ChessPieceModel):
	def __init__(self, player: Player, row: int, col: int):

		self.player = player
		self.row = row
		self.col = col
		self.type = PieceType.KNIGHT
		self.moves = 0

	def possibleMoves(self, chessBoard: ChessBoardProtocal) -> list[MoveCommand]:
		returnMoves = []
		
		for dr, dc in [(2, 1), (1, 2), (-2, -1), (-1, -2), (-2, 1), (-1, 2), (2, -1), (1, -2)]:
			newRow = self.row + dr
			newCol = self.col + dc
			if newRow >= 0 and newRow < 8 and newCol >= 0 and newCol < 8:
				if chessBoard.board[newRow][newCol] == None:
					returnMoves.append(
						MoveCommand(self.row, self.col, newRow, newCol, MoveCommandType.MOVE)
					)
				elif chessBoard.board[newRow][newCol].player != self.player:
					returnMoves.append(
						MoveCommand(self.row, self.col, newRow, newCol, MoveCommandType.CAPTURE)
					)

		# Validate For King Safety
		return [move for move in returnMoves if chessBoard.validateKingSafety(move)]

	def captureTargets(self, chessBoard: ChessBoardProtocal) -> list[tuple[int, int]]:
		returnMoves = []

		for dr, dc in [(2, 1), (1, 2), (-2, -1), (-1, -2), (-2, 1), (-1, 2), (2, -1), (1, -2)]:
			newRow = self.row + dr
			newCol = self.col + dc
			if newRow >= 0 and newRow < 8 and newCol >= 0 and newCol < 8:
				returnMoves.append((newRow, newCol))

		return returnMoves

