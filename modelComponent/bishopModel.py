# Import Model
from modelComponent.moveCommand import MoveCommand
from modelComponent.chessPieceModel import ChessPieceModel
from modelComponent.chessBoardProtocal import ChessBoardProtocal

# Enum
from appEnums import Player, MoveCommandType, PieceType

class BishopModel(ChessPieceModel):
	def __init__(self, player: Player, row: int, col: int):

		self.player = player
		self.row = row
		self.col = col
		self.type = PieceType.BISHOP
		self.moves = 0

	# List all Possible Moves from Location
	def possibleMoves(self, chessboard: ChessBoardProtocal) -> list[MoveCommand]:
		returnMoves = []

		for dr, dc in [(-1, 1), (1, 1), (1, -1), (-1, -1)]:
			for i in range(1, 8):
				newRow = self.row + dr * i
				newCol = self.col + dc * i

				if not (newRow >= 0 and newRow < 8 and newCol >= 0 and newCol < 8):
					break

				if chessboard.board[newRow][newCol] == None:
					returnMoves.append(
						MoveCommand(self.row, self.col, newRow, newCol, MoveCommandType.MOVE)
					)

				else:
					if chessboard.board[newRow][newCol].player != self.player:
						returnMoves.append(
							MoveCommand(self.row, self.col, newRow, newCol, MoveCommandType.CAPTURE)
						)
					break

		# Validate For King Safety
		return [move for move in returnMoves if chessboard.validateKingSafety(move)]

	# List of targets - Used to check for Castle / King Safety
	def captureTargets(self, chessBoard: ChessBoardProtocal) -> list[tuple[int, int]]:
		returnMoves = []

		for dr, dc in [(-1, 1), (1, 1), (1, -1), (-1, -1)]:
			for i in range(1, 8):
				newRow = self.row + dr * i
				newCol = self.col + dc * i

				if not (newRow >= 0 and newRow < 8 and newCol >= 0 and newCol < 8):
					break

				if chessBoard.board[newRow][newCol] == None:
					returnMoves.append((newRow, newCol))
				else: 
					returnMoves.append((newRow, newCol))
					break

		return returnMoves
