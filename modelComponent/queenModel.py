# Import Model
from modelComponent.moveCommand import MoveCommand
from modelComponent.chessPieceModel import ChessPieceModel

# Enum
from appEnums import Player, MoveCommandType, PieceType

class QueenModel(ChessPieceModel):
	def __init__(self, player: Player, row: int, col: int):

		self.player = player
		self.row = row
		self.col = col
		self.type = PieceType.QUEEN

	# Queen Value Table White
	queenValueTable = [
	    [-20, -10, -10,  -5,  -5, -10, -10, -20],
	    [-10,   0,   0,   0,   0,   0,   0, -10],
	    [-10,   0,   5,   5,   5,   5,   0, -10],
	    [ -5,   0,   5,   5,   5,   5,   0,  -5],
	    [ -5,   0,   5,   5,   5,   5,   0,  -5],
	    [-10,   0,   5,   5,   5,   5,   0, -10],
	    [-10,   0,   0,   0,   0,   0,   0, -10],
	    [-20, -10, -10,  -5,  -5, -10, -10, -20]
	];

	def pieceValue(self, chessBoard):
		if self.player == Player.BLACK:
			return 900 + self.queenValueTable[self.row][self.col]
		else:
			return 900 + self.queenValueTable[7 - self.row][self.col]

	# List all Possible Moves from Location
	def possibleMoves(self, chessBoardModel):
		returnMoves = []

		for direction in [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]:
			i = 1
			while(True):
				newRow = self.row + direction[0] * i
				newCol = self.col + direction[1] * i
				i += 1

				if not (newRow >= 0 and newRow < 8 and newCol >= 0 and newCol < 8):
					break

				if chessBoardModel.board[newRow][newCol] == None:
					returnMoves.append(
						MoveCommand(self.row, self.col, newRow, newCol, MoveCommandType.MOVE)
					)

				else:
					if chessBoardModel.board[newRow][newCol].player != self.player:
						returnMoves.append(
							MoveCommand(self.row, self.col, newRow, newCol, MoveCommandType.CAPTURE)
						)
						
					break

		# Validate For King Safety
		return [move for move in returnMoves if chessBoardModel.validateKingSafety(move)]

	# List of targets - Used to check for Castle / King Safety
	def captureTargets(self, chessBoardModel):
		returnMoves = []

		for direction in [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]:
			i = 1
			while(True):
				newRow = self.row + direction[0] * i
				newCol = self.col + direction[1] * i
				i += 1

				if not (newRow >= 0 and newRow < 8 and newCol >= 0 and newCol < 8):
					break

				if chessBoardModel.board[newRow][newCol] == None:
					returnMoves.append((newRow, newCol))

				else:
					if chessBoardModel.board[newRow][newCol].player != self.player:
						returnMoves.append((newRow, newCol))

					break


		return returnMoves

