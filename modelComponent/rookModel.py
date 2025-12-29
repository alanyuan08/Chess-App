# Import Model
from modelComponent.moveCommand import MoveCommand
from modelComponent.chessPieceModel import ChessPieceModel

# Enum
from appEnums import Player, MoveCommandType, PieceType

class RookModel(ChessPieceModel):
	def __init__(self, player: Player, row: int, col: int):

		self.player = player
		self.row = row
		self.col = col
		self.type = PieceType.ROOK
		self.moves = 0

	# Rook Value Table White
	rookValueTable = [
	    [ 0,  0,  0,  5,  5,  0,  0,  0],
	    [ 5, 10, 10, 10, 10, 10, 10,  5],
	    [-5,  0,  0,  0,  0,  0,  0, -5],
	    [-5,  0,  0,  0,  0,  0,  0, -5],
	    [-5,  0,  0,  0,  0,  0,  0, -5],
	    [-5,  0,  0,  0,  0,  0,  0, -5],
	    [-5,  0,  0,  0,  0,  0,  0, -5],
	    [ 0,  0,  0,  5,  5,  0,  0,  0] 
	]

	def pieceValue(self):
		return 500

	def computedValue(self, chessBoard):
		if self.player == Player.BLACK:
			return self.pieceValue() + self.rookValueTable[self.row][self.col]
		else:
			return self.pieceValue() + self.rookValueTable[7 - self.row][self.col]

	# List all Possible Moves from Location
	def possibleMoves(self, chessBoardModel):
		returnMoves = []

		for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
			for i in range(1, 8):
				newRow = self.row + dr * i
				newCol = self.col + dc * i

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

		for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
			for i in range(1, 8):
				newRow = self.row + dr * i
				newCol = self.col + dc * i

				if not (newRow >= 0 and newRow < 8 and newCol >= 0 and newCol < 8):
					break

				if chessBoardModel.board[newRow][newCol] == None:
					returnMoves.append((newRow, newCol))
	
				else:
					returnMoves.append((newRow, newCol))
					break

		return returnMoves

