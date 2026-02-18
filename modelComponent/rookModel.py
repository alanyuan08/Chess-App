# Import Model
from modelComponent.moveCommand import MoveCommand
from modelComponent.chessPieceModel import ChessPieceModel
from modelComponent.chessBoardProtocal import ChessBoardProtocal

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

	def fenValue(self) -> str:
		if self.player == Player.BLACK:
			return "r"
		else:
			return "R"

	def phaseWeight(self) -> int:
		return 2

	def pieceValue(self) -> int:
		return 500

	def computedValue(self, chessBoard: ChessBoardProtocal, phaseWeight: int) -> int:
		row = 0
		if self.player == Player.BLACK:
			row = self.row
		else:
			row = 7 - self.row

		return self.pieceValue() + self.rookValueTable[row][self.col]

	# List all Possible Moves from Location
	def possibleMoves(self, chessBoard: ChessBoardProtocal) -> list[MoveCommand]:
		returnMoves = []

		for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
			for i in range(1, 8):
				newRow = self.row + dr * i
				newCol = self.col + dc * i

				if not (newRow >= 0 and newRow < 8 and newCol >= 0 and newCol < 8):
					break

				if chessBoard.board[newRow][newCol] == None:
					returnMoves.append(
						MoveCommand(self.row, self.col, newRow, newCol, MoveCommandType.MOVE)
					)
				else:
					if chessBoard.board[newRow][newCol].player != self.player:
						returnMoves.append(
							MoveCommand(self.row, self.col, newRow, newCol, MoveCommandType.CAPTURE)
						)
					
					break

		# Validate For King Safety
		return [move for move in returnMoves if chessBoard.validateKingSafety(move)]

	# List of targets - Used to check for Castle / King Safety
	def captureTargets(self, chessBoard: ChessBoardProtocal) -> list[tuple[int, int]]:
		returnMoves = []

		for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
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

