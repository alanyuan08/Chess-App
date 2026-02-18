# Import Model
from modelComponent.moveCommand import MoveCommand
from modelComponent.chessPieceModel import ChessPieceModel
from modelComponent.chessBoardProtocal import ChessBoardProtocal

# Enum
from appEnums import Player, MoveCommandType, PieceType

# Math
import math

class QueenModel(ChessPieceModel):
	def __init__(self, player: Player, row: int, col: int):

		self.player = player
		self.row = row
		self.col = col
		self.type = PieceType.QUEEN
		self.moves = 0

	# Queen Value Table White
	queenValueTableEarlyGame = [
	    [-20, -15, -15, -10, -10, -15, -15, -20],
	    [-10,  -5,  -5,   0,   0,  -5,  -5, -10],
	    [ -5,   0,   0,   5,   5,   0,   0,  -5],
	    [ -5,   0,   5,  10,  10,   5,   0,  -5],
	    [ -5,   0,   5,  10,  10,   5,   0,  -5],
	    [ -5,   5,   5,   5,   5,   5,   5,  -5],
	    [  0,   5,  10,  10,  10,  10,   5,   0],
	    [ -5,  -5,   0,  10,   0,   0,  -5,  -5]
    ]

	queenValueTableEndGame = [
	    [-10,  -5,  -5,  -5,  -5,  -5,  -5, -10],
	    [ -5,   0,   5,   5,   5,   5,   0,  -5],
	    [  0,   5,  10,  15,  15,  10,   5,   0],
	    [  5,  10,  20,  30,  30,  20,  10,   5],
	    [  5,  10,  20,  30,  30,  20,  10,   5],
	    [  0,   5,  10,  15,  15,  10,   5,   0],
	    [ -5,   0,   5,   5,   5,   5,   0,  -5],
	    [-20, -15, -10,  -5,  -5, -10, -15, -20]
	]

	def fenValue(self) -> str:
		if self.player == Player.BLACK:
			return "q"
		else:
			return "Q"

	def pieceValue(self) -> int:
		return 900

	def phaseWeight(self) -> int:
		return 4

	def computedValue(self, chessBoard: ChessBoardProtocal, phaseWeight: int) -> int:
		row = 0
		if self.player == Player.BLACK:
			row = self.row
		else:
			row = 7 - self.row

		earlyGame = self.queenValueTableEarlyGame[row][self.col]
		endGame = self.queenValueTableEndGame[row][self.col]

		computedPhase = earlyGame * phaseWeight + endGame * (24 - phaseWeight) 

		return self.pieceValue() + math.ceil(computedPhase / 24)

	# List all Possible Moves from Location
	def possibleMoves(self, chessBoard: ChessBoardProtocal) -> list[MoveCommand]:
		returnMoves = []

		for dr, dc in [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]:
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

		for dr, dc in [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]:
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

