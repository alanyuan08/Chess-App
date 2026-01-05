# Import Model
from modelComponent.moveCommand import MoveCommand
from modelComponent.chessPieceModel import ChessPieceModel
from modelComponent.chessBoardProtocal import ChessBoardProtocal

# Enum
from appEnums import Player, MoveCommandType, PieceType

# Math
import math

class KingModel(ChessPieceModel):
	def __init__(self, player: Player, row: int, col: int):

		self.player = player
		self.row = row
		self.col = col
		self.type = PieceType.KING
		self.moves = 0

		self.castled = False

	# King Value Table White
	kingValueTableEarlyGame = [
	    [-50, -40, -40, -50, -50, -40, -40, -50],
	    [-30, -30, -30, -30, -30, -30, -30, -30],
	    [-30, -30, -30, -30, -30, -30, -30, -30],
	    [-30, -30, -30, -30, -30, -30, -30, -30],
	    [-20, -20, -20, -20, -20, -20, -20, -20],
	    [-10, -10, -10, -10, -10, -10, -10, -10],
	    [ 10,  10,   0,   0,   0,   0,  10,  10],
	    [ 20,  30,  10,   0,   0,  10,  30,  20]
	]

	kingValueTableEndGame = [
	    [-50, -30, -30, -30, -30, -30, -30, -50],
	    [-30, -10,   0,   0,   0,   0, -10, -30], 
	    [-30,   0,  15,  20,  20,  15,   0, -30],
	    [-30,   0,  30,  40,  40,  30,   0, -30],
	    [-30,   0,  30,  40,  40,  30,   0, -30],
	    [-30,   0,  15,  20,  20,  15,   0, -30],
	    [-30, -10,   0,   0,   0,   0, -10, -30],
	    [-50, -30, -30, -30, -30, -30, -30, -50] 
	]

	# Update Castle
	def updateCastle(self, castled: bool):
		self.castled = castled

	# King is not part of phase calculations
	def phaseWeight(self) -> int:
		return 0

	def pieceValue(self) -> int:
		return 10000

	def computedValue(self, chessBoard: ChessBoardProtocal, phaseWeight: int) -> int:
		returnScore = self.pieceValue()

		# Castle Bonus
		if self.castled:
			returnScore += 90
		else:
			if self.canQueenSideCastle(chessBoard):
				returnScore -= 30
			elif self.canKingSideCastle(chessBoard):
				returnScore -= 30

		# Value Table Bonus
		row = 0
		if self.player == Player.BLACK:
			row = self.row
		else:
			row = 7 - self.row

		earlyGame = self.kingValueTableEarlyGame[row][self.col]
		endGame = self.kingValueTableEndGame[row][self.col]

		computedPhase = earlyGame * phaseWeight + endGame * (24 - phaseWeight) 

		return returnScore + math.ceil(computedPhase / 24)

	# List all Possible Moves from Location
	def possibleMoves(self, chessBoard: ChessBoardProtocal) -> list[MoveCommand]:
		returnMoves = []

		for dr, dc in [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]:
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

		# King + Move to Castle are not in check
		opponentAttackTargets = chessBoard.allOpponentCaptureTargets()

		if (self.row, self.col) not in opponentAttackTargets:
			# Queen Side Castle
			if self.canQueenSideCastle(chessBoard):
				if all(chessBoard.board[self.row][i] == None for i in [1, 2, 3]):
					if (self.row, 2) not in opponentAttackTargets and (self.row, 3) not in opponentAttackTargets:
						returnMoves.append(
							MoveCommand(self.row, self.col, self.row, self.col-2, MoveCommandType.QUEENSIDECASTLE)
						)

			# King Side Castle
			if self.canKingSideCastle(chessBoard):
				if all(chessBoard.board[self.row][i] == None for i in [5, 6]):
					if (self.row, 5) not in opponentAttackTargets and (self.row, 6) not in opponentAttackTargets:
						returnMoves.append(
							MoveCommand(self.row, self.col, self.row, self.col+2, MoveCommandType.KINGSIDECASTLE)
						)

		# Move is Validated to not be in opponent attack target
		return [move for move in returnMoves if chessBoard.validateKingSafety(move)]
		
	# List of targets - Used to check for Castle
	def captureTargets(self, chessBoard: ChessBoardProtocal) -> list[tuple[int, int]]:
		returnMoves = []
		for dr, dc in [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]:
			newRow = self.row + dr
			newCol = self.col + dc
			if newRow >= 0 and newRow < 8 and newCol >= 0 and newCol < 8:
				returnMoves.append((newRow, newCol))

		return returnMoves

	def canQueenSideCastle(self, chessBoard: ChessBoardProtocal) -> bool:
		if self.moves == 0 and self._checkRook(chessBoard, self.row, 0):
			return True
		else:
			return False

	def canKingSideCastle(self, chessBoard: ChessBoardProtocal) -> bool:
		if self.moves == 0 and self._checkRook(chessBoard, self.row, 7):
			return True
		else:
			return False

	def _checkRook(self, chessBoard: ChessBoardProtocal, row: int, col: int) -> bool:
		rookPiece = chessBoard.board[row][col]
		if rookPiece != None and rookPiece.type == PieceType.ROOK \
			and rookPiece.player == self.player and rookPiece.moves == 0:
			return True
		else:
			return False

