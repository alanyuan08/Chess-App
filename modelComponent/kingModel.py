# Import Model
from modelComponent.moveCommand import MoveCommand
from modelComponent.chessPieceModel import ChessPieceModel

# Enum
from appEnums import Player, MoveCommandType, PieceType

class KingModel(ChessPieceModel):
	def __init__(self, player: Player, row: int, col: int):

		self.player = player
		self.row = row
		self.col = col
		self.type = PieceType.KING
		self.moves = 0

	def pieceValue(self, chessBoard):
		returnScore = 20000

		# Castle Bonus
		if self.player == Player.WHITE and chessBoard.whiteCastled:
			returnScore += 100
		if self.player == Player.BLACK and chessBoard.blackCastled:
			returnScore += 100

		# Reduction for cannot Castle
		if self.moves > 0:
			returnScore -= 60
		else:
			if self.checkRook(chessBoard, self.row, 0):
				returnScore -= 30
			elif self.checkRook(chessBoard, self.row, 7):
				returnScore -= 30

		return returnScore

	@staticmethod
	def opponent(player: Player):
		if player == Player.WHITE:
			return Player.BLACK
		else:
			return Player.WHITE

	def checkRook(self, chessBoardModel, row: int, col: int):
		rookPiece = chessBoardModel.board[row][col]
		if rookPiece != None and rookPiece.type == PieceType.ROOK \
			and rookPiece.player == self.player and rookPiece.moves == 0:
			return True
		else:
			return False

	# List all Possible Moves from Location
	def possibleMoves(self, chessBoardModel):
		returnMoves = []

		# King + Move to Castle are not in check
		opponent = KingModel.opponent(self.player)
		opponentAttackTargets = chessBoardModel._allPlayerCaptureTargets(opponent)

		for dr, dc in [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]:
			newRow = self.row + dr
			newCol = self.col + dc
			if newRow >= 0 and newRow < 8 and newCol >= 0 and newCol < 8:
				if (newRow, newCol) not in opponentAttackTargets:
					if chessBoardModel.board[newRow][newCol] == None:
						returnMoves.append(
							MoveCommand(self.row, self.col, newRow, newCol, MoveCommandType.MOVE)
						)
					elif chessBoardModel.board[newRow][newCol].player != self.player:
						returnMoves.append(
							MoveCommand(self.row, self.col, newRow, newCol, MoveCommandType.CAPTURE)
						)

		if (self.row, self.col) not in opponentAttackTargets:
			# Queen Side Castle
			if self.moves == 0 and self.checkRook(chessBoardModel, self.row, 0):
				if all(chessBoardModel.board[self.row][i] == None for i in [1, 2, 3]):
					if (self.row, 2) not in opponentAttackTargets and (self.row, 3) not in opponentAttackTargets:
						returnMoves.append(
							MoveCommand(self.row, self.col, self.row, self.col-2, MoveCommandType.QUEENSIDECASTLE)
						)

			# King Side Castle
			if self.moves == 0 and self.checkRook(chessBoardModel, self.row, 7):
				if all(chessBoardModel.board[self.row][i] == None for i in [5, 6]):
					if (self.row, 5) not in opponentAttackTargets and (self.row, 6) not in opponentAttackTargets:
						returnMoves.append(
							MoveCommand(self.row, self.col, self.row, self.col+2, MoveCommandType.KINGSIDECASTLE)
						)

		# Move is Validated to not be in opponent attack target
		return returnMoves
		
	# List of targets - Used to check for Castle
	def captureTargets(self, chessBoardModel):
		returnMoves = []
		for dr, dc in [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]:
			newRow = self.row + dr
			newCol = self.col + dc
			if newRow >= 0 and newRow < 8 and newCol >= 0 and newCol < 8:
				returnMoves.append((newRow, newCol))

		return returnMoves

	# Test King Safety
	def evaluateKingSafety(self, chessBoardModel):
		# Test for Opponent Knights
		for dr, dc in [(2, 1), (1, 2), (-2, -1), (-1, -2), (-2, 1), (-1, 2), (2, -1), (1, -2)]:
			newRow = self.row + dr
			newCol = self.col + dc
			if newRow >= 0 and newRow < 8 and newCol >= 0 and newCol < 8:
				target = chessBoardModel.board[newRow][newCol] 
				if target != None and target.type == PieceType.KNIGHT and target.player != self.player:
					return False

		# Test for Opponent Horizontals
		horizontalCapture = [PieceType.ROOK, PieceType.QUEEN]
		for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
			for i in range(1, 8):
				newRow = self.row + dr * i
				newCol = self.col + dc * i

				if not (newRow >= 0 and newRow < 8 and newCol >= 0 and newCol < 8):
					break

				target = chessBoardModel.board[newRow][newCol]
				if target != None:
					if target.player != self.player:
						if i == 1 and target.type == PieceType.KING: 
							return False
						if target.type in horizontalCapture:
							return False
					break # Path Blocked

		# Test for Diagonal Captures
		diagonalCapture = [PieceType.BISHOP, PieceType.QUEEN]
		for dr, dc in [(-1, 1), (1, 1), (1, -1), (-1, -1)]:
			for i in range(1, 8):
				newRow = self.row + dr * i
				newCol = self.col + dc * i

				if not (newRow >= 0 and newRow < 8 and newCol >= 0 and newCol < 8):
					break

				target = chessBoardModel.board[newRow][newCol]
				if target != None:
					if target.player != self.player:
						if i == 1 and target.type == PieceType.KING: 
							return False
						if target.type in diagonalCapture:
							return False
					break # Path Blocked

		# Test for enemy pawns 
		pawnRow = self.row - 1 
		if self.player == Player.WHITE:
			pawnRow = self.row + 1

		if 0 <= pawnRow < 8:
			for columnDirection in [-1, 1]:
				newCol = self.col + columnDirection

				if 0 <= newCol < 8:
					target = chessBoardModel.board[pawnRow][newCol]
					if target and target.type == PieceType.PAWN and target.player != self.player:
						return False

		return True



