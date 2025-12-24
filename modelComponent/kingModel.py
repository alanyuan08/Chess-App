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

	def pieceValue(self, chessBoard):
		returnScore = 20000
		if self.player == Player.WHITE:
			if chessBoard.whiteCastled:
				returnScore += 100
			else:
				if not chessBoard.whiteKingSideCanCastle:
					returnScore -= 30
				if not chessBoard.whiteQueenSideCanCastle:
					returnScore -= 30

		elif self.player == Player.BLACK:
			if chessBoard.blackCastled:
				returnScore += 100
			else:
				if not chessBoard.blackKingSideCanCastle:
					returnScore -= 30
				if not chessBoard.blackQueenSideCanCastle:
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
		if rookPiece != None and rookPiece.type == PieceType.ROOK and rookPiece.player == self.player:
			return True
		else:
			return False

	# List all Possible Moves from Location
	def possibleMoves(self, chessBoardModel):
		returnMoves = []

		for possibleMoves in [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]:
			newRow = self.row + possibleMoves[0]
			newCol = self.col + possibleMoves[1]
			if newRow >= 0 and newRow < 8 and newCol >= 0 and newCol < 8:
				if chessBoardModel.board[newRow][newCol] == None:
					returnMoves.append(
						MoveCommand(self.row, self.col, newRow, newCol, MoveCommandType.MOVE)
					)
				elif chessBoardModel.board[newRow][newCol].player != self.player:
					returnMoves.append(
						MoveCommand(self.row, self.col, newRow, newCol, MoveCommandType.CAPTURE)
					)

		# King + Move to Castle are not in check
		opponent = KingModel.opponent(self.player)
		opponentAttackTargets = chessBoardModel._allPlayerCaptureTargets(opponent)

		if self.player == Player.BLACK:
			# Black Queen Side Castle
			if chessBoardModel.blackQueenSideCanCastle and self.checkRook(chessBoardModel, 0, 7):
				nullBlock = 0
				for i in [1, 2, 3]:
					if chessBoardModel.board[7][i] == None and (7, i) not in opponentAttackTargets:
						nullBlock += 1
					
				if nullBlock == 3 and (self.row, self.col) not in opponentAttackTargets:
					returnMoves.append(
						MoveCommand(self.row, self.col, self.row, self.col-2, MoveCommandType.QUEENSIDECASTLE)
					)

			# Black King Side Castle
			if chessBoardModel.blackKingSideCanCastle and self.checkRook(chessBoardModel, 7, 7):
				nullBlock = 0
				for i in [5, 6]:
					if chessBoardModel.board[7][i] == None and (7, i) not in opponentAttackTargets:
						nullBlock += 1

				if nullBlock == 2 and (self.row, self.col) not in opponentAttackTargets:
					returnMoves.append(
						MoveCommand(self.row, self.col, self.row, self.col+2, MoveCommandType.KINGSIDECASTLE)
					)

		elif self.player == Player.WHITE:
			# White Queen Side Castle
			if chessBoardModel.whiteQueenSideCanCastle and self.checkRook(chessBoardModel, 0, 0):
				nullBlock = 0
				for i in [1, 2, 3]:
					if chessBoardModel.board[0][i] == None and (0, i) not in opponentAttackTargets:
						nullBlock += 1

				if nullBlock == 3 and (self.row, self.col) not in opponentAttackTargets:
					returnMoves.append(
						MoveCommand(self.row, self.col, self.row, self.col-2, MoveCommandType.QUEENSIDECASTLE)
					)

			# White King Side Castle
			if chessBoardModel.whiteKingSideCanCastle  and self.checkRook(chessBoardModel, 0, 7):
				nullBlock = 0
				for i in [5, 6]:
					if chessBoardModel.board[0][i] == None and (0, i) not in opponentAttackTargets:
						nullBlock += 1

				if nullBlock == 2 and (self.row, self.col) not in opponentAttackTargets:
					returnMoves.append(
						MoveCommand(self.row, self.col, self.row, self.col+2, MoveCommandType.KINGSIDECASTLE)
					)

		# Validate For King Safety
		return [move for move in returnMoves if chessBoardModel.validateKingSafety(move)]
		
	# List of targets - Used to check for Castle
	def captureTargets(self, chessBoardModel):
		returnMoves = []
		for possibleMoves in [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]:
			newRow = self.row + possibleMoves[0]
			newCol = self.col + possibleMoves[1]
			if newRow >= 0 and newRow < 8 and newCol >= 0 and newCol < 8:
				captureTarget = chessBoardModel.board[newRow][newCol]
				if captureTarget == None or captureTarget.player != self.player:
					returnMoves.append((newRow, newCol))

		return returnMoves

	# Test King Safety
	def evaluateKingSafety(self, chessBoardModel):
		# Test for Opponent Knights
		for possibleMoves in [(2, 1), (1, 2), (-2, -1), (-1, -2), (-2, 1), (-1, 2), (2, -1), (1, -2)]:
			newRow = self.row + possibleMoves[0]
			newCol = self.col + possibleMoves[1]
			if newRow >= 0 and newRow < 8 and newCol >= 0 and newCol < 8:
				target = chessBoardModel.board[newRow][newCol] 
				if target != None and target.type == PieceType.KNIGHT and target.player != self.player:
					return False

		# Test for Opponent Horizontals
		horizontalCapture = [PieceType.KING, PieceType.ROOK, PieceType.QUEEN]
		for direction in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
			i = 1
			while(True):
				newRow = self.row + direction[0] * i
				newCol = self.col + direction[1] * i
				i += 1

				if not (newRow >= 0 and newRow < 8 and newCol >= 0 and newCol < 8):
					break

				target = chessBoardModel.board[newRow][newCol]
				if target != None:
					if target.type in horizontalCapture and target.player != self.player:
						return False
					break

		# Test for Diagonal Captures
		diagonalCapture = [PieceType.KING, PieceType.BISHOP, PieceType.QUEEN]
		for direction in [(-1, 1), (1, 1), (1, -1), (-1, -1)]:
			i = 1
			while(True):
				newRow = self.row + direction[0] * i
				newCol = self.col + direction[1] * i
				i += 1

				if not (newRow >= 0 and newRow < 8 and newCol >= 0 and newCol < 8):
					break

				target = chessBoardModel.board[newRow][newCol]
				if target != None:
					if target.type in diagonalCapture and target.player != self.player:
						return False

					break

		# Test for enemy pawns 
		if self.player == Player.WHITE:
			# Black Pawn Moves Down, White King Checks below
			if self.row + 1 < 8 and self.col - 1 > 0:
				blackPawn = chessBoardModel.board[self.row + 1][self.col - 1]
				if blackPawn != None and blackPawn.type == PieceType.PAWN and blackPawn.player != self.player:
					return False

			if self.row + 1 < 8 and self.col + 1 < 8:
				blackPawn = chessBoardModel.board[self.row + 1][self.col + 1]
				if blackPawn != None and blackPawn.type == PieceType.PAWN and blackPawn.player != self.player:
					return False

		elif self.player == Player.BLACK:
			# White Pawn Moves Up, Black King Checks Above
			if self.row - 1 > 0 and self.col - 1 > 0:
				whitePawn = chessBoardModel.board[self.row - 1][self.col - 1]
				if whitePawn != None and whitePawn.type == PieceType.PAWN and whitePawn.player != self.player:
					return False

			if self.row - 1 > 0 and self.col + 1 < 8:
				whitePawn = chessBoardModel.board[self.row - 1][self.col + 1]
				if whitePawn != None and whitePawn.type == PieceType.PAWN and whitePawn.player != self.player:
					return False

		return True



