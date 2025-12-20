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

	def pieceValue(self):
		return 20000

	@staticmethod
	def opponent(player: Player):
		if player == Player.WHITE:
			return Player.BLACK
		else:
			return Player.WHITE

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
			if not chessBoardModel.blackKingMoved and not chessBoardModel.blackQueenSideRookMoved:
				nullBlock = 0
				for i in [1, 2, 3, 4]:
					if chessBoardModel.board[7][i] == None and (7, i) not in opponentAttackTargets:
						nullBlock += 1
					
				if nullBlock == 4 and (self.row, self.col) not in opponentAttackTargets:
					returnMoves.append(
						MoveCommand(self.row, self.col, self.row, self.col-2, MoveCommandType.QUEENSIDECASTLE)
					)

			# Black King Side Castle
			if not chessBoardModel.blackKingMoved and not chessBoardModel.blackKingSideRookMoved:
				nullBlock = 0
				for i in [4, 5, 6]:
					if chessBoardModel.board[7][i] == None and (7, i) not in opponentAttackTargets:
						nullBlock += 1

				if nullBlock == 3 and (self.row, self.col) not in opponentAttackTargets:
					returnMoves.append(
						MoveCommand(self.row, self.col, self.row, self.col+2, MoveCommandType.KINGSIDECASTLE)
					)

		elif self.player == Player.WHITE:
			# White Queen Side Castle
			if not chessBoardModel.whiteKingMoved and not chessBoardModel.whiteQueenSideRookMoved:
				nullBlock = 0
				for i in [1, 2, 3, 4]:
					if chessBoardModel.board[0][i] == None and (0, i) not in opponentAttackTargets:
						nullBlock += 1

				if nullBlock == 4 and (self.row, self.col) not in opponentAttackTargets:
					returnMoves.append(
						MoveCommand(self.row, self.col, self.row, self.col-2, MoveCommandType.QUEENSIDECASTLE)
					)

			# White King Side Castle
			if not chessBoardModel.whiteKingMoved and not chessBoardModel.whiteKingSideRookMoved:
				nullBlock = 0
				for i in [4, 5, 6]:
					if chessBoardModel.board[0][i] == None and (0, i) not in opponentAttackTargets:
						nullBlock += 1

				if nullBlock == 3 and (self.row, self.col) not in opponentAttackTargets:
					returnMoves.append(
						MoveCommand(self.row, self.col, self.row, self.col+2, MoveCommandType.KINGSIDECASTLE)
					)

		# Validate For King Safety
		return [move for move in returnMoves if chessBoardModel.validateKingSafety(move)]
		
	# List of targets - Used to check for Castle / King Safety
	def captureTargets(self, chessBoardModel):
		returnMoves = []

		returnMoves = []
		for possibleMoves in [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]:
			newRow = self.row + possibleMoves[0]
			newCol = self.col + possibleMoves[1]
			if newRow >= 0 and newRow < 8 and newCol >= 0 and newCol < 8:
				captureTarget = chessBoardModel.board[newRow][newCol]
				if captureTarget == None or captureTarget.player != self.player:
					returnMoves.append((newRow, newCol))

		return returnMoves



