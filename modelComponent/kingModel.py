# Import Model
from modelComponent.moveCommand import MoveCommand
from modelComponent.chessBoardModel import ChessBoardModel
from modelComponent.chessPieceModel import ChessPieceModel

# Enum
from appEnums import Player, MoveCommandType

class KingModel(ChessPieceModel):
	def __init__(self, player: Player, row: int, col: int):

		self.player = player
		self.row = row
		self.col = col

	def pieceValue(self):
		return 20000

	# List all Possible Moves from Location
	def possibleMoves(self, chessBoardModel: ChessBoardModel):
		returnMoves = []

		for possibleMoves in [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]:
			newRow = self.row + possibleMoves[0]
			newCol = self.col + possibleMoves[1]
			if newRow >= 0 and newRow < 8 and newCol >= 0 and newCol < 8:
				if chessBoardModel.board[newRow][newCol] == None:
					returnMoves.append(
						MoveCommand(self.row, self.col, newRow, newCol, MoveCommandType.MOVE, self.player)
					)
				elif chessBoardModel.board[newRow][newCol].player != chessBoardModel.player:
					returnMoves.append(
						MoveCommand(self.row, self.col, newRow, newCol, MoveCommandType.CAPTURE, self.player)
					)

		# King + Move to Castle are not in check
		opponent = ChessBoardModel.opponent(self.player)
		opponentAttackTargets = chessBoardModel.allOpponentCaptureTargets(opponent)

		# Black Queen Side Castle
		if not chessBoardModel.blackKingMoved and not chessBoardModel.blackQueenSideRookMoved:
			nullBlock = 0
			for i in [1, 2, 3]:
				if chessBoardModel.board[7][i] == None and (7, i) not in opponentAttackTargets:
					nullBlock += 1
				
			if nullBlock == 3 and (self.row, self.col) not in opponentAttackTargets:
				returnMoves.append(
					MoveCommand(self.row, self.col, self.row, self.col-2, MoveCommandType.QUEENSIDECASTLE, self.player)
				)

		# Black King Side Castle
		if not chessBoardModel.blackKingMoved and not chessBoardModel.blackKingSideRookMoved:
			nullBlock = 0
			for i in [5, 6]:
				if chessBoardModel.board[7][i] == None and (7, i) not in opponentAttackTargets:
					nullBlock += 1

			if nullBlock == 2 and (self.row, self.col) not in opponentAttackTargets:
				returnMoves.append(
					MoveCommand(self.row, self.col, self.row, self.col+2, MoveCommandType.KINGSIDECASTLE, self.player)
				)

		# White Queen Side Castle
		if not chessBoardModel.whiteKingMoved and not chessBoardModel.whiteQueenSideRookMoved:
			nullBlock = 0
			for i in [1, 2, 3]:
				if chessBoardModel.board[0][i] == None and (0, i) not in opponentAttackTargets:
					nullBlock += 1

			if nullBlock == 3 and (self.row, self.col) not in opponentAttackTargets:
				returnMoves.append(
					MoveCommand(self.row, self.col, self.row, self.col-2, MoveCommandType.QUEENSIDECASTLE, self.player)
				)

		# White King Side Castle
		if not chessBoardModel.whiteKingMoved and not chessBoardModel.whiteKingSideRookMoved:
			nullBlock = 0
			for i in [5, 6]:
				if chessBoardModel.board[0][i] == None and (0, i) not in opponentAttackTargets:
					nullBlock += 1

			if nullBlock == 2 and (self.row, self.col) not in opponentAttackTargets:
				returnMoves.append(
					MoveCommand(self.row, self.col, self.row, self.col+2, MoveCommandType.KINGSIDECASTLE, self.player)
				)

		# Validate For King Safety
		return filter(chessBoardModel.kingSafety, returnMoves)

	# List of targets - Used to check for Castle / King Safety
	def captureTargets(self, chessBoardModel: ChessBoardModel):
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


