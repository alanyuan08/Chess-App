# Import Model
from modelComponent.moveCommand import MoveCommand
from modelComponent.chessPieceModel import ChessPieceModel

# Enum
from appEnums import Player, MoveCommandType, PieceType

class PawnModel(ChessPieceModel):
	def __init__(self, player: Player, row: int, col: int):

		self.player = player
		self.row = row
		self.col = col
		self.type = PieceType.PAWN
		self.moves = 0

	# Pawn Value Table White
	pawnValueTable = [
	    [ 0,  0,  0,  0,  0,  0,  0,  0],
	    [50, 50, 50, 50, 50, 50, 50, 50],
	    [10, 10, 20, 30, 30, 20, 10, 10],
	    [ 5,  5, 10, 25, 25, 10,  5,  5],
	    [ 0,  0,  0, 20, 20,  0,  0,  0],
	    [ 5, -5,-10,  0,  0,-10, -5,  5],
	    [ 5, 10, 10,-20,-20, 10, 10,  5],
	    [ 0,  0,  0,  0,  0,  0,  0,  0] 
	]

	def pieceValue(self, chessBoard):
		if self.player == Player.BLACK:
			return 100 + self.pawnValueTable[self.row][self.col]
		else:
			return 100 + self.pawnValueTable[7 - self.row][self.col]

	def checkOpponentPawn(self, chessBoardModel, row: int, col: int):
		rookPiece = chessBoardModel.board[row][col]
		if rookPiece != None and rookPiece.type == PieceType.ROOK and rookPiece.player == self.player:
			return True
		else:
			return False

	def possibleMoves(self, chessBoardModel):
		returnMoves = []

		# Black Player - Pawn Moves Down
		if self.player == Player.BLACK:
			# Promote
			if self.row == 1:
				if self.col > 0:
					target = chessBoardModel.board[self.row-1][self.col-1]
					if target != None and target.player != self.player:
						returnMoves.append(
							MoveCommand(self.row, self.col, self.row-1, self.col-1, MoveCommandType.PROMOTE)
						)

				if self.col < 7:
					target = chessBoardModel.board[self.row-1][self.col+1]
					if target != None and target.player != self.player:
						returnMoves.append(
							MoveCommand(self.row, self.col, self.row-1, self.col+1, MoveCommandType.PROMOTE)
						)

				if chessBoardModel.board[self.row-1][self.col] == None:
					returnMoves.append(
						MoveCommand(self.row, self.col, self.row-1, self.col, MoveCommandType.PROMOTE)
					)
			elif self.row > 1:
				# Double Pawn Move
				if self.row == 6:
					if chessBoardModel.board[self.row-1][self.col] == None and chessBoardModel.board[self.row-2][self.col] == None:
						returnMoves.append(
							MoveCommand(self.row, self.col, self.row-2, self.col, MoveCommandType.PAWNOPENMOVE)
						)

				# En Passant
				if self.row == 3:
					opponentEnPassantCol = chessBoardModel.whiteEnPassantColumn
					if opponentEnPassantCol == self.col-1 or opponentEnPassantCol == self.col+1:

						opponentPawn = chessBoardModel.board[self.row][opponentEnPassantCol]
						if opponentPawn.player != self.player:
							returnMoves.append(
								MoveCommand(self.row, self.col, self.row-1, opponentEnPassantCol, MoveCommandType.ENPASSANT)
							)

				# Normal
				if self.col > 0:
					target = chessBoardModel.board[self.row-1][self.col-1]
					if target != None and target.player != self.player:
						returnMoves.append(
							MoveCommand(self.row, self.col, self.row-1, self.col-1, MoveCommandType.CAPTURE)
						)

				if self.col < 7:
					target = chessBoardModel.board[self.row-1][self.col+1]
					if target != None and target.player != self.player:
						returnMoves.append(
							MoveCommand(self.row, self.col, self.row-1, self.col+1, MoveCommandType.CAPTURE)
						)

				# Pawn Forward
				if chessBoardModel.board[self.row-1][self.col] == None:
					returnMoves.append(
						MoveCommand(self.row, self.col, self.row-1, self.col, MoveCommandType.MOVE)
					)

		# White Player - Pawn Moves Up
		elif self.player == Player.WHITE:
			# Promote
			if self.row == 6:
				if self.col > 0:
					target = chessBoardModel.board[self.row+1][self.col-1]
					if target != None and target.player != self.player:
						returnMoves.append(
							MoveCommand(self.row, self.col, self.row+1, self.col-1, MoveCommandType.PROMOTE)
						)

				if self.col < 7:
					target = chessBoardModel.board[self.row+1][self.col+1]
					if target != None and target.player != self.player:
						returnMoves.append(
							MoveCommand(self.row, self.col, self.row+1, self.col+1, MoveCommandType.PROMOTE)
						)

				if chessBoardModel.board[self.row+1][self.col] == None:
					returnMoves.append(
						MoveCommand(self.row, self.col, self.row+1, self.col, MoveCommandType.PROMOTE)
					)

			elif self.row < 6:
				# Double Pawn Move
				if self.row == 1:
					if chessBoardModel.board[self.row+1][self.col] == None and chessBoardModel.board[self.row+2][self.col] == None:
						returnMoves.append(
							MoveCommand(self.row, self.col, self.row+2, self.col, MoveCommandType.PAWNOPENMOVE)
						)

				# En passant
				if self.row == 4:
					opponentEnPassantCol = chessBoardModel.blackEnPassantColumn
					if opponentEnPassantCol == self.col-1 or opponentEnPassantCol == self.col+1:

						opponentPawn = chessBoardModel.board[self.row][opponentEnPassantCol]
						if opponentPawn.player != self.player:
							returnMoves.append(
								MoveCommand(self.row, self.col, self.row+1, opponentEnPassantCol, MoveCommandType.ENPASSANT)
							)

				# Normal
				if self.col > 0:
					target = chessBoardModel.board[self.row+1][self.col-1]
					if target != None and target.player != self.player:
						returnMoves.append(
							MoveCommand(self.row, self.col, self.row+1, self.col-1, MoveCommandType.CAPTURE)
						)

				if self.col < 7:
					target = chessBoardModel.board[self.row+1][self.col+1]
					if target != None and target.player != self.player:
						returnMoves.append(
							MoveCommand(self.row, self.col, self.row+1, self.col+1, MoveCommandType.CAPTURE)
						)

				# Pawn Forward
				if chessBoardModel.board[self.row+1][self.col] == None:
					returnMoves.append(
						MoveCommand(self.row, self.col, self.row+1, self.col, MoveCommandType.MOVE)
					)

		# Validate For King Safety
		return [move for move in returnMoves if chessBoardModel.validateKingSafety(move)]

	def captureTargets(self, chessBoardModel):
		returnMoves = []
		if self.player == Player.BLACK:
			if self.row > 0:
				# Capture Point
				if self.col > 0:
					returnMoves.append((self.row-1, self.col-1))

				if self.col < 7:
					returnMoves.append((self.row-1, self.col+1))

		elif self.player == Player.WHITE:
			if self.row < 7:
				# Capture Point
				if self.col > 0:
					returnMoves.append((self.row+1, self.col-1))

				if self.col < 7:
					returnMoves.append((self.row+1, self.col+1))

		return returnMoves
