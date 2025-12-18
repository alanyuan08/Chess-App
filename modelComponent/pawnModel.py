# Import Model
from modelComponent.moveCommand import MoveCommand
from modelComponent.chessBoardModel import ChessBoardModel
from modelComponent.chessPieceModel import ChessPieceModel

# Enum
from appEnums import Player, MoveCommandType, PieceType

class PawnModel(ChessPieceModel):
	def __init__(self, player: Player, row: int, col: int):

		self.player = player
		self.row = row
		self.col = col
		self.type = PieceType.PAWN

	def pieceValue(self):
		return 100

	def possibleMoves(self, chessBoardModel: ChessBoardModel):
		returnMoves = []

		# Black Player - Pawn Moves Down
		if self.player == Player.BLACK:
			# Promote
			if self.row == 1:
				target = chessBoardModel.board[self.row-1][self.col-1]
				if self.col > 0 and target != None and target.player != self.player:
					returnMoves.append(
						MoveCommand(self.row, self.col, self.row-1, self.col-1, MoveCommandType.PROMOTE, self.player)
					)

				target = chessBoardModel.board[self.row-1][self.col+1]
				if self.col < 7 and target != None and target.player != self.player:
						returnMoves.append(
							MoveCommand(self.row, self.col, self.row-1, self.col+1, MoveCommandType.PROMOTE, self.player)
						)

				if chessBoardModel.board[self.row-1][self.col] == None:
					returnMoves.append(
						MoveCommand(self.row, self.col, self.row-1, self.col, MoveCommandType.PROMOTE, self.player)
					)
			else:
				# Double Pawn Move
				if self.row == 6:
					if chessBoardModel.board[self.row-1][self.col] == None and chessBoardModel.board[self.row-2][self.col] == None:
						returnMoves.append(
							MoveCommand(self.row, self.col, self.row-2, self.col, MoveCommandType.PAWNOPENMOVE, self.player)
						)

				# En Passant
				if self.row == 3:
					opponentEnPassantCol = chessBoardModel.enPassantColumn
					if opponentEnPassantCol == self.col-1 or opponentEnPassantCol == self.col+1:
						returnMoves.append(
							MoveCommand(self.row, self.col, self.row-1, opponentEnPassantCol, MoveCommandType.ENPASSANT, self.player)
						)

				# Normal
				target = chessBoardModel.board[self.row-1][self.col-1]
				if self.col > 0 and target != None and target.player != self.player:
					returnMoves.append(
						MoveCommand(self.row, self.col, self.row-1, self.col-1, MoveCommandType.CAPTURE, self.player)
					)

				target = chessBoardModel.board[self.row-1][self.col+1]
				if self.col < 7 and target != None and target.player != self.player:
					returnMoves.append(
						MoveCommand(self.row, self.col, self.row-1, self.col+1, MoveCommandType.CAPTURE, self.player)
					)

				# Pawn Forward
				if chessBoardModel.board[self.row-1][self.col] == None:
					returnMoves.append(
						MoveCommand(self.row, self.col, self.row-1, self.col, MoveCommandType.MOVE, self.player)
					)

		# White Player - Pawn Moves Up
		elif self.player == Player.WHITE:
			# Promote
			if self.row == 6:
				target = chessBoardModel.board[self.row+1][self.col-1]
				if self.col > 0 and target != None and target.player != self.player:
					returnMoves.append(
						MoveCommand(self.row, self.col, self.row+1, self.col-1, MoveCommandType.PROMOTE, self.player)
					)

				target = chessBoardModel.board[self.row+1][self.col+1]
				if col < 7 and target != None and target.player != self.player:
					returnMoves.append(
						MoveCommand(self.row, self.col, self.row+1, self.col+1, MoveCommandType.PROMOTE, self.player)
					)

				if chessBoardModel.board[self.row+1][self.col] == None:
					returnMoves.append(
						MoveCommand(self.row, self.col, self.row+1, self.col, MoveCommandType.PROMOTE, self.player)
					)

			else:
				# Double Pawn Move
				if self.row == 1:
					if chessBoardModel.board[self.row+1][self.col] == None and chessBoardModel.board[self.row+2][self.col] == None:
						returnMoves.append(
							MoveCommand(self.row, self.col, self.row+2, self.col, MoveCommandType.PAWNOPENMOVE, self.player)
						)

				# En passant
				if self.row == 4:
					opponentEnPassantCol = chessBoardModel.enPassantColumn
					if opponentEnPassantCol == self.col-1 or opponentEnPassantCol == self.col+1:
						returnMoves.append(
							MoveCommand(self.row, self.col, self.row+1, opponentEnPassantCol, MoveCommandType.ENPASSANT, self.player)
						)

				# Normal
				target = chessBoardModel.board[self.row+1][self.col-1]
				if self.col > 0 and target != None and target!= self.player:
					returnMoves.append(
						MoveCommand(self.row, self.col, self.row+1, self.col-1, MoveCommandType.CAPTURE, self.player)
					)

				target = chessBoardModel.board[self.row+1][self.col+1]
				if self.col < 7 and target != None and target.player != self.player:
					returnMoves.append(
						MoveCommand(self.row, self.col, self.row+1, self.col+1, MoveCommandType.CAPTURE, self.player)
					)

				# Pawn Forward
				if chessBoardModel.board[self.row+1][self.col] == None:
					returnMoves.append(
						MoveCommand(self.row, self.col, self.row+1, self.col, MoveCommandType.MOVE, self.player)
					)

		# Validate For King Safety
		return filter(chessBoardModel.kingSafety, returnMoves)

	def captureTargets(self, chessBoardModel: ChessBoardModel):
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


