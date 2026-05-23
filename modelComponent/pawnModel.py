# Import Model
from modelComponent.moveCommand import MoveCommand
from modelComponent.chessPieceModel import ChessPieceModel
from modelComponent.chessBoardProtocal import ChessBoardProtocal

# Enum
from appEnums import Player, MoveCommandType, PieceType

# Math
import math

class PawnModel(ChessPieceModel):
	def __init__(self, player: Player, row: int, col: int):

		self.player = player
		self.row = row
		self.col = col
		self.type = PieceType.PAWN
		self.moves = 0

	def possibleMoves(self, chessBoard: ChessBoardProtocal) -> list[MoveCommand]:
		returnMoves = []

		# Black Player - Pawn Moves Down
		if self.player == Player.BLACK:
			# Promote
			if self.row == 1:
				if self.col > 0:
					target = chessBoard.board[self.row-1][self.col-1]
					if target != None and target.player != self.player:
						for promote_type in [MoveCommandType.PROMOTION_QUEEN, MoveCommandType.PROMOTION_ROOK, \
                			MoveCommandType.PROMOTION_BISHOP, MoveCommandType.PROMOTION_KNIGHT]:
							returnMoves.append(
								MoveCommand(self.row, self.col, self.row-1, self.col-1, promote_type)
							)

				if self.col < 7:
					target = chessBoard.board[self.row-1][self.col+1]
					if target != None and target.player != self.player:
						for promote_type in [MoveCommandType.PROMOTION_QUEEN, MoveCommandType.PROMOTION_ROOK, \
                			MoveCommandType.PROMOTION_BISHOP, MoveCommandType.PROMOTION_KNIGHT]:
							returnMoves.append(
								MoveCommand(self.row, self.col, self.row-1, self.col+1, promote_type)
							)

				if chessBoard.board[self.row-1][self.col] == None:
					for promote_type in [MoveCommandType.PROMOTION_QUEEN, MoveCommandType.PROMOTION_ROOK, \
                		MoveCommandType.PROMOTION_BISHOP, MoveCommandType.PROMOTION_KNIGHT]:
						returnMoves.append(
							MoveCommand(self.row, self.col, self.row-1, self.col, promote_type)
						)

			elif self.row > 1:
				# Double Pawn Move
				if self.row == 6:
					if chessBoard.board[self.row-1][self.col] == None and chessBoard.board[self.row-2][self.col] == None:
						returnMoves.append(
							MoveCommand(self.row, self.col, self.row-2, self.col, MoveCommandType.PAWNOPENMOVE)
						)

				# En Passant
				if self.row == 3:
					opponentEnPassantCol = chessBoard.enPassant
					if opponentEnPassantCol != 8:
						if opponentEnPassantCol == self.col-1 or opponentEnPassantCol == self.col+1:

							opponentPawn = chessBoard.board[self.row][opponentEnPassantCol]
							if opponentPawn.player != self.player:
								returnMoves.append(
									MoveCommand(self.row, self.col, self.row-1, opponentEnPassantCol, MoveCommandType.ENPASSANT)
							)

				# Normal
				if self.col > 0:
					target = chessBoard.board[self.row-1][self.col-1]
					if target != None and target.player != self.player:
						returnMoves.append(
							MoveCommand(self.row, self.col, self.row-1, self.col-1, MoveCommandType.CAPTURE)
						)

				if self.col < 7:
					target = chessBoard.board[self.row-1][self.col+1]
					if target != None and target.player != self.player:
						returnMoves.append(
							MoveCommand(self.row, self.col, self.row-1, self.col+1, MoveCommandType.CAPTURE)
						)

				# Pawn Forward
				if chessBoard.board[self.row-1][self.col] == None:
					returnMoves.append(
						MoveCommand(self.row, self.col, self.row-1, self.col, MoveCommandType.MOVE)
					)

		# White Player - Pawn Moves Up
		elif self.player == Player.WHITE:
			# Promote
			if self.row == 6:
				if self.col > 0:
					target = chessBoard.board[self.row+1][self.col-1]
					if target != None and target.player != self.player:
						for promote_type in [MoveCommandType.PROMOTION_QUEEN, MoveCommandType.PROMOTION_ROOK, \
                			MoveCommandType.PROMOTION_BISHOP, MoveCommandType.PROMOTION_KNIGHT]:
							returnMoves.append(
								MoveCommand(self.row, self.col, self.row+1, self.col-1, promote_type)
							)

				if self.col < 7:
					target = chessBoard.board[self.row+1][self.col+1]
					if target != None and target.player != self.player:
						for promote_type in [MoveCommandType.PROMOTION_QUEEN, MoveCommandType.PROMOTION_ROOK, \
                			MoveCommandType.PROMOTION_BISHOP, MoveCommandType.PROMOTION_KNIGHT]:
							returnMoves.append(
								MoveCommand(self.row, self.col, self.row+1, self.col+1, promote_type)
							)

				if chessBoard.board[self.row+1][self.col] == None:
					for promote_type in [MoveCommandType.PROMOTION_QUEEN, MoveCommandType.PROMOTION_ROOK, \
                			MoveCommandType.PROMOTION_BISHOP, MoveCommandType.PROMOTION_KNIGHT]:
						returnMoves.append(
							MoveCommand(self.row, self.col, self.row+1, self.col, promote_type)
						)

			elif self.row < 6:
				# Double Pawn Move
				if self.row == 1:
					if chessBoard.board[self.row+1][self.col] == None and chessBoard.board[self.row+2][self.col] == None:
						returnMoves.append(
							MoveCommand(self.row, self.col, self.row+2, self.col, MoveCommandType.PAWNOPENMOVE)
						)

				# En passant
				if self.row == 4:
					opponentEnPassantCol = chessBoard.enPassant
					if opponentEnPassantCol != 8:
						if opponentEnPassantCol == self.col-1 or opponentEnPassantCol == self.col+1:

							opponentPawn = chessBoard.board[self.row][opponentEnPassantCol]
							if opponentPawn.player != self.player:
								returnMoves.append(
									MoveCommand(self.row, self.col, self.row+1, opponentEnPassantCol, MoveCommandType.ENPASSANT)
								)

				# Normal
				if self.col > 0:
					target = chessBoard.board[self.row+1][self.col-1]
					if target != None and target.player != self.player:
						returnMoves.append(
							MoveCommand(self.row, self.col, self.row+1, self.col-1, MoveCommandType.CAPTURE)
						)

				if self.col < 7:
					target = chessBoard.board[self.row+1][self.col+1]
					if target != None and target.player != self.player:
						returnMoves.append(
							MoveCommand(self.row, self.col, self.row+1, self.col+1, MoveCommandType.CAPTURE)
						)

				# Pawn Forward
				if chessBoard.board[self.row+1][self.col] == None:
					returnMoves.append(
						MoveCommand(self.row, self.col, self.row+1, self.col, MoveCommandType.MOVE)
					)

		# Validate For King Safety
		return [move for move in returnMoves if chessBoard.validateKingSafety(move)]

	def captureTargets(self, chessBoard: ChessBoardProtocal) -> list[tuple[int, int]]:
		returnMoves = []
		if self.player == Player.BLACK:
			if self.row > 0:
				if self.col > 0:
					returnMoves.append((self.row-1, self.col-1))
				if self.col < 7:
					returnMoves.append((self.row-1, self.col+1))

		elif self.player == Player.WHITE:
			if self.row < 8:
				if self.col > 0:
					returnMoves.append((self.row+1, self.col-1))
				if self.col < 7:
					returnMoves.append((self.row+1, self.col+1))

		return returnMoves
