# Enum
from appEnums import PieceType, Player, MoveCommandType

# Model
from modelComponent.chessBoardModel import ChessBoardModel

class MoveCommand:
	def __init__(self, startRow: int, startCol: int, endRow: int, 
		endCol, type: PieceType, player: Player):
		self.startRow = startRow
		self.startCol = startCol

		self.endRow = endRow
		self.endCol = endCol

		self.moveType = type
		self.player = player

	def __str__(self):
		return "(" + str(self.startRow) + ", " + str(self.startCol) + ") " + "(" + str(self.endRow) + ", " + str(self.endCol) + ") " + str(self.moveType)

	# Moves the Piece
	def movePiece(self, chessBoardModel: ChessBoardModel):
		# Set enPassant to Null - Reset this if the opponent does a double pawn move
		chessBoardModel.enPassantColumn = None

		match self.moveType:
			# Queen Side Castle
			case MoveCommandType.QUEENSIDECASTLE:
				if self.player == Player.BLACK:
					chessBoardModel.blackKingMoved = True
					chessBoardModel.blackQueenSideRookMoved = True

					# Move the King 2 steps to the left
					chessBoardModel._movePieceOnBoard(7, 4, 7, 2)

					# Move the Rook to the right of the king
					chessBoardModel._movePieceOnBoard(7, 0, 7, 3)

					# Set the Black King Square
					chessBoardModel.blackKingSquare = (7, 2)
					chessBoardModel.blackKingMoved = True

				elif self.player == Player.WHITE:
					chessBoardModel.whiteKingMoved = True
					chessBoardModel.whiteQueenSideRookMoved = True

					# Move the King 2 steps to the left
					chessBoardModel._movePieceOnBoard(0, 4, 0, 2)

					# Move the Rook to the right of the king
					chessBoardModel._movePieceOnBoard(0, 0, 0, 3)

					# Set the White King Square
					chessBoardModel.whiteKingSquare = (0, 2)
					chessBoardModel.whiteKingMoved = True

			# King Side Castle
			case MoveCommandType.KINGSIDECASTLE:
				if self.player == Player.BLACK:
					chessBoardModel.blackKingMoved = True
					chessBoardModel.blackQueenSideRookMoved = True

					# Move the King 2 steps to the right
					chessBoardModel._movePieceOnBoard(7, 4, 7, 6)

					# Move the Rook to the right of the king
					chessBoardModel._movePieceOnBoard(7, 7, 7, 5)

					# Set the Black King Square
					chessBoardModel.blackKingSquare = (7, 6)
					chessBoardModel.blackKingMoved = True

				elif self.player == Player.WHITE:
					chessBoardModel.whiteKingMoved = True
					chessBoardModel.whiteQueenSideRookMoved = True

					# Move the King 2 steps to the right
					chessBoardModel._movePieceOnBoard(0, 4, 0, 6)

					# Move the Rook to the right of the king
					chessBoardModel._movePieceOnBoard(0, 7, 0, 5)

					# Set the White King Square
					chessBoardModel.whiteKingSquare = (0, 6)
					chessBoardModel.whiteKingMoved = True

			# Move Piece
			case MoveCommandType.MOVE | MoveCommandType.CAPTURE:
				# Move Starting Piece to Capture Point
				chessBoardModel._movePieceOnBoard(self.startRow, self.startCol, self.endRow, self.endCol)

				movePiece = chessBoardModel.board[self.endRow][self.endCol]
				# Update the King Square
				if movePiece == KingModel():
					if self.player == Player.BLACK:
						chessBoardModel.blackKingSquare = (self.endRow, self.endCol)
						chessBoardModel.blackKingMoved = True
					elif self.player == Player.WHITE:
						chessBoardModel.whiteKingSquare = (self.endRow, self.endCol)
						chessBoardModel.whiteKingMoved = True

			# Double Pawn Move
			case MoveCommandType.PAWNOPENMOVE:
				# Move the piece from start to end
				chessBoardModel._movePieceOnBoard(self.startRow, self.startCol, self.endRow, self.endCol)

				chessBoardModel.enPassantColumn = self.endCol

			# Promote Pawn
			case MoveCommandType.PROMOTE:
				# Promote the Pawn to a Queen
				chessBoardModel.board[self.startRow][self.startCol] = None
				newQueen = ChessPieceFactory.createChessPiece(PieceType.QUEEN, self.player, self.endRow, self.endCol)
				chessBoardModel.board[self.endRow][self.endCol] = newQueen

			# En Passant
			case MoveCommandType.ENPASSANT:
				# Move the Pawn to the target
				chessBoardModel._movePieceOnBoard(self.startRow, self.startCol, self.endRow, self.endCol)

				# Remove En Passant Pawn
				chessBoardModel.board[self.startRow][self.endCol] = None

		# Swap the Player Turn
		chessBoardModel.playerTurn = ChessBoardModel.opponent(chessBoardModel.playerTurn)
		return 

