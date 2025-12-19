# Enum 
from appEnums import PieceType, Player, MoveCommandType

# Factory
from modelFactory.chessPieceFactory import ChessPieceFactory

import copy

# Controller 
class ChessBoardModel():
	def __init__(self):
		# Create the Chess Board
		self.board = []

		# Init Player as White
		self.playerTurn = Player.WHITE

		# Set Human Player
		self.humanPlayers = []

		# En Passant Column - Set after pawn move, then cleared 
		self.enPassantColumn = None

		# Used to check if player can Castle
		self.blackKingMoved = False
		self.blackKingSideRookMoved = False
		self.blackQueenSideRookMoved = False

		self.whiteKingMoved = False
		self.whiteKingSideRookMoved = False
		self.whiteQueenSideRookMoved = False 

		# Used to Check for King Safety
		self.whiteKingSquare = (0, 4)
		self.blackKingSquare = (7, 4)

	@staticmethod
	def opponent(player: Player):
		if player == Player.WHITE:
			return Player.BLACK
		else:
			return Player.WHITE

	# Validate the Move
	def validateMove(self, initRow: int, initCol: int, targetRow: int, targetCol: int, player: Player):
		# It's not your turn to move
		if player != self.playerTurn:
			return None

		# Validate the Move Command is a Possible Move
		targetPiece = self.board[initRow][initCol]

		if targetPiece.player == player:
			possibleMoves = targetPiece.possibleMoves(self)
			for cmd in possibleMoves:
				if cmd.endRow == targetRow and cmd.endCol == targetCol:
					return cmd

		return None

	# Validate King Safety for player after making move
	def validateKingSafety(self, cmd):
		testBoard = copy.deepcopy(self)
		testBoard.movePiece(cmd)

		opponent = ChessBoardModel.opponent(cmd.player)
		if cmd.player == Player.WHITE:
			if testBoard.whiteKingSquare in testBoard._allPlayerCaptureTargets(opponent):
				return False
		elif cmd.player == Player.BLACK:
			if testBoard.blackKingSquare in testBoard._allPlayerCaptureTargets(opponent):
				return False

		return True

	# This is used to determine king safety and castle
	def _allPlayerCaptureTargets(self, player: Player):
		attackSquare = {}
		for row in range(0, 8):
			for col in range(0, 8):
				targetPiece = self.board[row][col]
				if targetPiece != None and targetPiece.player == player:

					possibleMoves = targetPiece.captureTargets(self)
					for move in possibleMoves:
						attackSquare[move] = True
		
		return attackSquare

	# Move Piece on ChessBoard
	def _movePieceOnBoard(self, startRow: int, startCol: int, endRow: int, endCol: int):
		movePiece = self.board[startRow][startCol]
		self.board[endRow][endCol] = movePiece
		self.board[startRow][startCol] = None

		movePiece.row = endRow
		movePiece.col = endCol

	def movePiece(self, cmd):
		# Set enPassant to Null - Reset this if the opponent does a double pawn move
		self.enPassantColumn = None

		match cmd.moveType:
			# Queen Side Castle
			case MoveCommandType.QUEENSIDECASTLE:
				if cmd.player == Player.BLACK:
					self.blackKingMoved = True
					self.board.blackQueenSideRookMoved = True

					# Move the King 2 steps to the left
					self._movePieceOnBoard(7, 4, 7, 2)

					# Move the Rook to the right of the king
					self._movePieceOnBoard(7, 0, 7, 3)

					# Set the Black King Square
					self.blackKingSquare = (7, 2)
					self.blackKingMoved = True

				elif cmd.player == Player.WHITE:
					self.whiteKingMoved = True
					self.whiteQueenSideRookMoved = True

					# Move the King 2 steps to the left
					self._movePieceOnBoard(0, 4, 0, 2)

					# Move the Rook to the right of the king
					self._movePieceOnBoard(0, 0, 0, 3)

					# Set the White King Square
					self.whiteKingSquare = (0, 2)
					self.whiteKingMoved = True

			# King Side Castle
			case MoveCommandType.KINGSIDECASTLE:
				if cmd.player == Player.BLACK:
					self.blackKingMoved = True
					self.blackQueenSideRookMoved = True

					# Move the King 2 steps to the right
					self._movePieceOnBoard(7, 4, 7, 6)

					# Move the Rook to the right of the king
					self._movePieceOnBoard(7, 7, 7, 5)

					# Set the Black King Square
					self.blackKingSquare = (7, 6)
					self.blackKingMoved = True

				elif cmd.player == Player.WHITE:
					self.whiteKingMoved = True
					self.whiteQueenSideRookMoved = True

					# Move the King 2 steps to the right
					self._movePieceOnBoard(0, 4, 0, 6)

					# Move the Rook to the right of the king
					self._movePieceOnBoard(0, 7, 0, 5)

					# Set the White King Square
					self.whiteKingSquare = (0, 6)
					self.whiteKingMoved = True

			# Move Piece
			case MoveCommandType.MOVE | MoveCommandType.CAPTURE:
				# Move Starting Piece to Capture Point
				self._movePieceOnBoard(cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

				movePiece = self.board[cmd.endRow][cmd.endCol]
				# Update the King Square
				if movePiece.type == PieceType.KING:
					if cmd.player == Player.BLACK:
						self.blackKingSquare = (cmd.endRow, cmd.endCol)
						self.blackKingMoved = True
					elif cmd.player == Player.WHITE:
						self.whiteKingSquare = (cmd.endRow, cmd.endCol)
						self.whiteKingMoved = True

			# Double Pawn Move
			case MoveCommandType.PAWNOPENMOVE:
				# Move the piece from start to end
				self._movePieceOnBoard(cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

				self.enPassantColumn = cmd.endCol

			# Promote Pawn
			case MoveCommandType.PROMOTE:
				# Promote the Pawn to a Queen
				self.board[cmd.startRow][cmd.startCol] = None
				self.board[cmd.endRow][cmd.endCol] = ChessPieceFactory.createChessPiece(PieceType.QUEEN, cmd.player, cmd.endRow, cmd.endCol)

			# En Passant
			case MoveCommandType.ENPASSANT:
				# Move the Pawn to the target
				self._movePieceOnBoard(cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

				# Remove En Passant Pawn
				self.board[cmd.startRow][cmd.endCol] = None

		# Swap the Player Turn
		self.playerTurn = ChessBoardModel.opponent(self.playerTurn)
		return 