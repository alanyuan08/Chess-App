# Enum 
from appEnums import PieceType, Player, MoveCommandType

# Factory
from modelFactory.chessPieceFactory import ChessPieceFactory

# Model
from modelComponent.moveCommand import MoveCommand

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

	# Danger Moves
	quiescenceMoveCmd = [MoveCommandType.PROMOTE, MoveCommandType.CAPTURE, MoveCommandType.ENPASSANT]

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
	def validateKingSafety(self, cmd: MoveCommand):
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

	# Return all Valid moves for currentPlayer
	def allValidMoves(self):
		validMoves = []

		for row in range(0, 8):
			for col in range(0, 8):
				if self.board[row][col] != None:
					if self.board[row][col].player == self.playerTurn:
						possibleMoves = self.board[row][col].possibleMoves(self)
						validMoves += possibleMoves

		return validMoves

	# Compute Board Value - Assume White is the Protagonist
	def computeBoardValue(self):
		returnValue = 0

		for row in range(0, 8):
			for col in range(0, 8):
				if self.board[row][col] != None:
					if self.board[row][col].player == Player.WHITE:
						returnValue += self.board[row][col].pieceValue()
					else:
						returnValue -= self.board[row][col].pieceValue()

		return returnValue

	# Return all Capture Moves
	def allQuiesceneMoves(self):
		validMoves = []

		for cmd in self.allValidMoves():
			target = self.board[cmd.endRow][cmd.endCol]
			if target != None and target.type == PieceType.KING and cmd.moveType in self.quiescenceMoveCmd:
				validMoves.append(cmd)

		return validMoves

	# Validate King Safety for player after making move
	def quietState(self):
		captureMoves = self.allQuiesceneMoves()

		if len(captureMoves) == 0:
			return True
		else:
			return False

	# MinMaxSearch -> Quiesce for Horizon Problem
	def minMaxQuiesceSearch(self, maximizingPlayer):
		# Termination Condition
		if self.quietState():
			return self.computeBoardValue()

		if maximizingPlayer:
			bestValue = float('-inf')
			for cmd in self.allQuiesceneMoves():
				testBoard = copy.deepcopy(self)
				testBoard.movePiece(cmd)

				computeValue = testBoard.minMaxQuiesceSearch(False)
				bestValue = max(bestValue, computeValue)
				del testBoard

			return bestValue

		else:
			bestValue = float('inf')
			for cmd in self.allQuiesceneMoves():
				testBoard = copy.deepcopy(self)
				testBoard.movePiece(cmd)

				computeValue = testBoard.minMaxQuiesceSearch(True)
				bestValue = min(bestValue, computeValue)
				del testBoard

			return bestValue

	# MinMaxSearch -> General
	def minMaxSearch(self, maximizingPlayer, depth):
		# Termination Condition
		if depth <= 1:
			if self.quietState():
				return self.computeBoardValue()
			else:
				return self.minMaxQuiesceSearch(maximizingPlayer)

		else:
			if maximizingPlayer:
				bestValue = float('-inf')
				for cmd in self.allValidMoves():
					testBoard = copy.deepcopy(self)
					testBoard.movePiece(cmd)

					computeValue = testBoard.minMaxSearch(False, depth - 1)
					bestValue = max(bestValue, computeValue)
					del testBoard

				return bestValue

			else:
				worstValue = float('inf')
				for cmd in self.allValidMoves():
					testBoard = copy.deepcopy(self)
					testBoard.movePiece(cmd)

					computeValue = testBoard.minMaxSearch(True, depth - 1)
					worstValue = min(worstValue, computeValue)
					del testBoard

				return worstValue

	# Compute Best Move Using Min-Max
	def computeBestMove(self):
		returnCmd = None
		worstValue = float('inf')

		for cmd in self.allValidMoves():
			testBoard = copy.deepcopy(self)
			testBoard.movePiece(cmd)
			returnValue = testBoard.minMaxSearch(False, 2)
			del testBoard	

			if returnValue < worstValue:
				worstValue = min(worstValue, returnValue)
				returnCmd = cmd

		return returnCmd

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

		# Update the King Square
		if movePiece.type == PieceType.KING:
			if movePiece.player == Player.BLACK:
				self.blackKingSquare = (endRow, endCol)
				self.blackKingMoved = True

			elif movePiece.player == Player.WHITE:
				self.whiteKingSquare = (endRow, endCol)
				self.whiteKingMoved = True

		# Update Rook 
		if movePiece.type == PieceType.ROOK:
			if movePiece.player == Player.BLACK:
				if startCol == 0:
					self.blackQueenSideRookMoved = True
				elif startCol == 7:
					self.blackKingSideRookMoved = True

			elif movePiece.player == Player.WHITE:
				if startCol == 0:
					self.whiteQueenSideRookMoved = True
				elif startCol == 7:
					self.whiteKingSideRookMoved = True	

	def movePiece(self, cmd: MoveCommand):
		# Set enPassant to Null - Reset this if the opponent does a double pawn move
		self.enPassantColumn = None

		match cmd.moveType:
			# Queen Side Castle
			case MoveCommandType.QUEENSIDECASTLE:
				if cmd.player == Player.BLACK:
					# Move the King 2 steps to the left
					self._movePieceOnBoard(7, 4, 7, 2)

					# Move the Rook to the right of the king
					self._movePieceOnBoard(7, 0, 7, 3)

				elif cmd.player == Player.WHITE:
					# Move the King 2 steps to the left
					self._movePieceOnBoard(0, 4, 0, 2)

					# Move the Rook to the right of the king
					self._movePieceOnBoard(0, 0, 0, 3)

			# King Side Castle
			case MoveCommandType.KINGSIDECASTLE:
				if cmd.player == Player.BLACK:
					# Move the King 2 steps to the right
					self._movePieceOnBoard(7, 4, 7, 6)

					# Move the Rook to the right of the king
					self._movePieceOnBoard(7, 7, 7, 5)

				elif cmd.player == Player.WHITE:
					# Move the King 2 steps to the right
					self._movePieceOnBoard(0, 4, 0, 6)

					# Move the Rook to the right of the king
					self._movePieceOnBoard(0, 7, 0, 5)

			# Move Piece
			case MoveCommandType.MOVE | MoveCommandType.CAPTURE:
				# Move Starting Piece to Capture Point
				self._movePieceOnBoard(cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

				movePiece = self.board[cmd.endRow][cmd.endCol]

			# Double Pawn Move
			case MoveCommandType.PAWNOPENMOVE:
				# Move the piece from start to end
				self._movePieceOnBoard(cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

				self.enPassantColumn = cmd.endCol

			# Promote Pawn
			case MoveCommandType.PROMOTE:
				# Promote the Pawn to a Queen
				self._movePieceOnBoard(cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

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