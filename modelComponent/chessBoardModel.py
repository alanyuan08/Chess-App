# Enum 
from appEnums import PieceType, Player, MoveCommandType

# Factory
from modelFactory.chessPieceFactory import ChessPieceFactory

# Model
from modelComponent.moveCommand import MoveCommand

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
		self.whiteKingSquareRow = 0
		self.whiteKingSquareCol = 4

		self.blackKingSquareRow = 7
		self.blackKingSquareCol = 4

	# Danger Moves
	quiescenceMoveCmd = [MoveCommandType.PROMOTE, MoveCommandType.CAPTURE, MoveCommandType.ENPASSANT]

	# Store NonBoard States -> Used for Undo
	def nonBoardState(self):
		nonBoardStates = {}

		nonBoardStates['playerTurn'] = self.playerTurn 
		nonBoardStates['enPassantColumn'] = self.enPassantColumn 
		
		nonBoardStates['blackKingMoved'] = self.blackKingMoved
		nonBoardStates['blackKingSideRookMoved'] = self.blackKingSideRookMoved
		nonBoardStates['blackQueenSideRookMoved'] = self.blackQueenSideRookMoved

		nonBoardStates['whiteKingMoved'] = self.whiteKingMoved
		nonBoardStates['whiteKingSideRookMoved'] = self.whiteKingSideRookMoved
		nonBoardStates['whiteQueenSideRookMoved'] = self.whiteQueenSideRookMoved
		
		nonBoardStates['whiteKingSquareRow'] = self.whiteKingSquareRow
		nonBoardStates['whiteKingSquareCol'] = self.whiteKingSquareCol

		nonBoardStates['blackKingSquareRow'] = self.blackKingSquareRow
		nonBoardStates['blackKingSquareCol'] = self.blackKingSquareCol

		return nonBoardStates

	# Reset Board State
	def resetBoardState(self, nonBoardStates):

		self.playerTurn  = nonBoardStates['playerTurn']
		self.enPassantColumn = nonBoardStates['enPassantColumn']
		
		self.blackKingMoved = nonBoardStates['blackKingMoved'] 
		self.blackKingSideRookMoved = nonBoardStates['blackKingSideRookMoved'] 
		self.blackQueenSideRookMoved = nonBoardStates['blackQueenSideRookMoved']

		self.whiteKingMoved = nonBoardStates['whiteKingMoved']
		self.whiteKingSideRookMoved = nonBoardStates['whiteKingSideRookMoved']
		self.whiteQueenSideRookMoved = nonBoardStates['whiteQueenSideRookMoved']
		
		self.whiteKingSquareRow = nonBoardStates['whiteKingSquareRow'] 
		self.whiteKingSquareCol = nonBoardStates['whiteKingSquareCol']

		self.blackKingSquareRow = nonBoardStates['blackKingSquareRow']
		self.blackKingSquareCol = nonBoardStates['blackKingSquareCol']

		return

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
		if targetPiece == None:
			return None

		if targetPiece.player == player:
			possibleMoves = targetPiece.possibleMoves(self)
			for cmd in possibleMoves:
				if cmd.endRow == targetRow and cmd.endCol == targetCol:
					return cmd

		return None

	# Validate King Safety for player after making move
	def validateKingSafety(self, cmd: MoveCommand):
		# This is explictly defined here to avoid confusion after the move
		currentPlayer = self.playerTurn

		nonBoardState = self.nonBoardState()
		removedPiece = self.movePiece(cmd)

		if currentPlayer == Player.BLACK:
			kingTuple = (self.blackKingSquareRow, self.blackKingSquareCol)
			if kingTuple in self._allPlayerCaptureTargets(Player.WHITE):
				return False
		elif currentPlayer == Player.WHITE:
			kingTuple = (self.whiteKingSquareRow, self.whiteKingSquareCol)
			if kingTuple in self._allPlayerCaptureTargets(Player.BLACK):
				return False

		self.undoMove(cmd, removedPiece)
		self.resetBoardState(nonBoardState)

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
			if cmd.moveType in self.quiescenceMoveCmd:
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
		noisyMoves = self.allQuiesceneMoves()

		if len(noisyMoves) == 0:
			return self.computeBoardValue()

		if maximizingPlayer:
			bestValue = float('-inf')
			for cmd in noisyMoves:
				nonBoardState = self.nonBoardState()
				removedPiece = self.movePiece(cmd)

				computeValue = self.minMaxQuiesceSearch(False)
				bestValue = max(bestValue, computeValue)

				self.undoMove(cmd, removedPiece)
				self.resetBoardState(nonBoardState)

			return bestValue

		else:
			worstValue = float('inf')
			for cmd in noisyMoves:
				nonBoardState = self.nonBoardState()
				removedPiece = self.movePiece(cmd)

				computeValue = self.minMaxQuiesceSearch(True)
				worstValue = min(worstValue, computeValue)

				self.undoMove(cmd, removedPiece)
				self.resetBoardState(nonBoardState)

			return worstValue

	# MinMaxSearch -> General
	def minMaxSearch(self, maximizingPlayer, depth):
		# Termination Condition
		if depth == 0:
			if maximizingPlayer:
				return self.minMaxQuiesceSearch(False)
			else:
				return self.minMaxQuiesceSearch(True)

		else:
			if maximizingPlayer:
				bestValue = float('-inf')
				for cmd in self.allValidMoves():
					nonBoardState = self.nonBoardState()
					removedPiece = self.movePiece(cmd)

					computeValue = self.minMaxSearch(False, depth - 1)
					bestValue = max(bestValue, computeValue)

					self.undoMove(cmd, removedPiece)
					self.resetBoardState(nonBoardState)

				return bestValue

			else:
				worstValue = float('inf')
				for cmd in self.allValidMoves():
					nonBoardState = self.nonBoardState()
					removedPiece = self.movePiece(cmd)

					computeValue = self.minMaxSearch(True, depth - 1)
					worstValue = min(worstValue, computeValue)

					self.undoMove(cmd, removedPiece)
					self.resetBoardState(nonBoardState)

				return worstValue

	# Compute Best Move Using Min-Max
	def computeBestMove(self):
		returnCmd = None
		worstValue = float('inf')

		for cmd in self.allValidMoves():
			nonBoardState = self.nonBoardState()
			removedPiece = self.movePiece(cmd)

			returnValue = self.minMaxSearch(False, 0)
			if returnValue < worstValue:
				worstValue = min(worstValue, returnValue)
				returnCmd = cmd
								
			self.undoMove(cmd, removedPiece)
			self.resetBoardState(nonBoardState)

		return returnCmd

	# This is used to determine king safety and castle
	def _allPlayerCaptureTargets(self, player):
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
				self.blackKingSquareRow = endRow
				self.blackKingSquareCol = endCol
				self.blackKingMoved = True

			elif movePiece.player == Player.WHITE:
				self.whiteKingSquareRow = endRow
				self.whiteKingSquareCol = endCol
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

	def _undoMoveOnBoard(self, originalRow: int, originalCol: int, currentRow: int, currentCol: int):
		movePiece = self.board[currentRow][currentCol]
		self.board[originalRow][originalCol] = movePiece
		movePiece.row = originalRow
		movePiece.col = originalCol

		self.board[currentRow][currentCol] = None

	def undoMove(self, cmd: MoveCommand, restorePiece):
		# Swap the Player Turn
		self.playerTurn = ChessBoardModel.opponent(self.playerTurn)

		match cmd.moveType:
			# Queen Side Castle (Undo)
			case MoveCommandType.QUEENSIDECASTLE:
				if self.playerTurn == Player.BLACK:
					# Move the King 2 steps to the left
					self._undoMoveOnBoard(7, 4, 7, 2)

					# Move the Rook to the right of the king
					self._undoMoveOnBoard(7, 0, 7, 3)

				elif self.playerTurn == Player.WHITE:
					# Move the King 2 steps to the left
					self._undoMoveOnBoard(0, 4, 0, 2)

					# Move the Rook to the right of the king
					self._undoMoveOnBoard(0, 0, 0, 3)

			# King Side Castle
			case MoveCommandType.KINGSIDECASTLE:
				if self.playerTurn == Player.BLACK:
					# Move the King 2 steps to the right
					self._undoMoveOnBoard(7, 4, 7, 6)

					# Move the Rook to the right of the king
					self._undoMoveOnBoard(7, 7, 7, 5)

				elif self.playerTurn == Player.WHITE:
					# Move the King 2 steps to the right
					self._undoMoveOnBoard(0, 4, 0, 6)

					# Move the Rook to the right of the king
					self._undoMoveOnBoard(0, 7, 0, 5)

			# Move Piece
			case MoveCommandType.MOVE | MoveCommandType.CAPTURE:
				# Move Starting Piece to Capture Point
				self._undoMoveOnBoard(cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

				# Restore Piece
				self.board[cmd.endRow][cmd.endCol] = restorePiece

			# Double Pawn Move
			case MoveCommandType.PAWNOPENMOVE:
				# Move the piece from start to end
				self._undoMoveOnBoard(cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

			# Promote Pawn
			case MoveCommandType.PROMOTE:
				# Promote the Pawn to a Queen
				self.board[cmd.endRow][cmd.endCol] = None

				# Store Removed Piece
				self.board[cmd.startRow][cmd.endCol] = restorePiece

			# En Passant
			case MoveCommandType.ENPASSANT:
				# Move the Pawn to the target
				self._undoMoveOnBoard(cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

				# Store Removed Piece
				self.board[cmd.startRow][cmd.endCol] = restorePiece

		return 

	def movePiece(self, cmd: MoveCommand):
		# Set enPassant to Null - Reset this if the opponent does a double pawn move
		self.enPassantColumn = None

		# Captured Piece 
		removedPiece = None

		match cmd.moveType:
			# Queen Side Castle
			case MoveCommandType.QUEENSIDECASTLE:
				if self.playerTurn == Player.BLACK:
					# Move the King 2 steps to the left
					self._movePieceOnBoard(7, 4, 7, 2)

					# Move the Rook to the right of the king
					self._movePieceOnBoard(7, 0, 7, 3)

				elif self.playerTurn == Player.WHITE:
					# Move the King 2 steps to the left
					self._movePieceOnBoard(0, 4, 0, 2)

					# Move the Rook to the right of the king
					self._movePieceOnBoard(0, 0, 0, 3)

			# King Side Castle
			case MoveCommandType.KINGSIDECASTLE:
				if self.playerTurn == Player.BLACK:
					# Move the King 2 steps to the right
					self._movePieceOnBoard(7, 4, 7, 6)

					# Move the Rook to the right of the king
					self._movePieceOnBoard(7, 7, 7, 5)

				elif self.playerTurn == Player.WHITE:
					# Move the King 2 steps to the right
					self._movePieceOnBoard(0, 4, 0, 6)

					# Move the Rook to the right of the king
					self._movePieceOnBoard(0, 7, 0, 5)

			# Move Piece
			case MoveCommandType.MOVE | MoveCommandType.CAPTURE:
				# Store the Removed Piece
				removedPiece = self.board[cmd.startRow][cmd.startCol]

				# Move Starting Piece to Capture Point
				self._movePieceOnBoard(cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

			# Double Pawn Move
			case MoveCommandType.PAWNOPENMOVE:
				# Move the piece from start to end
				self._movePieceOnBoard(cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

				self.enPassantColumn = cmd.endCol

			# Promote Pawn
			case MoveCommandType.PROMOTE:
				# Store Removed Piece
				removedPiece = self.board[cmd.startRow][cmd.endCol]

				self.board[cmd.endRow][cmd.endCol] = ChessPieceFactory.createChessPiece(PieceType.QUEEN, self.playerTurn, cmd.endRow, cmd.endCol)

			# En Passant
			case MoveCommandType.ENPASSANT:
				# Store Removed Piece
				removedPiece = self.board[cmd.startRow][cmd.endCol]

				# Move the Pawn to the target
				self._movePieceOnBoard(cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

				# Remove En Passant Pawn
				self.board[cmd.startRow][cmd.endCol] = None

		# Swap the Player Turn
		self.playerTurn = ChessBoardModel.opponent(self.playerTurn)

		return removedPiece
