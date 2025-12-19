# Enum 
from appEnums import PieceType, Player, MoveCommandType

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

	@staticmethod
	def opponent(player: Player):
		if player == Player.WHITE:
			return Player.BLACK
		else:
			return Player.WHITE

	# Validate King Safety for Player
	def kingSafety(self, player: Player):
		opponent = ChessBoardModel.opponent(player)
		if player == Player.WHITE:
			if self.whiteKingSquare in self._allPlayerCaptureTargets(opponent):
				return False
		elif player == Player.BLACK:
			if self.blackKingSquare in self._allPlayerCaptureTargets(opponent):
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

