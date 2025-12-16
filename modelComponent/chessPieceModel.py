from appEnums import PieceType, Player, MoveCommandType

class ChessPieceModel:
	def __init__(self, player: Player, type: PieceType):
		self.player = player
		self.type = type

	@staticmethod
	def createInitChessBoard():
		heavyMaterialLine = [PieceType.ROOK, PieceType.KNIGHT, PieceType.BISHOP, PieceType.QUEEN, \
				PieceType.KING, PieceType.BISHOP, PieceType.KNIGHT, PieceType.ROOK]

		board = [[None for _ in range(8)] for _ in range(8)]
		
		# Poplute Black Pieces
		for col in range(0, 8):
			board[7][col] = ChessPieceModel(Player.BLACK, heavyMaterialLine[col])

		# Poplute Black Pawns
		for col in range(0, 8):
			board[6][col] = ChessPieceModel(Player.BLACK, PieceType.PAWN)

		# Poplute White Pawns
		for col in range(0, 8):
			board[1][col] = ChessPieceModel(Player.WHITE, PieceType.PAWN)

		# Poplute White Pieces
		for col in range(0, 8):
			board[0][col] = ChessPieceModel(Player.WHITE, heavyMaterialLine[col])

		return board

	def pieceValue(self):
		match self.type:
			case PieceType.KING:
				return 10**10
			case PieceType.QUEEN:
				return 900
			case PieceType.KNIGHT:
				return 300
			case PieceType.ROOK:
				return 500
			case PieceType.PAWN:
				return 100
			case PieceType.BISHOP:
				return 300