from appEnums import PieceType, Player, MoveCommandType

class ChessPieceModel:
	def __init__(self, player: Player, type: PieceType):
		self.player = player
		self.type = type

	@staticmethod
	def returnChessPieceProperties():
		returnPlacements = []

		for player in [[7, Player.BLACK], [0, Player.WHITE]]:
			placement = [PieceType.ROOK, PieceType.KNIGHT, PieceType.BISHOP, PieceType.QUEEN, \
				PieceType.KING, PieceType.BISHOP, PieceType.KNIGHT, PieceType.ROOK]

			for i in range(0, 8):
				item = [player[0], i, player[1], placement[i]]
				returnPlacements.append(item)

		for player in [[6, Player.BLACK], [1, Player.WHITE]]:

			for i in range(0, 8):
				item = [player[0], i, player[1], PieceType.PAWN]
				returnPlacements.append(item)

		return returnPlacements


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