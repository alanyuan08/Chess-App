# Enum
from appEnums import PieceType, Player, MoveCommandType

# Import Models
from modelComponent.kingModel import KingModel
from modelComponent.queenModel import QueenModel
from modelComponent.knightModel import KnightModel
from modelComponent.rookModel import RookModel
from modelComponent.pawnModel import PawnModel
from modelComponent.bishopModel import BishopModel
from modelComponent.chessPieceModel import ChessPieceModel

class ChessPieceFactory:

	@staticmethod
	def createChessPiece(type: PieceType, 
		player: Player, row: int, col: int) -> ChessPieceModel:
		match type:
			case PieceType.KING:
				return KingModel(player, row, col)
			case PieceType.QUEEN:
				return QueenModel(player, row, col)
			case PieceType.KNIGHT:
				return KnightModel(player, row, col)
			case PieceType.ROOK:
				return RookModel(player, row, col)
			case PieceType.PAWN:
				return PawnModel(player, row, col)
			case PieceType.BISHOP:
				return BishopModel(player, row, col)

