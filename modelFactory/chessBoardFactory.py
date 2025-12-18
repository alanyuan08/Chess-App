# Enum
from appEnums import PieceType, Player, MoveCommandType

# Import Factory
from modelFactory.chessPieceFactory import ChessPieceFactory

# Import Models
from modelComponent.chessBoardModel import ChessBoardModel

class ChessBoardFactory:

	@staticmethod
	def createChessBoard(humanPlayers: list[Player]):
		newBoard = ChessBoardModel()

		materialLine = [PieceType.ROOK, PieceType.KNIGHT, PieceType.BISHOP, PieceType.QUEEN, \
				PieceType.KING, PieceType.BISHOP, PieceType.KNIGHT, PieceType.ROOK]

		newBoard.board = [[None for _ in range(8)] for _ in range(8)]
		newBoard.humanPlayers = humanPlayers
		
		# Poplute Black Pieces
		for col in range(0, 8):
			type = materialLine[col]
			newBoard.board[7][col] = ChessPieceFactory.createChessPiece(type, Player.BLACK, 7, col)

		# Poplute Black Pawns
		for col in range(0, 8):
			newBoard.board[6][col] = ChessPieceFactory.createChessPiece(PieceType.PAWN, Player.BLACK, 6, col)

		# Poplute White Pawns
		for col in range(0, 8):
			newBoard.board[1][col] = ChessPieceFactory.createChessPiece(PieceType.PAWN, Player.WHITE, 1, col)

		# Poplute White Pieces
		for col in range(0, 8):
			type = materialLine[col]
			newBoard.board[0][col] = ChessPieceFactory.createChessPiece(type, Player.WHITE, 0, col)

		return newBoard
