# Enum
from appEnums import PieceType, Player, MoveCommandType

# Import Models
from modelComponent.chessBoardModel import ChessBoardModel

# Abstract Base Class
from abc import ABC, abstractmethod

class ChessPieceModel(ABC):
	def __init__(self, player: Player, row: int, col: int):
		pass

	@abstractmethod
	def pieceValue(self):
		pass

	@abstractmethod
	def possibleMoves(self, chessBoardModel: ChessBoardModel):
		pass

	@abstractmethod
	def captureTargets(self, chessBoardModel: ChessBoardModel):
		pass

	# Validate targetRow / targetCol is valid
	def validateMoveTarget(self, targetRow: int, targetCol: int, 
		chessBoardModel: ChessBoardModel):

		# Validate the Move Command is a Possible Move
		possibleMoves = self.possibleMoves(chessBoardModel)
		for cmd in possibleMoves:
			if cmd.endRow == targetRow and cmd.endCol == targetCol:
				return cmd

		return None


