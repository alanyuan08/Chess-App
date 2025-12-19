# Enum
from appEnums import PieceType, Player, MoveCommandType

# Abstract Base Class
from abc import ABC, abstractmethod

# Used to test next move
import copy

class ChessPieceModel(ABC):
	def __init__(self, player: Player, row: int, col: int):
		pass

	@abstractmethod
	def pieceValue(self):
		pass

	@abstractmethod
	def possibleMoves(self, chessBoardModel):
		pass

	@abstractmethod
	def captureTargets(self, chessBoardModel):
		pass
