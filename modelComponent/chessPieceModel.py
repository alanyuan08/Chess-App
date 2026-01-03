# Enum
from appEnums import PieceType, Player, MoveCommandType

# Abstract Base Class
from abc import ABC, abstractmethod

# Import Model
from modelComponent.moveCommand import MoveCommand
from modelComponent.chessBoardProtocal import ChessBoardProtocal

# Used to test next move
import copy

class ChessPieceModel(ABC):
	def __init__(self, player: Player, row: int, col: int):
		pass

	@abstractmethod
	def phaseWeight(self) -> int:
		pass

	@abstractmethod
	def pieceValue(self) -> int:
		pass

	@abstractmethod
	def computedValue(self, chessBoard: ChessBoardProtocal, phaseWeight: int) -> int:
		pass

	@abstractmethod
	def possibleMoves(self, chessBoard: ChessBoardProtocal) -> list[MoveCommand]:
		pass

	@abstractmethod
	def captureTargets(self, chessBoard: ChessBoardProtocal) -> list[tuple[int, int]]:
		pass
