# Enum
from appEnums import PieceType, Player, MoveCommandType

class MoveCommand:
	def __init__(self, startRow: int, startCol: int, endRow: int, 
		endCol, type: PieceType):
		self.startRow = startRow
		self.startCol = startCol

		self.endRow = endRow
		self.endCol = endCol

		self.moveType = type

	def __str__(self):
		return "(" + str(self.startRow) + ", " + str(self.startCol) + ") " \
		+ "(" + str(self.endRow) + ", " + str(self.endCol) + ") " + str(self.moveType)


