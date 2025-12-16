from PySide6.QtCore import Signal, QObject
from appEnums import Player
from modelComponent import MoveCommand
		
class CommunicatorProxy(QObject):
	# Item -> Controller
	move_request = Signal(int, int, int, int, Player)

	def signal_move_request(self, initRow, initCol, newRow, newCol, player):
		self.move_request.emit(initRow, initCol, newRow, newCol, player)

	# Controller -> Item
	update_request = Signal(MoveCommand)

	def signal_update_request(self, moveCommand: MoveCommand):
		self.update_request.emit(moveCommand)
