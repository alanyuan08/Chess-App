from PySide6.QtCore import Signal, QObject
from appEnums import Player

# Models
from modelComponent.moveCommand import MoveCommand
		
class CommunicatorProxy(QObject):
	# Item -> Controller
	move_request = Signal(int, int, int, int, Player)

	def signal_move_request(self, initRow: int, initCol: int, 
		newRow: int, newCol: int, player: Player):
		self.move_request.emit(initRow, initCol, newRow, newCol, player)

	# Controller -> Item
	update_request = Signal(MoveCommand, Player)

	def signal_update_request(self, moveCommand: MoveCommand, 
		newPlayerTurn: Player):
		self.update_request.emit(moveCommand, newPlayerTurn)

	# Controller -> Item
	player_lose = Signal(Player)

	def signal_player_lose(self, playerLose: Player):
		self.player_lose.emit(playerLose)
