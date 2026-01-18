from PySide6.QtCore import Signal, QObject
from appEnums import Player, GameState

# Models
from modelComponent.moveCommand import MoveCommand
		
class CommunicatorProxy(QObject):
	# Item -> Controller
	move_request = Signal(int, int, int, int, Player)

	def signal_move_request(self, initRow: int, initCol: int, 
		newRow: int, newCol: int, player: Player):
		self.move_request.emit(initRow, initCol, newRow, newCol, player)

	# Controller -> Item
	update_request = Signal(MoveCommand)

	def signal_update_request(self, moveCommand: MoveCommand):
		self.update_request.emit(moveCommand)

	# Controller -> Game state
	game_state = Signal(GameState, Player)

	def signal_game_state(self, gameState: GameState, playerTurn: Player):
		self.game_state.emit(gameState, playerTurn)
