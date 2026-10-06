from PySide6.QtCore import Signal, QObject
from appEnums import Player, GameState

# Models
from modelComponent.moveCommand import MoveCommand
		
class CommunicatorProxy(QObject):
	# Item -> Controller
	moveRequest = Signal(int, int, int, int, Player)

	def signalMoveRequest(self, initRow: int, initCol: int, 
		newRow: int, newCol: int, player: Player):
		self.moveRequest.emit(initRow, initCol, newRow, newCol, player)

	# Controller -> Item
	updateRequest = Signal(MoveCommand)

	def signalUpdateRequest(self, moveCommand: MoveCommand):
		self.updateRequest.emit(moveCommand)

	# Controller -> Game state
	updateGameState = Signal(GameState, Player)

	def signalUpdateGameState(self, gameState: GameState, playerTurn: Player):
		self.updateGameState.emit(gameState, playerTurn)
