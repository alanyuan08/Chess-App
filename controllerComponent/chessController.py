from PySide6.QtCore import QObject

from appEnums import PieceType, Player, MoveCommandType
from communicatorProxy import CommunicatorProxy

# Models
from modelComponent.moveCommand import MoveCommand

# Controller 
class ChessBoardController(QObject):
	def __init__(self, chessBoard):
		self.communicatorProxy = CommunicatorProxy()
		self.communicatorProxy.update_request.connect(chessBoard.updatePosition)

		# Simulate DB Access
		self.chessBoardModels = []


	def on_move_executed(self, initRow, initCol, targetRow, targetCol, player):
		if player == self.playerTurn:
			possibleMoves = self.chessBoardModels[0]._possibleMoves(initRow, initCol, player)
			for moveCommand in possibleMoves:
				if moveCommand.endRow == targetRow and moveCommand.endCol == targetCol: 
					# Communicate the command to FrontEnd
					self.communicatorProxy.signal_update_request(moveCommand)
					# Update the Controller
					self.movePiece(moveCommand)
