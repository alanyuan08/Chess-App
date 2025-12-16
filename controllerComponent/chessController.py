from PySide6.QtCore import QObject

from appEnums import PieceType, Player, MoveCommandType
from communicatorProxy import CommunicatorProxy

# Models
from modelComponent.moveCommand import MoveCommand

# Controller 
class ChessBoardController(QObject):
	def __init__(self):
		self.communicatorProxy = CommunicatorProxy()

		# Backend ChessBoard Model
		self.chessBoardModel = None

	def addChessBoard(self, chessBoardView, chessBoardModel):
		self.chessBoardModel = chessBoardModel

		self.communicatorProxy.update_request.connect(chessBoardView.updatePosition)

	def on_move_executed(self, initRow, initCol, targetRow, targetCol, player):
		if player == self.chessBoardModel.playerTurn:
			possibleMoves = self.chessBoardModel._possibleMoves(initRow, initCol, player)
			for moveCommand in possibleMoves:
				if moveCommand.endRow == targetRow and moveCommand.endCol == targetCol: 
					# Communicate the command to FrontEnd
					self.communicatorProxy.signal_update_request(moveCommand)
					# Update the Controller
					self.chessBoardModel.movePiece(moveCommand)
