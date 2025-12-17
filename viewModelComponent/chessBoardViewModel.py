from appEnums import PieceType, Player, MoveCommandType
from communicatorProxy import CommunicatorProxy

# Models
from modelComponent.moveCommand import MoveCommand
from modelComponent.chessBoardModel import ChessBoardModel

# Controller 
class ChessBoardViewModel():
	def __init__(self, chessBoardView, chessBoardModel):
		self.communicatorProxy = CommunicatorProxy()

		# Backend ChessBoard Model
		self.chessBoardModel = chessBoardModel
		self.communicatorProxy.update_request.connect(chessBoardView.updatePosition)
		chessBoardView.connectViewModel(self)

	def on_move_executed(self, initRow: int, initCol: int, targetRow: int, 
			targetCol: int, player: Player):

		moveCommand = self.chessBoardModel.validateAndReturnCommand(initRow, 
			initCol, targetRow, targetCol, player)

		if moveCommand != None:
			# Move for the Chess Model
			self.chessBoardModel.movePiece(moveCommand)
			# Communicate the command to FrontEnd
			self.communicatorProxy.signal_update_request(moveCommand)

			# Opponent Takes Turn
			opponentPlayer = ChessBoardModel.returnOpponent(player)
			opponentCmd = self.chessBoardModel.computeBestValue(opponentPlayer)

			if opponentCmd != None:
				self.chessBoardModel.movePiece(opponentCmd)
				# Communicate the command to FrontEnd
				self.communicatorProxy.signal_update_request(opponentCmd)
			else:
				print("YOU WIN")