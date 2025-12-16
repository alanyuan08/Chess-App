from appEnums import PieceType, Player, MoveCommandType
from communicatorProxy import CommunicatorProxy

# Models
from modelComponent.moveCommand import MoveCommand

# Controller 
class ChessBoardController():
	def __init__(self):
		self.communicatorProxy = CommunicatorProxy()

		# Backend ChessBoard Model
		self.chessBoardModel = None

	def addChessBoard(self, chessBoardView, chessBoardModel):
		self.chessBoardModel = chessBoardModel

		self.communicatorProxy.update_request.connect(chessBoardView.updatePosition)

	def on_move_executed(self, initRow: int, initCol: int, targetRow: int, 
			targetCol: int, player: Player):

		# Attempt to move ChessModel Piece
		moveCommand = self.chessBoardModel.movePiece(initRow, initCol, targetRow, targetCol, player)

		if moveCommand != None:
			# Communicate the command to FrontEnd
			self.communicatorProxy.signal_update_request(moveCommand)
