from appEnums import PieceType, Player, MoveCommandType
from communicatorProxy import CommunicatorProxy

# Models
from modelComponent.moveCommand import MoveCommand
from modelComponent.chessBoardModel import ChessBoardModel

# QTCore
from PySide6.QtCore import QRunnable, QThreadPool, Slot

# Controller 
class ChessBoardViewModel():
    def __init__(self, chessBoardView, chessBoardModel):
        self.communicatorProxy = CommunicatorProxy()

        # Backend ChessBoard Model
        self.chessBoardModel = chessBoardModel
        self.communicatorProxy.update_request.connect(chessBoardView.updatePosition)
        chessBoardView.connectViewModel(self)

        # Compute Opponent Turn ThreadPool 
        self.threadpool = QThreadPool()

    def on_move_executed(self, initRow: int, initCol: int, targetRow: int, 
            targetCol: int, player: Player):

        moveCommand = self.chessBoardModel.validateMove(initRow, 
            initCol, targetRow, targetCol, player)

        if moveCommand != None:
            # Move for the Chess Model
            self.chessBoardModel.movePiece(moveCommand)
            # Communicate the command to FrontEnd
            self.communicatorProxy.signal_update_request(moveCommand)

            # Run the compute for the Opponent's Move
            # opponentPlayer = ChessBoardModel.opponent(player)
            # worker = Worker(
            #     self.takeOpponentTurn, player=opponentPlayer
            # ) 

            # Execute
            # self.threadpool.start(worker)

    # def takeOpponentTurn(self, player):
    #    # Opponent Takes Turn
    #    opponentCmd = self.chessBoardModel.computeBestValue(player)

    #    if opponentCmd != None:
    #        self.chessBoardModel.movePiece(opponentCmd)
    #        # Communicate the command to FrontEnd
    #        self.communicatorProxy.signal_update_request(opponentCmd)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    @Slot()
    def run(self):
        self.fn(*self.args, **self.kwargs)
