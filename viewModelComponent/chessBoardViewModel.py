from appEnums import PieceType, Player, MoveCommandType
from communicatorProxy import CommunicatorProxy

# Models
from modelComponent.moveCommand import MoveCommand
from modelComponent.chessBoardModel import ChessBoardModel

# QTCore
import traceback
import sys
import time
from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Slot, Signal

# Controller 
class ChessBoardViewModel():
    def __init__(self, chessBoardView, chessBoardModel):
        self.communicatorProxy = CommunicatorProxy()

        # Backend ChessBoard Model
        self.chessBoardModel = chessBoardModel
        self.communicatorProxy.update_request.connect(chessBoardView.updatePosition)
        chessBoardView.connectViewModel(self)

        self.threadpool = QThreadPool()

    def on_move_executed(self, initRow: int, initCol: int, targetRow: int, 
            targetCol: int, player: Player):

        moveCommand = self.chessBoardModel.validateAndReturnCommand(initRow, 
            initCol, targetRow, targetCol, player)

        if moveCommand != None:
            # Move for the Chess Model
            self.chessBoardModel.movePiece(moveCommand)
            # Communicate the command to FrontEnd
            self.communicatorProxy.signal_update_request(moveCommand)


            opponentPlayer = ChessBoardModel.returnOpponent(player)
            worker = Worker(
                self.computerTurn, player=opponentPlayer
            )  # Any other args, kwargs are passed to the run function

            # Execute
            self.threadpool.start(worker)

    def computerTurn(self, player):
        # Opponent Takes Turn
        opponentCmd = self.chessBoardModel.computeBestValue(player)

        if opponentCmd != None:
            self.chessBoardModel.movePiece(opponentCmd)
            # Communicate the command to FrontEnd
            self.communicatorProxy.signal_update_request(opponentCmd)

class Worker(QRunnable):
    """Worker thread.

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread.
                     Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function
    """

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    @Slot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception:
            traceback.print_exc()
