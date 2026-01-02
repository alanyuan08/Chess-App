# Communication Proxy
from communicatorProxy import CommunicatorProxy

# Models
from modelComponent.moveCommand import MoveCommand
from modelComponent.chessBoardModel import ChessBoardModel

# QTCore
from PySide6.QtCore import QRunnable, QThreadPool, Slot

# Enum
from appEnums import PieceType, Player, MoveCommandType

# Controller 
class ChessBoardViewModel():
    def __init__(self, chessBoardView, chessBoardModel: ChessBoardModel):
        self.communicatorProxy = CommunicatorProxy()

        # Backend ChessBoard Game
        self.chessGameModel = chessBoardModel
        self.communicatorProxy.update_request.connect(chessBoardView.updatePosition)
        chessBoardView.connectViewModel(self)

        # Compute Opponent Turn ThreadPool 
        self.threadpool = QThreadPool()

    def on_move_executed(self, initRow: int, initCol: int, targetRow: int, 
            targetCol: int, player: Player):

        moveCommand = self.chessGameModel.chessBoard.validateMove(initRow, 
            initCol, targetRow, targetCol, player)

        if moveCommand != None:
            # Move for the Chess Model
            self.chessGameModel.chessBoard.movePiece(moveCommand)
            # Communicate the command to FrontEnd
            self.communicatorProxy.signal_update_request(moveCommand)

            # Run the compute for the Opponent's Move
            worker = Worker(
                self.takeOpponentTurn
            )

            # Execute
            self.threadpool.start(worker)

    def takeOpponentTurn(self):
        # Opponent Takes Turn
        opponentCmd = self.chessGameModel.computeBestMove()

        if opponentCmd != None:
            self.chessGameModel.chessBoard.movePiece(opponentCmd)
            # Communicate the command to FrontEnd
            self.communicatorProxy.signal_update_request(opponentCmd)
        else:
            print("White Wins")

class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    @Slot()
    def run(self):
        self.fn(*self.args, **self.kwargs)
