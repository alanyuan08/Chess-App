# Communication Proxy
from communicatorProxy import CommunicatorProxy

# Models
from modelComponent.chessBoardModel import ChessBoardModel

# QTCore
from PySide6.QtCore import QRunnable, QThreadPool, Slot

# Enum
from appEnums import Player, GameState

import rust_compute
import traceback

# Controller 
class ChessBoardViewModel():
    def __init__(self, chessBoardView, chessBoardModel: ChessBoardModel):
        self.communicatorProxy = CommunicatorProxy()

        # Backend ChessBoard Game
        self.chessGameModel = chessBoardModel
        self.communicatorProxy.updateRequest.connect(chessBoardView.updatePosition)
        self.communicatorProxy.updateGameState.connect(chessBoardView.updateGameState)
        chessBoardView.connectViewModel(self)

        # Compute Opponent Turn ThreadPool 
        self.threadpool = QThreadPool()

        # Init Game State
        self.communicatorProxy.signalUpdateGameState(
            self.chessGameModel.gameState, 
            self.chessGameModel.gamePlayerTurn
        )

        # White Moves First
        if self.computerTurn():
            # Run the compute for the Opponent's Move
            self.threadpool.start(Worker(
                self.takeOpponentTurn
            ))

    def computerTurn(self):
        gameModel = self.chessGameModel
        if gameModel.gamePlayerTurn not in gameModel.humanPlayers:
            return True
        else:
            return False

    def onMoveExecuted(self, initRow: int, initCol: int, targetRow: int, 
            targetCol: int, player: Player):

        moveCommand = self.chessGameModel.validateMove(initRow, 
            initCol, targetRow, targetCol, player)

        # Game Over / Board is Locked
        if self.chessGameModel.gameState != GameState.PLAYING:
            return 

        if moveCommand != None:
            # Move the Chess Piece
            self.chessGameModel.movePiece(moveCommand)

            # Communicate the command to FrontEnd
            self.communicatorProxy.signalUpdateRequest(moveCommand)

            # Update Game State
            self.communicatorProxy.signalUpdateGameState(
                self.chessGameModel.gameState, 
                self.chessGameModel.gamePlayerTurn
            )

            # Run the compute for the Opponent's Move
            if self.computerTurn():
                self.threadpool.start(Worker(
                    self.takeOpponentTurn
                ))

    def takeOpponentTurn(self):
        try:
            # Rust Integration
            print(rust_compute.compute_next_move(self.chessGameModel.returnChessUCIMoves()))

            # Compute Opponent Move
            opponentCmd = self.chessGameModel.computeBestMove()        

            # Move the Chess Piece
            self.chessGameModel.movePiece(opponentCmd)

            # Communicate the command to FrontEnd
            self.communicatorProxy.signalUpdateRequest(opponentCmd)

            # Update Game State
            self.communicatorProxy.signalUpdateGameState(
                self.chessGameModel.gameState, 
                self.chessGameModel.gamePlayerTurn
            )
        except Exception:
            # This force-prints the full error to your console
            traceback.print_exc() 
class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    @Slot()
    def run(self):
        self.fn(*self.args, **self.kwargs)
