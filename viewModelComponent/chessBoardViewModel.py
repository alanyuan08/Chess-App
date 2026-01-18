# Communication Proxy
from communicatorProxy import CommunicatorProxy

# Models
from modelComponent.moveCommand import MoveCommand
from modelComponent.chessBoardModel import ChessBoardModel

# QTCore
from PySide6.QtCore import QRunnable, QThreadPool, Slot

# Enum
from appEnums import PieceType, Player, MoveCommandType
import random

# Controller 
class ChessBoardViewModel():
    def __init__(self, chessBoardView, chessBoardModel: ChessBoardModel):
        self.communicatorProxy = CommunicatorProxy()

        # Backend ChessBoard Game
        self.chessGameModel = chessBoardModel
        self.communicatorProxy.update_request.connect(chessBoardView.updatePosition)
        self.communicatorProxy.game_state.connect(chessBoardView.updateGameState)
        chessBoardView.connectViewModel(self)

        # Compute Opponent Turn ThreadPool 
        self.threadpool = QThreadPool()

        # Init Game State
        self.communicatorProxy.signal_game_state(
            self.chessGameModel.gameState, 
            self.chessGameModel.gamePlayerTurn
        )

        # White Moves First
        if self.computerTurn():
            # Run the compute for the Opponent's Move
            self.threadpool.start(Worker(
                self.stopGapWhiteOpening
            ))

    # This is a stop-gap to introduce variance if white is going first
    def stopGapWhiteOpening(self):
        kingPawn = MoveCommand(1, 4, 3, 4, MoveCommandType.PAWNOPENMOVE)
        queenPawn = MoveCommand(1, 3, 3, 3, MoveCommandType.PAWNOPENMOVE)

        kingKnight = MoveCommand(0, 1, 2, 2, MoveCommandType.MOVE)
        queenKnight = MoveCommand(0, 6, 2, 5, MoveCommandType.MOVE)

        returnArray = [kingPawn, queenPawn, kingKnight, queenKnight]
        result = random.choice([0, 1, 2, 3])

        cmd = returnArray[result]

        self.chessGameModel.movePiece(cmd)
        # Communicate the command to FrontEnd
        self.communicatorProxy.signal_update_request(cmd)

        # Update Game State
        self.communicatorProxy.signal_game_state(
            self.chessGameModel.gameState, 
            self.chessGameModel.gamePlayerTurn
        )

    def computerTurn(self):
        gameModel = self.chessGameModel
        if gameModel.gamePlayerTurn not in gameModel.humanPlayers:
            return True
        else:
            return False

    def on_move_executed(self, initRow: int, initCol: int, targetRow: int, 
            targetCol: int, player: Player):

        moveCommand = self.chessGameModel.validateMove(initRow, 
            initCol, targetRow, targetCol, player)

        if moveCommand != None:
            # Move for the Chess Model
            self.chessGameModel.movePiece(moveCommand)
            # Communicate the command to FrontEnd
            self.communicatorProxy.signal_update_request(moveCommand)

            # Update Game State
            self.communicatorProxy.signal_game_state(
                self.chessGameModel.gameState, 
                self.chessGameModel.gamePlayerTurn
            )

            # Run the compute for the Opponent's Move
            if self.computerTurn():
                self.threadpool.start(Worker(
                    self.takeOpponentTurn
                ))

    def takeOpponentTurn(self):
        # Opponent Takes Turn
        opponentCmd = self.chessGameModel.computeBestMove()

        self.chessGameModel.movePiece(opponentCmd)
        # Communicate the command to FrontEnd
        self.communicatorProxy.signal_update_request(opponentCmd)

        # Update Game State
        self.communicatorProxy.signal_game_state(
            self.chessGameModel.gameState, 
            self.chessGameModel.gamePlayerTurn
        )


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    @Slot()
    def run(self):
        self.fn(*self.args, **self.kwargs)
