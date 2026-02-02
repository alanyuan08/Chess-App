# Import Factory
from modelFactory.chessBoardFactory import ChessBoardFactory

# Model
from modelComponent.chessBoardModel import ChessBoardModel
from modelComponent.moveCommand import MoveCommand
from modelComponent.openingMoveProtocal import OpeningMoveNodeProtocal
from modelComponent.chessBoardZobrist import ChessBoardZobrist

# Factory
from modelFactory.chessPieceFactory import ChessPieceFactory

# Enum
from appEnums import PieceType, Player, MoveCommandType, GameState, TTBoundType

# Multi Process
import multiprocessing
import concurrent.futures

# Controller 
class ChessGameModel():
    def __init__(self, humanPlayers: list[Player], chessBoard: list[list[ChessBoardModel]], 
        openingHandBook: OpeningMoveNodeProtocal):
        self.chessBoard = chessBoard

        # Set Human Player
        self.humanPlayers = humanPlayers

        # Game Turn - Chess Board Turn may be different due to backtracking
        self.gamePlayerTurn = Player.WHITE

        # Set Player Lose
        self.gameState = GameState.PLAYING

        # Opening Handbook - Node Represents Current Move
        self.currOpeningMove = openingHandBook

    # Move Piece
    def movePiece(self, cmd: MoveCommand):
        # Move the Chess Piece
        self.chessBoard.movePiece(cmd)

        if self.currOpeningMove:
            self.currOpeningMove = self.currOpeningMove.stepForward(cmd)

        self.gamePlayerTurn = ChessBoardModel.opponent(self.gamePlayerTurn)

        # Three Move Repetition Draw
        if self.chessBoard.checkThreeMoveRepetiton():
            self.gameState = GameState.DRAW

        # No Moves - Determine Checkmate or Draw
        if len(self.chessBoard.allValidMoves()) == 0:
            if self.gamePlayerTurn == Player.WHITE:
                if self.chessBoard.checkMate():
                    self.gameState = GameState.BLACKWIN
                else:
                    self.gameState = self.gameState.DRAW
            else:
                if self.chessBoard.checkMate():
                    self.gameState = GameState.WHITEWIN
                else:
                    self.gameState = self.gameState.DRAW
    # Validate Move
    def validateMove(self, initRow: int, initCol: int, targetRow: int, 
        targetCol: int, player: Player) -> MoveCommand:
        # It's not your turn to move
        if player != self.gamePlayerTurn:
            return None

        return self.chessBoard.validateMove(initRow, initCol, targetRow, targetCol, player)

    # Take Opponent Turn
    def computeBestMove(self) -> MoveCommand:
        if self.currOpeningMove and self.currOpeningMove.hasSubsequentCmd():
            return self.currOpeningMove.randomSubsequentCmd()

        commandList = self.chessBoard.allValidMoves()
        commandList.sort(key=lambda move: self.chessBoard._getMovePriority(move), reverse=True)

        alpha = float('-inf')
        beta = float('inf')

        bestScore = float('-inf')
        bestMove = None

        if len(commandList) == 0:
            return None

        # Younger Brother Best Move
        initMove = commandList[0]
        for depth in range(1, 6):

            # Compute the most optimal search move            
            if bestMove != None:
                print(depth, bestMove)
                initMove = bestMove

            removedPiece, prevEnPassant = self.chessBoard.movePiece(initMove)
            # Search the first move normally to get a strong alpha value quickly
            score = (-1) * self._negamax(depth, (-1) * beta, (-1) * alpha) 
            self.chessBoard.undoMove(initMove, removedPiece, prevEnPassant)

            # Older Brother 
            bestScore = score
            bestMove = initMove
            alpha = score # Establish the strong alpha

            # Younger Brother Parallel Search
            remaining_moves = [item for item in commandList if item != initMove]
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count() - 1) as executor:
                futures = [
                    executor.submit(self._negamaxWorker, cmd, alpha, beta, depth) 
                    for cmd in remaining_moves
                ]
                
                for future in concurrent.futures.as_completed(futures):
                    move, score = future.result()
                    if score > bestScore:
                        bestScore = score
                        bestMove = move

        return bestMove

    # -----

    # This worker runs in a separate process
    def _negamaxWorker(self, cmd: MoveCommand, currAlpha: int, currBeta: int, 
            depth: int) -> (MoveCommand, int):
        removedPiece, prevEnPassant = self.chessBoard.movePiece(cmd)
        
        # Negamx search for Best Position
        score = (-1) * self._negamax(depth - 1, (-1) * currBeta, (-1) * currAlpha)
                
        return cmd, score

    # MinMaxSearch -> General
    def _negamax(self, depth: int, alpha: int, beta: int, ply: int = 0) -> int:
        validMoves = self.chessBoard.allValidMoves()
        validMoves.sort(key=lambda move: self.chessBoard._getMovePriority(move), reverse=True)

        # Store Original Alpha 
        originalAlpha = alpha

        # Three Move Repetition Draw
        if self.chessBoard.checkThreeMoveRepetiton():
            return 0

        # No Valid Moves = Lose / Draw
        if len(validMoves) == 0:
            return self.chessBoard.resolveEndGame(ply)

        # Termination Condition
        if depth == 0:
            return self.chessBoard._quiesceneSearch(alpha, beta)
        else:
            maxEval = float('-inf')

            for cmd in validMoves:
                prevzobristHash = self.chessBoard.zobristHash

                removedPiece, prevEnPassant = self.chessBoard.movePiece(cmd)
                score = (-1) * self._negamax(depth - 1, (-1) * beta, (-1) * alpha, ply + 1)
                self.chessBoard.undoMove(cmd, removedPiece, prevEnPassant)

                maxEval = max(maxEval, score)
                alpha = max(alpha, score)
                
                if alpha >= beta:
                    break

            return maxEval