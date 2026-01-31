# Import Factory
from modelFactory.chessBoardFactory import ChessBoardFactory

# Model
from modelComponent.chessBoardModel import ChessBoardModel
from modelComponent.moveCommand import MoveCommand
from modelComponent.openingMoveProtocal import OpeningMoveNodeProtocal
from modelComponent.chessBoardZobrist import ChessBoardZobrist
from modelComponent.transpositionTable import TranspositionTable

# Factory
from modelFactory.chessPieceFactory import ChessPieceFactory

# Enum
from appEnums import PieceType, Player, MoveCommandType, GameState, TTBoundType

# Multi Process
import multiprocessing

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

    # Shared Transposition Table for NegaMax Processing
    ttTable = None

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

        bestScore = float('-inf')
        bestMove = None

        # Init Shared Transposition Table
        sharedttTable = TranspositionTable()

        with multiprocessing.Pool(initializer=ChessGameModel.init_worker, 
            initargs=(sharedttTable,)) as pool:
            for move, score in pool.imap_unordered(self._negamaxWorker, commandList):
                if score > bestScore:
                    bestScore = score
                    bestMove = move
        
        return bestMove

    # -----

    @staticmethod
    def init_worker(sharedttTable: TranspositionTable):
        global ttTable
        ttTable = sharedttTable

    # This worker runs in a separate process
    def _negamaxWorker(self, cmd: MoveCommand, depth: int = 5, 
        currAlpha: int = float('-inf'), currBeta: int = float('inf')) -> (MoveCommand, int):
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

        # Retrieve ttEntry Cache
        ttEntry = ttTable.probe(self.chessBoard.zobristHash)
        if ttEntry and ttEntry.depth >= depth:
            if ttEntry.flag == TTBoundType.EXACT:
                return ttEntry.score

            elif ttEntry.flag == TTBoundType.UPPERBOUND and ttEntry.score <= alpha:
                return ttEntry.score

            elif ttEntry.flag == TTBoundType.LOWERBOUND and ttEntry.score >= beta:
                return ttEntry.score

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

            # Store Value
            ttValue = maxEval
            flag = None
            if ttValue <= originalAlpha:
                flag = TTBoundType.UPPERBOUND

            elif ttValue >= beta:
                flag = TTBoundType.LOWERBOUND

            else:
                flag = TTBoundType.EXACT
            ttTable.store(self.chessBoard.zobristHash, maxEval, depth, flag)

            return maxEval

