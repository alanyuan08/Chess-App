# Import Factory
from modelFactory.chessBoardFactory import ChessBoardFactory

# Model
from modelComponent.chessBoardModel import ChessBoardModel
from modelComponent.moveCommand import MoveCommand

# Factory
from modelFactory.chessPieceFactory import ChessPieceFactory

# Enum
from appEnums import PieceType, Player, MoveCommandType

# Normal
import math

# Controller 
class ChessGameModel():
    def __init__(self, playersArray):
        self.chessBoard = ChessBoardFactory.createChessBoard(playersArray)

        # Stores the Transposition Table
        self.transpositionTable = {}

        # Set Human Player
        self.humanPlayers = playersArray

        # Set Game Turn 
        self.gamePlayerTurn = Player.WHITE

    # Move Piece
    def movePiece(self, cmd: MoveCommand):
        self.chessBoard.movePiece(cmd)

        self.gamePlayerTurn = ChessBoardModel.opponent(self.gamePlayerTurn)

    # Validate Move
    def validateMove(self, initRow: int, initCol: int, targetRow: int, 
        targetCol: int, player: Player) -> MoveCommand:
        # It's not your turn to move
        if player != self.gamePlayerTurn:
            return None

        return self.chessBoard.validateMove(initRow, initCol, targetRow, targetCol, player)

    # Take Opponent Turn
    def computeBestMove(self) -> MoveCommand:
        commandList = self.chessBoard.allValidMoves()
        commandList.sort(key=lambda move: self._getMovePriority(move), reverse=True)

        alpha = float('-inf')
        beta = float('inf')

        bestScore = float('-inf')
        bestMove = None

        for cmd in commandList:
            removedPiece = self.chessBoard.movePiece(cmd)
            score = (-1) * self._negamax(3, (-1) * beta, (-1) * alpha)
            self.chessBoard.undoMove(cmd, removedPiece)

            print(cmd)
            if score > bestScore:
                bestScore = score
                bestMove = cmd

            alpha = max(alpha, score)

        return bestMove

    # Compute Board Value - White is Positive/ Black is Negative
    def _computeBoardValue(self) -> int:
        returnValue = 0

        phaseWeight = self._calculateGamePhase()
        board = self.chessBoard.board
        # Compute for Piece
        for row in range(0, 8):
            for col in range(0, 8):
                gamePiece = board[row][col]
                if gamePiece != None:
                    if gamePiece.player == Player.WHITE:
                        returnValue += gamePiece.computedValue(
                            self.chessBoard, phaseWeight)
                    else:
                        returnValue -= gamePiece.computedValue(
                            self.chessBoard, phaseWeight)

        # Compute for Double/ Isolated Pawns
        returnValue -= self._pawnPenalizer(Player.WHITE, phaseWeight)
        returnValue += self._pawnPenalizer(Player.BLACK, phaseWeight)

        if self.chessBoard.playerTurn == Player.WHITE:
            return returnValue
        else:
            return (-1) * returnValue

    # MinMaxSearch -> General
    def _negamax(self, depth, alpha, beta) -> int:
        validMoves = self.chessBoard.allValidMoves()
        validMoves.sort(key=lambda move: self._getMovePriority(move), reverse=True)
        
        # No Valid Moves = Lose
        if len(validMoves) == 0:
            return self.resolveEndGame()

        # Termination Condition
        if depth == 0:
            return self._quiesceneSearch(alpha, beta)
        else:
            maxEval = float('-inf')

            for cmd in validMoves:
                removedPiece = self.chessBoard.movePiece(cmd)
                score = (-1) * self._negamax(depth - 1, (-1) * beta, (-1) * alpha)
                self.chessBoard.undoMove(cmd, removedPiece)

                maxEval = max(maxEval, score)
                alpha = max(alpha, score)
                
                if alpha >= beta:
                    break
                    
            return maxEval

    # MinMaxSearch -> General
    def _quiesceneSearch(self, alpha, beta, depth = 0) -> int:
        staticEval = self._computeBoardValue()

        if depth >= 10:
            return staticEval

        if staticEval >= beta:
            return beta

        if staticEval > alpha:
            alpha = staticEval

        validMoves = self.chessBoard.allValidMoves()
        validMoves.sort(key=lambda move: self._getMovePriority(move), reverse=True)

        # No Valid Moves = Lose
        if len(validMoves) == 0:
            return self.resolveEndGame()

        for cmd in self._allQuiesceneMoves(validMoves):
            removedPiece = self.chessBoard.movePiece(cmd)
            score = (-1) * self._quiesceneSearch((-1) * beta, (-1) * alpha, depth + 1)
            self.chessBoard.undoMove(cmd, removedPiece)

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    # Resolve End Game -> Called when No Valid Moves
    def resolveEndGame(self) -> int:
        opponentAttackTargets = self.chessBoard.allOpponentCaptureTargets()

        kingTuple = self.chessBoard.kingTuple
        if kingTuple in opponentAttackTargets:
            if self.playerTurn == Player.WHITE:
                return float('-inf')
            else:
                return float('inf')
        else:
            return 0

    # Return all Capture Moves
    def _allQuiesceneMoves(self, validMoves) -> list[MoveCommand]:
        # QuiescenceMoves
        quiescenceMoveCmd = [MoveCommandType.PROMOTE, MoveCommandType.CAPTURE, \
            MoveCommandType.ENPASSANT]

        return list(filter(
            lambda cmd: cmd.moveType in quiescenceMoveCmd, validMoves
        ))

    # Compute Pawn Penalizer for Isolated / Double Pawn
    def _pawnPenalizer(self, player, phaseWeight) -> int:
        filePawnCount = [0 for _ in range(8)]

        board = self.chessBoard.board
        for row in range(0, 8):
            for col in range(0, 8):
                piece = board[row][col]
                if piece != None and piece.type == PieceType.PAWN and piece.player == player:
                    filePawnCount[col] += 1

        penalizer = 0
        # Pawn Penalizer
        for file in range(0, 8):
            pawnCount = filePawnCount[file]

            if pawnCount > 0:
                # Penalizer for Double / Triple/ Quad Pawns
                if pawnCount > 1:
                    penalizer += (pawnCount - 1) * 15

                # Compute for Isolated Pawns
                isIsolated = True
                if file > 0 and filePawnCount[file-1] > 0:
                    isIsolated = False
                if file < 7 and filePawnCount[file+1] > 0:
                    isIsolated = False
                    
                if isIsolated:
                    earlyGame = 15
                    endGame = 25
                    computedPhase = earlyGame * phaseWeight + endGame * (24 - phaseWeight) 
                
                    penalizer += math.ceil(computedPhase / 24)

        return penalizer

    # Maximum Value is 24, used for early/ mid board evaluation
    def _calculateGamePhase(self) -> int:
        totalPhaseWeight = 0

        board = self.chessBoard.board
        for row in range(0, 8):
            for col in range(0, 8):
                if board[row][col] != None:
                    totalPhaseWeight += board[row][col].phaseWeight()

        return totalPhaseWeight


    # Compute Move Priority
    def _getMovePriority(self, cmd: MoveCommand) -> int:
        board = self.chessBoard.board

        # 1. Promotions (High Priority)
        if cmd.moveType == MoveCommandType.PROMOTE:
            return 90000 # Treat as Queen value
        
        # 2. Captures (MVV-LVA)
        if cmd.moveType in [MoveCommandType.CAPTURE, MoveCommandType.ENPASSANT]:
            capturedPieceVal = 0
            if cmd.moveType == MoveCommandType.ENPASSANT:
                capturedPieceVal = board[cmd.startRow][cmd.endCol].pieceValue()
            else:
                capturedPieceVal = board[cmd.endRow][cmd.endCol].pieceValue()

            startingPieceVal = board[cmd.startRow][cmd.startCol].pieceValue()  
            
            return (capturedPieceVal * 10) - startingPieceVal

        # 3. Castling (Mid Priority)
        if cmd.moveType in [MoveCommandType.KINGSIDECASTLE, MoveCommandType.QUEENSIDECASTLE]:
            return 50

        # 4. Standard Moves (Low Priority)
        return 0

