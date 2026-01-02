# Import Factory
from modelFactory.chessBoardFactory import ChessBoardFactory

# Model
from modelComponent.chessBoardModel import ChessBoardModel

# Enum
from appEnums import PieceType, Player, MoveCommandType

import math

# Controller 
class ChessGameModel():
    def __init__(self, playersArray):
        self.chessBoard = ChessBoardFactory.createChessBoard(playersArray)

        # Stores the Transposition Table
        self.transpositionTable = {}

    # Take Opponent Turn
    def computeBestMove(self):
        commandList = self.chessBoard.allValidMoves()

        alpha = float('-inf')
        beta = float('inf')

        bestScore = float('-inf')
        returnCmd = None

        for cmd in commandList:
            removedPiece = self.chessBoard.movePiece(cmd)
            returnValue = (-1) * self.negamax(2 , (-1) * beta, (-1) * alpha)
            print(cmd)
            if returnValue > bestScore:
                bestScore = returnValue
                returnCmd = cmd

            alpha = max(alpha, returnValue)
            self.chessBoard.undoMove(cmd, removedPiece)

        return returnCmd

    # Compute Board Value - White is Positive/ Black is Negative
    def computeBoardValue(self):
        returnValue = 0

        phaseWeight = self._calculateGamePhase()
        # Compute for Piece
        for row in range(0, 8):
            for col in range(0, 8):
                if self.chessBoard.board[row][col] != None:
                    if self.chessBoard.board[row][col].player == Player.WHITE:
                        returnValue += self.chessBoard.board[row][col].computedValue(
                            self.chessBoard, phaseWeight)
                    else:
                        returnValue -= self.chessBoard.board[row][col].computedValue(
                            self.chessBoard, phaseWeight)

        # Compute for Double/ Isolated Pawns
        returnValue += self._pawnPenalizer(Player.WHITE, phaseWeight)
        returnValue -= self._pawnPenalizer(Player.BLACK, phaseWeight)

        return returnValue

    # MinMaxSearch -> General
    def negamax(self, depth, alpha, beta):
        validMoves = self.chessBoard.allValidMoves()
        
        # No Valid Moves = Lose
        if len(validMoves) == 0:
            opponent = ChessBoardModel.opponent(self.playerTurn)
            opponentAttackTargets = self._allPlayerCaptureTargets(opponent)
            if self.playerTurn == Player.WHITE:
                if (self.chessBoard.whiteKingSquareRow, self.chessBoard.whiteKingSquareCol) \
                    in opponentAttackTargets:
                    return float('-inf')
                else:
                    # Check Stalemate
                    return 0
            else:
                if (self.chessBoard.blackKingSquareRow, self.chessBoard.blackKingSquareCol) \
                    in opponentAttackTargets:
                    return float('-inf')
                else:
                    # Check Stalemate
                    return 0

        # Termination Condition
        if depth == 0:
            return self.quiesceneSearch(alpha, beta)
        else:
            maxEval = float('-inf')

            for cmd in validMoves:
                removedPiece = self.chessBoard.movePiece(cmd)
                score = (-1) * self.negamax(depth - 1, (-1) * beta, (-1) * alpha)
                self.chessBoard.undoMove(cmd, removedPiece)

                maxEval = max(maxEval, score)
                alpha = max(alpha, score)
                
                if alpha >= beta:
                    break
                    
            return maxEval

    # MinMaxSearch -> General
    def quiesceneSearch(self, alpha, beta, depth = 0):
        staticEval = self.computeBoardValue()

        if depth >= 10:
            return staticEval

        if staticEval >= beta:
            return beta

        if staticEval > alpha:
            alpha = staticEval

        validMoves = self.chessBoard.allValidMoves()

        # No Valid Moves = Lose
        if len(validMoves) == 0:
            opponent = ChessBoardModel.opponent(self.playerTurn)
            opponentAttackTargets = self.chessBoard._allPlayerCaptureTargets(opponent)
            if self.playerTurn == Player.WHITE:
                if (self.chessBoard.whiteKingSquareRow, self.chessBoard.whiteKingSquareCol) \
                    in opponentAttackTargets:
                    return float('-inf')
                else:
                    # Check Stalemate
                    return 0
            else:
                if (self.chessBoard.blackKingSquareRow, self.chessBoard.blackKingSquareCol) \
                    in opponentAttackTargets:
                    return float('-inf')
                else:
                    # Check Stalemate
                    return 0

        for cmd in self._allQuiesceneMoves(validMoves):
            removedPiece = self.chessBoard.movePiece(cmd)
            score = (-1) * self.quiesceneSearch((-1) * beta, (-1) * alpha, depth + 1)
            self.chessBoard.undoMove(cmd, removedPiece)

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    # Return all Capture Moves
    def _allQuiesceneMoves(self, validMoves):
        # QuiescenceMoves
        quiescenceMoveCmd = [MoveCommandType.PROMOTE, MoveCommandType.CAPTURE, \
            MoveCommandType.ENPASSANT]

        return list(filter(
            lambda cmd: cmd.moveType in quiescenceMoveCmd, validMoves
        ))

    # Compute Pawn Penalizer for Isolated / Double Pawn
    def _pawnPenalizer(self, player, phaseWeight):
        filePawnCount = [0 for _ in range(8)]

        for row in range(0, 8):
            for col in range(0, 8):
                piece = self.chessBoard.board[row][col]
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
                    earlyGame  = 15
                    endGame = 25
                    computedPhase = earlyGame * phaseWeight + endGame * (24 - phaseWeight) 
                
                    penalizer += math.ceil(computedPhase / 24)

        return penalizer

    # Maximum Value is 24, used for early/ mid board evaluation
    def _calculateGamePhase(self):
        totalPhaseWeight = 0
        
        for row in range(0, 8):
            for col in range(0, 8):
                if self.chessBoard.board[row][col] != None:
                    totalPhaseWeight += self.chessBoard.board[row][col].phaseWeight()

        return totalPhaseWeight
