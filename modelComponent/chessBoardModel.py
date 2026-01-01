# Enum 
from appEnums import PieceType, Player, MoveCommandType

# Factory
from modelFactory.chessPieceFactory import ChessPieceFactory

# Model
from modelComponent.moveCommand import MoveCommand

# Multi Process Pool
import multiprocessing
import copy
import math

# Controller 
class ChessBoardModel():
    def __init__(self):
        # Create the Chess Board
        self.board = []

        # Init Player as White
        self.playerTurn = Player.WHITE

        # Set Human Player
        self.humanPlayers = []

        # En Passant Column - Set after pawn move, then cleared 
        self.whiteEnPassantColumn = None
        self.blackEnPassantColumn = None

        # Used to Keep Score for Castle
        self.whiteCastled = False
        self.blackCastled = False

        # Used to Check for King Safety
        self.whiteKingSquareRow = 0
        self.whiteKingSquareCol = 4

        self.blackKingSquareRow = 7
        self.blackKingSquareCol = 4
        
    # Danger Moves
    quiescenceMoveCmd = [MoveCommandType.PROMOTE, MoveCommandType.CAPTURE, MoveCommandType.ENPASSANT]

    @staticmethod
    def opponent(player: Player):
        if player == Player.WHITE:
            return Player.BLACK
        else:
            return Player.WHITE

    # Validate the Move
    def validateMove(self, initRow: int, initCol: int, targetRow: int, targetCol: int, player: Player):
        # It's not your turn to move
        if player != self.playerTurn:
            return None

        # Validate the Move Command is a Possible Move
        targetPiece = self.board[initRow][initCol]
        if targetPiece == None:
            return None

        if targetPiece.player == player:
            possibleMoves = targetPiece.possibleMoves(self)
            for cmd in possibleMoves:
                if cmd.endRow == targetRow and cmd.endCol == targetCol:
                    return cmd

        return None

    # Validate King Safety for player after making move
    def validateKingSafety(self, cmd: MoveCommand):
        # This is explictly defined here to avoid confusion after the move
        currentPlayer = self.playerTurn
        removedPiece = self.movePiece(cmd)

        returnValue = False
        if currentPlayer == Player.BLACK:
            kingPiece = self.board[self.blackKingSquareRow][self.blackKingSquareCol]
            returnValue = kingPiece.evaluateKingSafety(self)

        elif currentPlayer == Player.WHITE:
            kingPiece = self.board[self.whiteKingSquareRow][self.whiteKingSquareCol]
            returnValue = kingPiece.evaluateKingSafety(self)

        self.undoMove(cmd, removedPiece)
        return returnValue

    # Compute Move Priority
    def getMovePriority(self, cmd, board):
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

    # Return all Valid moves for currentPlayer
    def allValidMoves(self):
        validMoves = []

        for row in range(0, 8):
            for col in range(0, 8):
                if self.board[row][col] != None:
                    if self.board[row][col].player == self.playerTurn:
                        for cmd in self.board[row][col].possibleMoves(self):
                            validMoves.append(cmd)

        validMoves.sort(key=lambda m: self.getMovePriority(m, self.board), reverse=True)

        return validMoves


    # Maximum Value is 24, used for early/ mid board evaluation
    def calculateGamePhase(self):
        totalPhaseWeight = 0
        
        for row in range(0, 8):
            for col in range(0, 8):
                if self.board[row][col] != None:
                    totalPhaseWeight += self.board[row][col].phaseWeight()

        return totalPhaseWeight

    # Compute Pawn Penalizer for Isolated / Double Pawn
    def pawnPenalizer(self, player, phaseWeight):
        filePawnCount = [0 for _ in range(8)]

        for row in range(0, 8):
            for col in range(0, 8):
                piece = self.board[row][col]
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

    # Compute Board Value - Assume White is the Protagonist
    def computeBoardValue(self):
        returnValue = 0

        phaseWeight = self.calculateGamePhase()
        # Compute for Piece
        for row in range(0, 8):
            for col in range(0, 8):
                if self.board[row][col] != None:
                    if self.board[row][col].player == Player.WHITE:
                        returnValue += self.board[row][col].computedValue(self, phaseWeight)
                    else:
                        returnValue -= self.board[row][col].computedValue(self, phaseWeight)

        # Compute for Double/ Isolated Pawns
        returnValue += self.pawnPenalizer(Player.WHITE, phaseWeight)
        returnValue -= self.pawnPenalizer(Player.BLACK, phaseWeight)

        return returnValue

    # Return all Capture Moves
    def allQuiesceneMoves(self, validMoves):
        return list(filter(
            lambda cmd: cmd.moveType in self.quiescenceMoveCmd, validMoves
        ))

    # MinMaxSearch -> General
    def quiesceneSearch(self, alpha, beta, depth = 0):
        staticEval = self.computeBoardValue()

        if depth >= 10:
            return staticEval

        if staticEval >= beta:
            return beta

        if staticEval > alpha:
            alpha = staticEval

        validMoves = self.allValidMoves()
        for cmd in self.allQuiesceneMoves(validMoves):
            removedPiece = self.movePiece(cmd)
            score = (-1) * self.quiesceneSearch((-1) * beta, (-1) * alpha, depth + 1)
            self.undoMove(cmd, removedPiece)

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    # MinMaxSearch -> General
    def negamax(self, depth, alpha, beta):
        validMoves = self.allValidMoves()
        
        # No Valid Moves = Lose
        if len(validMoves) == 0:
            opponent = ChessBoardModel.opponent(self.playerTurn)
            opponentAttackTargets = self._allPlayerCaptureTargets(opponent)
            if self.playerTurn == Player.WHITE:
                if (self.whiteKingSquareRow, self.whiteKingSquareCol) in opponentAttackTargets:
                    return float('-inf')
                else:
                    # Check Stalemate
                    return 0
            else:
                if (self.blackKingSquareRow, self.blackKingSquareCol) in opponentAttackTargets:
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
                removedPiece = self.movePiece(cmd)
                score = (-1) * self.negamax(depth - 1, (-1) * beta, (-1) * alpha)
                self.undoMove(cmd, removedPiece)

                maxEval = max(maxEval, score)
                alpha = max(alpha, score)
                
                if alpha >= beta:
                    break
                    
            return maxEval

    # Iterative Deepening
    def computeMoveWrapperDepth2(self, cmd):
        return self._defaultMoveWrapper(cmd, 2)

    def _defaultMoveWrapper(self, cmd, depth):
        newBoard = copy.deepcopy(self)
        newBoard.movePiece(cmd)

        alpha = float('-inf')
        beta = float('inf')
        score = (-1) * newBoard.negamax(depth, (-1) * beta, (-1) * alpha)
        return score, cmd

    # Take Opponent Turn
    def computeBestMove(self):
        commandList = self.allValidMoves()
        with multiprocessing.Pool() as pool:
            scores_and_moves = pool.map(self.computeMoveWrapperDepth2, commandList)

        scores_and_moves.sort(key=lambda x: x[0], reverse=True)

        alpha = float('-inf')
        beta = float('inf')
        bestScore = float('-inf')

        for score, cmd in scores_and_moves:
            removedPiece = self.movePiece(cmd)
            returnValue = (-1) * self.negamax(4 , (-1) * beta, (-1) * alpha)
            print(cmd)
            if returnValue > bestScore:
                bestScore = returnValue
                returnCmd = cmd

            alpha = max(alpha, returnValue)
            self.undoMove(cmd, removedPiece)

        return returnCmd

    # This is used to determine king safety and castle
    def _allPlayerCaptureTargets(self, player):
        captureSquares = {}

        for row in range(0, 8):
            for col in range(0, 8):
                targetPiece = self.board[row][col]
                if targetPiece != None and targetPiece.player == player:

                    captureTargets = targetPiece.captureTargets(self)
                    for target in captureTargets:
                        captureSquares[target] = True
        
        return captureSquares

    def _updateKingSquare(self, kingRow: int, kingCol: int):
        # Update the King Square
        if self.board[kingRow][kingCol].type == PieceType.KING:
            if self.board[kingRow][kingCol].player == Player.BLACK:
                self.blackKingSquareRow = kingRow
                self.blackKingSquareCol = kingCol

            elif self.board[kingRow][kingCol].player == Player.WHITE:
                self.whiteKingSquareRow = kingRow
                self.whiteKingSquareCol = kingCol

    # Move Piece on ChessBoard
    def _movePieceOnBoard(self, startRow: int, startCol: int, endRow: int, endCol: int):
        self.board[endRow][endCol] = self.board[startRow][startCol]
        self.board[endRow][endCol].row = endRow
        self.board[endRow][endCol].col = endCol
        self.board[endRow][endCol].moves += 1

        # Remove the Init Piece
        self.board[startRow][startCol] = None

        # Update the King Square
        self._updateKingSquare(endRow, endCol)

    def _undoMoveOnBoard(self, originalRow: int, originalCol: int, currentRow: int, currentCol: int):        
        self.board[originalRow][originalCol] = self.board[currentRow][currentCol] 
        self.board[originalRow][originalCol].row = originalRow
        self.board[originalRow][originalCol].col = originalCol

        # Undo Castle will take in account of both Rook and King
        self.board[originalRow][originalCol].moves -= 1

        # Remove the Final Piece
        self.board[currentRow][currentCol] = None

        # Update the King Square
        self._updateKingSquare(originalRow, originalCol)

    def undoMove(self, cmd: MoveCommand, restorePiece):
        # Swap the Player Turn
        self.playerTurn = ChessBoardModel.opponent(self.playerTurn)

        match cmd.moveType:
            # Queen Side Castle (Undo)
            case MoveCommandType.QUEENSIDECASTLE:
                if self.playerTurn == Player.BLACK:
                    # Move the King 2 steps to the left
                    self._undoMoveOnBoard(7, 4, 7, 2)

                    # Move the Rook to the right of the king
                    self._undoMoveOnBoard(7, 0, 7, 3)

                    # Update Castle
                    self.blackCastled = False

                elif self.playerTurn == Player.WHITE:
                    # Move the King 2 steps to the left
                    self._undoMoveOnBoard(0, 4, 0, 2)

                    # Move the Rook to the right of the king
                    self._undoMoveOnBoard(0, 0, 0, 3)

                    # Update Castle
                    self.whiteCastled = False

            # King Side Castle
            case MoveCommandType.KINGSIDECASTLE:
                if self.playerTurn == Player.BLACK:
                    # Move the King 2 steps to the right
                    self._undoMoveOnBoard(7, 4, 7, 6)

                    # Move the Rook to the right of the king
                    self._undoMoveOnBoard(7, 7, 7, 5)

                    # Update Castle
                    self.blackCastled = False

                elif self.playerTurn == Player.WHITE:
                    # Move the King 2 steps to the right
                    self._undoMoveOnBoard(0, 4, 0, 6)

                    # Move the Rook to the right of the king
                    self._undoMoveOnBoard(0, 7, 0, 5)

                    # Update Castle
                    self.whiteCastled = False

            # Move Piece
            case MoveCommandType.MOVE:
                # Move Starting Piece to Move Point
                self._undoMoveOnBoard(cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

            case MoveCommandType.CAPTURE:
                # Move Starting Piece to Capture Point
                self._undoMoveOnBoard(cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

                # Restore Piece
                self.board[cmd.endRow][cmd.endCol] = restorePiece

            # Double Pawn Move
            case MoveCommandType.PAWNOPENMOVE:
                # Move the piece from start to end
                self._undoMoveOnBoard(cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

            # Promote Pawn
            case MoveCommandType.PROMOTE:
                # Promote the Pawn to a Queen
                self.board[cmd.endRow][cmd.endCol] = restorePiece

                # Store Removed Piece
                self.board[cmd.startRow][cmd.startCol] = ChessPieceFactory.createChessPiece(
                    PieceType.PAWN, self.playerTurn, cmd.startRow, cmd.startCol)

            # En Passant
            case MoveCommandType.ENPASSANT:
                # Move the Pawn to the target
                self._undoMoveOnBoard(cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

                # Store Removed Piece
                self.board[cmd.startRow][cmd.endCol] = restorePiece

                # Unset the En Passant Column
                if self.playerTurn == Player.WHITE:
                    self.whiteEnPassantColumn = None
                else:
                    self.blackEnPassantColumn = None

        return 

    def movePiece(self, cmd: MoveCommand):
        # Set enPassant to Null - Reset this if the opponent does a double pawn move
        self.whiteEnPassantColumn = None
        self.blackEnPassantColumn = None

        # Captured Piece 
        removedPiece = None

        match cmd.moveType:
            # Queen Side Castle
            case MoveCommandType.QUEENSIDECASTLE:
                # Determine the row of the Castle 
                row = 7 if self.playerTurn == Player.BLACK else 0

                # Move the King 2 steps to the left
                self._movePieceOnBoard(row, 4, row, 2)

                # Move the Rook to the right of the king
                self._movePieceOnBoard(row, 0, row, 3)

                # Update Castle
                if self.playerTurn == Player.BLACK:
                    self.blackCastled = True
                else:
                    self.whiteCastled = True

            # King Side Castle
            case MoveCommandType.KINGSIDECASTLE:
                # Determine the row of the Castle 
                row = 7 if self.playerTurn == Player.BLACK else 0

                # Move the King 2 steps to the left
                self._movePieceOnBoard(row, 4, row, 6)

                # Move the Rook to the right of the king
                self._movePieceOnBoard(row, 7, row, 5)

                # Update Castle
                if self.playerTurn == Player.BLACK:
                    self.blackCastled = True
                else:
                    self.whiteCastled = True

            # Move Piece
            case MoveCommandType.MOVE:
                # Move Starting Piece to Capture Point
                self._movePieceOnBoard(cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

            case MoveCommandType.CAPTURE:
                # Store the Removed Piece
                removedPiece = self.board[cmd.endRow][cmd.endCol]
                self.board[cmd.endRow][cmd.endCol] = None

                # Move Starting Piece to Capture Point
                self._movePieceOnBoard(cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

            # Double Pawn Move
            case MoveCommandType.PAWNOPENMOVE:
                # Move the piece from start to end
                self._movePieceOnBoard(cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

                if self.playerTurn == Player.WHITE:
                    self.whiteEnPassantColumn = cmd.endCol
                else:
                    self.blackEnPassantColumn = cmd.endCol

            # Promote Pawn
            case MoveCommandType.PROMOTE:
                # Promote MAY be a capture or a move
                removedPiece = self.board[cmd.endRow][cmd.endCol]

                self.board[cmd.endRow][cmd.endCol] = ChessPieceFactory.createChessPiece(
                    PieceType.QUEEN, self.playerTurn, cmd.endRow, cmd.endCol)

            # En Passant
            case MoveCommandType.ENPASSANT:
                # Store Removed Piece
                removedPiece = self.board[cmd.startRow][cmd.endCol]

                # Move the Pawn to the target
                self._movePieceOnBoard(cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

                # Remove En Passant Pawn
                self.board[cmd.startRow][cmd.endCol] = None

        # Swap the Player Turn
        self.playerTurn = ChessBoardModel.opponent(self.playerTurn)

        # Create a new copy of the removed Piece
        return removedPiece
