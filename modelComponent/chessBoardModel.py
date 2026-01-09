# Enum 
from appEnums import PieceType, Player, MoveCommandType

# Model 
from modelComponent.moveCommand import MoveCommand
from modelComponent.chessPieceModel import ChessPieceModel
from modelComponent.chessBoardZobrist import ChessBoardZobrist

# Factory
from modelFactory.chessPieceFactory import ChessPieceFactory

# Standard
import math
from typing import Optional, Union
import copy

# Controller 
class ChessBoardModel():
    def __init__(self):
        # Create the Chess Board
        self.board = [[None for _ in range(8)] for _ in range(8)]

        # This is maintained for backtracking
        self.playerTurn = Player.WHITE

        # En Passant Column
        self.enPassant = None

        # Used to Check for King Safety
        self.whiteKingSquareRow = 0
        self.whiteKingSquareCol = 4

        self.blackKingSquareRow = 7
        self.blackKingSquareCol = 4

        # Use for Zobrist Hash
        self.whiteCanQueenSide = True
        self.whiteCanKingSide = True

        self.blackCanQueenSide = True
        self.blackCanKingSide = True

        # Zobrist Hash
        self.zobristHash = ChessBoardZobrist.computeInitValue(self)

    # This worker runs in a separate process
    def _negamax_worker(self, move_to_search, current_alpha, current_beta, depth):
        newBoard = copy.deepcopy(self)
        removedPiece, prevEnPassant, prevCastleIndex = newBoard.movePiece(move_to_search)
        
        # We search with the narrow window established by the PV move
        score = (-1) * newBoard._negamax(depth - 1, (-1) * current_beta, (-1) * current_alpha)
                
        return move_to_search, score

    # Compute Board Value - White is Positive/ Black is Negative
    def _computeBoardValue(self) -> int:
        returnValue = 0

        phaseWeight = self._calculateGamePhase()
        # Compute for Piece
        for row in range(0, 8):
            for col in range(0, 8):
                gamePiece = self.board[row][col]
                if gamePiece != None:
                    if gamePiece.player == Player.WHITE:
                        returnValue += gamePiece.computedValue(
                            self, phaseWeight)
                    else:
                        returnValue -= gamePiece.computedValue(
                            self, phaseWeight)

        # Compute for Double/ Isolated Pawns
        returnValue -= self._pawnPenalizer(Player.WHITE, phaseWeight)
        returnValue += self._pawnPenalizer(Player.BLACK, phaseWeight)

        if self.playerTurn == Player.WHITE:
            return returnValue
        else:
            return (-1) * returnValue

    # MinMaxSearch -> General
    def _negamax(self, depth: int, alpha: int, beta: int, ply: int = 0) -> int:
        validMoves = self.allValidMoves()
        validMoves.sort(key=lambda move: self._getMovePriority(move), reverse=True)

        # No Valid Moves = Lose
        if len(validMoves) == 0:
            return self.resolveEndGame(ply)

        # Termination Condition
        if depth == 0:
            return self._quiesceneSearch(alpha, beta)
        else:
            maxEval = float('-inf')

            for cmd in validMoves:
                removedPiece, prevEnPassant, prevCastleIndex = self.movePiece(cmd)
                score = (-1) * self._negamax(depth - 1, (-1) * beta, (-1) * alpha, ply + 1)
                self.undoMove(cmd, removedPiece, prevEnPassant, prevCastleIndex)

                maxEval = max(maxEval, score)
                alpha = max(alpha, score)
                
                if alpha >= beta:
                    break

            return maxEval


    # Resolve End Game -> Called when No Valid Moves
    def resolveEndGame(self, ply: int) -> int:
        opponentAttackTargets = self.allOpponentCaptureTargets()

        kingTuple = self.kingTuple(self.playerTurn)
        if kingTuple in opponentAttackTargets:
            return -1000000 + ply 
        
        # Otherwise, it is a Stalemate (Draw)
        return 0

    # MinMaxSearch -> General
    def _quiesceneSearch(self, alpha: int, beta: int, depth: int = 0) -> int:
        staticEval = self._computeBoardValue()

        if staticEval >= beta:
            return beta

        if staticEval > alpha:
            alpha = staticEval

        if depth >= 10:
            return staticEval

        validMoves = self.allValidMoves()
        validMoves.sort(key=lambda move: self._getMovePriority(move), reverse=True)

        for cmd in self._allQuiesceneMoves(validMoves):
            removedPiece, prevEnPassant, prevCastleIndex = self.movePiece(cmd)
            score = (-1) * self._quiesceneSearch((-1) * beta, (-1) * alpha, depth + 1)
            self.undoMove(cmd, removedPiece, prevEnPassant, prevCastleIndex)

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    # Return all Capture Moves
    def _allQuiesceneMoves(self, validMoves: list[MoveCommand]) -> list[MoveCommand]:
        # QuiescenceMoves
        quiescenceMoveCmd = [MoveCommandType.PROMOTE, MoveCommandType.CAPTURE, \
            MoveCommandType.ENPASSANT]

        return list(filter(
            lambda cmd: cmd.moveType in quiescenceMoveCmd, validMoves
        ))

    # Compute Pawn Penalizer for Isolated / Double Pawn
    def _pawnPenalizer(self, player: Player, phaseWeight: int) -> int:
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
                    earlyGame = 15
                    endGame = 25
                    computedPhase = earlyGame * phaseWeight + endGame * (24 - phaseWeight) 
                
                    penalizer += math.ceil(computedPhase / 24)

        return penalizer

    # Maximum Value is 24, used for early/ mid board evaluation
    def _calculateGamePhase(self) -> int:
        totalPhaseWeight = 0

        for row in range(0, 8):
            for col in range(0, 8):
                if self.board[row][col] != None:
                    totalPhaseWeight += self.board[row][col].phaseWeight()

        return totalPhaseWeight

    # Compute Move Priority
    def _getMovePriority(self, cmd: MoveCommand) -> int:
        # 1. Promotions (High Priority)
        if cmd.moveType == MoveCommandType.PROMOTE:
            return 90000 # Treat as Queen value
        
        # 2. Captures (MVV-LVA)
        if cmd.moveType in [MoveCommandType.CAPTURE, MoveCommandType.ENPASSANT]:
            capturedPieceVal = 0
            if cmd.moveType == MoveCommandType.ENPASSANT:
                capturedPieceVal = self.board[cmd.startRow][cmd.endCol].pieceValue()
            else:
                capturedPieceVal = self.board[cmd.endRow][cmd.endCol].pieceValue()

            startingPieceVal = self.board[cmd.startRow][cmd.startCol].pieceValue()  
            
            return (capturedPieceVal * 10) - startingPieceVal

        # 3. Castling (Mid Priority)
        if cmd.moveType in [MoveCommandType.KINGSIDECASTLE, MoveCommandType.QUEENSIDECASTLE]:
            return 50

        # 4. Standard Moves (Low Priority)
        return 0

    # --------

    @staticmethod
    def opponent(player: Player):
        if player == Player.WHITE:
            return Player.BLACK
        else:
            return Player.WHITE

    # Return all Valid moves for currentPlayer
    def allValidMoves(self) -> list[MoveCommand]:
        validMoves = []

        for row in range(0, 8):
            for col in range(0, 8):
                if self.board[row][col] != None:
                    if self.board[row][col].player == self.playerTurn:
                        for cmd in self.board[row][col].possibleMoves(self):
                            validMoves.append(cmd)

        return validMoves

    # Validate the Move
    def validateMove(self, initRow: int, initCol: int, targetRow: int, 
        targetCol: int, player: Player) -> MoveCommand:
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

    # Move Piece
    def movePiece(self, cmd: MoveCommand) -> Union[Optional[ChessPieceModel], int, int]:
        # Store for Undo
        removedPiece = None
        prevEnPassant = self.enPassant
        prevCastleIndex = ChessBoardZobrist.castleIndex(self)

        # Set enPassant to Null
        self.enPassant = None

        # Update ZobristHash
        ChessBoardZobrist.forwardUpdate(self, cmd, prevEnPassant)

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
                self.updateKingCastleFlag(True)

                # Forward Castle
                ChessBoardZobrist.forwardCastle(self, prevCastleIndex)

            # King Side Castle
            case MoveCommandType.KINGSIDECASTLE:
                # Determine the row of the Castle 
                row = 7 if self.playerTurn == Player.BLACK else 0

                # Move the King 2 steps to the left
                self._movePieceOnBoard(row, 4, row, 6)

                # Move the Rook to the right of the king
                self._movePieceOnBoard(row, 7, row, 5)

                # Update Castle
                self.updateKingCastleFlag(True)

                # Forward Castle
                ChessBoardZobrist.forwardCastle(self, prevCastleIndex)

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

                self.enPassant = cmd.endCol

            # Promote Pawn
            case MoveCommandType.PROMOTE:
                # Promote MAY be a capture or a move
                removedPiece = self.board[cmd.endRow][cmd.endCol]

                # Promote Piece Type is provided from above
                self.board[cmd.endRow][cmd.endCol] = ChessPieceFactory.createChessPiece(
                    PieceType.QUEEN, self.playerTurn, cmd.endRow, cmd.endCol)

                # Remove Pawn
                self.board[cmd.startRow][cmd.startCol] = None

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
        return removedPiece, self.enPassant, prevCastleIndex

    # Undo Move - Used to for Pruning
    def undoMove(self, cmd: MoveCommand, restorePiece: Optional[ChessPieceModel], 
        prevEnPassant: int, prevCastleIndex: int) -> None:
        # Swap the Player Turn
        self.playerTurn = ChessBoardModel.opponent(self.playerTurn)

        # Undo ZobristHash
        ChessBoardZobrist.backwardUpdate(self, cmd, restorePiece, prevEnPassant)

        match cmd.moveType:
            # Queen Side Castle (Undo)
            case MoveCommandType.QUEENSIDECASTLE:
                # Determine the row of the Castle 
                row = 7 if self.playerTurn == Player.BLACK else 0

                # Move the Rook to the right of the king
                self._undoMoveOnBoard(row, 0, row, 3)

                # Move the King 2 steps to the left
                self._undoMoveOnBoard(row, 4, row, 2)

                # Update Castle
                self.updateKingCastleFlag(False)

                # Backward Castle
                ChessBoardZobrist.backwardCastle(self, prevCastleIndex)

            # King Side Castle
            case MoveCommandType.KINGSIDECASTLE:
                 # Determine the row of the Castle 
                row = 7 if self.playerTurn == Player.BLACK else 0

                # Move the Rook to the right of the king
                self._undoMoveOnBoard(row, 7, row, 5)

                # Move the King 2 steps to the left
                self._undoMoveOnBoard(row, 4, row, 6)

                # Update Castle
                self.updateKingCastleFlag(False)

                # Backward Castle
                ChessBoardZobrist.backwardCastle(self, prevCastleIndex)

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

        # Set En Passant to previous State
        self.enPassant = prevEnPassant

    # This is used to determine castle eligability
    def allOpponentCaptureTargets(self) -> set[tuple[int, int]]:
        captureSquares = set()

        opponent = ChessBoardModel.opponent(self.playerTurn)
        for row in range(0, 8):
            for col in range(0, 8):
                targetPiece = self.board[row][col]
                if targetPiece != None and targetPiece.player == opponent:

                    captureTargets = targetPiece.captureTargets(self)
                    for target in captureTargets:
                        captureSquares.add(target)
        
        return captureSquares

    # Used for evaluating Castling Rights
    def _updateKing(self, startRow: int, startCol: int) -> None:
        kingPiece = self.board[startRow][startCol]

        # Update the King Square
        if kingPiece.type == PieceType.KING:
            if kingPiece.player == Player.BLACK:
                self.blackKingSquareRow = startRow
                self.blackKingSquareCol = startCol

                self.blackCanQueenSide = \
                    kingPiece.canQueenSideCastle(self)
                self.blackCanKingSide = \
                    kingPiece.canKingSideCastle(self)

            else:
                self.whiteKingSquareRow = startRow
                self.whiteKingSquareCol = startCol

                self.whiteCanQueenSide = \
                    kingPiece.canQueenSideCastle(self)

                self.whiteCanKingSide = \
                    kingPiece.canKingSideCastle(self)

    # Helper Method, Move Piece
    def _movePieceOnBoard(self, startRow: int, startCol: int, endRow: int, endCol: int) -> None:
        self.board[endRow][endCol] = self.board[startRow][startCol]
        self.board[endRow][endCol].row = endRow
        self.board[endRow][endCol].col = endCol
        self.board[endRow][endCol].moves += 1

        # Remove the Init Piece
        self.board[startRow][startCol] = None

        # Update King Square
        self._updateKing(endRow, endCol)

    # Helper Method, Undo Move Piece
    def _undoMoveOnBoard(self, originalRow: int, originalCol: int, 
        currentRow: int, currentCol: int) -> None:        
        self.board[originalRow][originalCol] = self.board[currentRow][currentCol] 
        self.board[originalRow][originalCol].row = originalRow
        self.board[originalRow][originalCol].col = originalCol

        # Undo Castle will take in account of both Rook and King
        self.board[originalRow][originalCol].moves -= 1

        # Remove the Final Piece
        self.board[currentRow][currentCol] = None

        # Update the King Square
        self._updateKing(originalRow, originalCol)

    # Validate King Safety for player after making move
    def validateKingSafety(self, cmd: MoveCommand) -> bool:
        testPlayerTurn = self.playerTurn
        removedPiece, prevEnPassant, prevCastleIndex = self.movePiece(cmd)
        returnValue = self._testKingSafety(testPlayerTurn)
        self.undoMove(cmd, removedPiece, prevEnPassant, prevCastleIndex)
        return returnValue

    # Update King Castle
    def updateKingCastleFlag(self, castled: bool):
        kingRow, kingCol = self.kingTuple(self.playerTurn)
        self.board[kingRow][kingCol].updateCastle(castled)

    # Retrieve King Tuple
    def kingTuple(self, player: Player) -> tuple[int, int]:
        if player == Player.WHITE:
            return (self.whiteKingSquareRow, self.whiteKingSquareCol)
        else:
            return (self.blackKingSquareRow, self.blackKingSquareCol)

    # Test King Safety
    def _testKingSafety(self, kingPlayer: Player) -> bool:
        row, col = self.kingTuple(kingPlayer)

        # Test for Opponent Knights
        for dr, dc in [(2, 1), (1, 2), (-2, -1), (-1, -2), (-2, 1), (-1, 2), (2, -1), (1, -2)]:
            newRow = row + dr
            newCol = col + dc
            if newRow >= 0 and newRow < 8 and newCol >= 0 and newCol < 8:
                target = self.board[newRow][newCol] 
                if target != None and target.type == PieceType.KNIGHT \
                    and target.player != kingPlayer:
                    return False

        # Test for Opponent Horizontals
        horizontalCapture = [PieceType.ROOK, PieceType.QUEEN]
        for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            for i in range(1, 8):
                newRow = row + dr * i
                newCol = col + dc * i

                if not (newRow >= 0 and newRow < 8 and newCol >= 0 and newCol < 8):
                    break

                target = self.board[newRow][newCol]
                if target != None:
                    if target.player != kingPlayer:
                        if i == 1 and target.type == PieceType.KING: 
                            return False
                        if target.type in horizontalCapture:
                            return False
                    break # Path Blocked

        # Test for Diagonal Captures
        diagonalCapture = [PieceType.BISHOP, PieceType.QUEEN]
        for dr, dc in [(-1, 1), (1, 1), (1, -1), (-1, -1)]:
            for i in range(1, 8):
                newRow = row + dr * i
                newCol = col + dc * i

                if not (newRow >= 0 and newRow < 8 and newCol >= 0 and newCol < 8):
                    break

                target = self.board[newRow][newCol]
                if target != None:
                    if target.player != kingPlayer:
                        if i == 1 and target.type == PieceType.KING: 
                            return False
                        if target.type in diagonalCapture:
                            return False
                    break # Path Blocked

        # Test for enemy pawns 
        pawnRow = row - 1 
        if kingPlayer == Player.WHITE:
            pawnRow = row + 1

        if 0 <= pawnRow < 8:
            for columnDirection in [-1, 1]:
                newCol = col + columnDirection

                if 0 <= newCol < 8:
                    target = self.board[pawnRow][newCol]
                    if target and target.type == PieceType.PAWN \
                        and target.player != kingPlayer:
                        return False

        return True

