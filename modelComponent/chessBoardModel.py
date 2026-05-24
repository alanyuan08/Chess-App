# Enum 
from appEnums import PieceType, Player, MoveCommandType, PROMOTION_MAP

# Model 
from modelComponent.moveCommand import MoveCommand
from modelComponent.chessPieceModel import ChessPieceModel
from modelComponent.chessBoardZobrist import ChessBoardZobrist

# Factory
from modelFactory.chessPieceFactory import ChessPieceFactory

# Standard
import math
from typing import Optional, Union

# Controller 
class ChessBoardModel():
    def __init__(self):
        # Create the Chess Board
        self.board = [[None for _ in range(8)] for _ in range(8)]

        # This is maintained for backtracking
        self.playerTurn = Player.WHITE

        # En Passant Column -> 8 is Null
        self.enPassant = 8

        # Used to Check for King Safety
        self.whiteKingSquareRow = 0
        self.whiteKingSquareCol = 4

        self.blackKingSquareRow = 7
        self.blackKingSquareCol = 4

        # Used to Store previous Moves in 
        self.previousMoves = []

        # Use for Zobrist Hash
        self.whiteCanQueenSide = True
        self.whiteCanKingSide = True

        self.blackCanQueenSide = True
        self.blackCanKingSide = True

        # Zobrist Hash
        self.zobristHash = ChessBoardZobrist.computeInitValue(self)

        # Traversed Positions - Zobrist Hash
        self.traversedPositions = {}
        self.forwardPosition()

    # Used to Determine if Checkmate Or Draw
    def checkMate(self) -> bool:
        opponentAttackTargets = self.allOpponentCaptureTargets()
        kingTuple = self.kingTuple(self.playerTurn)
        if kingTuple in opponentAttackTargets:
            return True

        return False

    def forwardPosition(self):
        self.traversedPositions[self.zobristHash] = \
            self.traversedPositions.get(self.zobristHash, 0) + 1

    def backtrackPosition(self):
        self.traversedPositions[self.zobristHash] = \
            self.traversedPositions.get(self.zobristHash, 0) - 1

    def checkThreeMoveRepetiton(self):
        if self.traversedPositions.get(self.zobristHash, 0) == 3:
            return True
        else:
            return False

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
    def movePiece(self, cmd: MoveCommand) -> Union[Optional[ChessPieceModel], int]:

        # Store for Undo
        removedPiece = None
        prevCastleIndex = ChessBoardZobrist.castleIndex(self)
        prevEnPassant = self.enPassant

        # Set enPassant to Null
        self.enPassant = 8

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
            case (MoveCommandType.PROMOTION_QUEEN | 
                MoveCommandType.PROMOTION_ROOK | 
                MoveCommandType.PROMOTION_BISHOP | 
                MoveCommandType.PROMOTION_KNIGHT):
                # Promote MAY be a capture or a move
                removedPiece = self.board[cmd.endRow][cmd.endCol]

                promote_piece = PROMOTION_MAP.get(cmd.moveType, None)

                # Promote Piece Type is provided from above
                self.board[cmd.endRow][cmd.endCol] = ChessPieceFactory.createChessPiece(
                    promote_piece, self.playerTurn, cmd.endRow, cmd.endCol)

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

        # Recompute Castle for both Players
        self.recomputeCastle(cmd, removedPiece)

        # Update Zobrist Castle to Curr Castle Index
        ChessBoardZobrist.forwardCastle(self, prevCastleIndex)

        # Update En Passant Forward
        ChessBoardZobrist.forwardEnPassant(self, prevEnPassant)

        # Update Board Position
        self.forwardPosition()

        # Append Previous Position
        self.previousMoves.append(
            f"{cmd.startCol}{cmd.startRow}{cmd.endCol}{cmd.endRow}{cmd.moveType.value}"
        )

        # Create a new copy of the removed Piece
        return removedPiece, prevEnPassant

    # Undo Move - Used to for Pruning
    def undoMove(self, cmd: MoveCommand, restorePiece: Optional[ChessPieceModel], prevEnPassant: int) -> None:
        # Swap the Player Turn
        self.playerTurn = ChessBoardModel.opponent(self.playerTurn)
        prevCastleIndex = ChessBoardZobrist.castleIndex(self)

        # Remove Last Move
        self.previousMoves.pop()

        # Backtrack Board Position
        self.backtrackPosition()

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
            case (MoveCommandType.PROMOTION_QUEEN | 
                MoveCommandType.PROMOTION_ROOK | 
                MoveCommandType.PROMOTION_BISHOP | 
                MoveCommandType.PROMOTION_KNIGHT):
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

        # Recompute Castle for both Players
        self.recomputeCastle(cmd, restorePiece)

        # Update Zobrist Castle
        ChessBoardZobrist.forwardCastle(self, prevCastleIndex)

        # Update En Passant Forward
        ChessBoardZobrist.forwardEnPassant(self, prevEnPassant)

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
    def recomputeCastle(self, cmd: MoveCommand, capturedPiece: Optional[ChessPieceModel]) -> None:
        if cmd == MoveCommandType.MOVE:
            movePiece = self.board[cmd.endRow][cmd.endCol]

            if movePiece.type == PieceType.ROOK or movePiece.type == PieceType.KING:
                if movePiece.player == Player.BLACK:
                    # Black King/ Rook Moved, Recompute
                    blackKing = \
                        self.board[self.blackKingSquareRow][self.blackKingSquareCol]

                    self.blackCanQueenSide = \
                        blackKing.canQueenSideCastle(self)
                    self.blackCanKingSide = \
                        blackKing.canKingSideCastle(self)
                else:
                    # White King / Rook Move, Recompute
                    whiteKing = \
                        self.board[self.whiteKingSquareRow][self.whiteKingSquareCol]

                    self.whiteCanQueenSide = \
                        whiteKing.canQueenSideCastle(self)

                    self.whiteCanKingSide = \
                        whiteKing.canKingSideCastle(self)

        elif cmd == MoveCommandType.CAPTURE and capturedPiece.type == PieceType.ROOK:
            if capturedPiece.player == Player.BLACK:
                if movepiece.type == PieceType.ROOK:
                    blackKing = \
                        self.board[self.blackKingSquareRow][self.blackKingSquareCol]

                    self.blackCanQueenSide = \
                        blackKing.canQueenSideCastle(self)
                    self.blackCanKingSide = \
                        blackKing.canKingSideCastle(self)

            elif capturedPiece.player == Player.WHITE:
                if movepiece.type == PieceType.ROOK:
                    # White King / Rook Move, Recompute
                    whiteKing = \
                        self.board[self.whiteKingSquareRow][self.whiteKingSquareCol]

                    self.whiteCanQueenSide = \
                        whiteKing.canQueenSideCastle(self)

                    self.whiteCanKingSide = \
                        whiteKing.canKingSideCastle(self)

    # Update King Square
    def _updateKingSquare(self, startRow: int, startCol: int) -> None:
        movePiece = self.board[startRow][startCol]

        # Update the King Square
        if movePiece.type == PieceType.KING:
            if movePiece.player == Player.BLACK:
                self.blackKingSquareRow = startRow
                self.blackKingSquareCol = startCol

            else:
                self.whiteKingSquareRow = startRow
                self.whiteKingSquareCol = startCol

    # Helper Method, Move Piece
    def _movePieceOnBoard(self, startRow: int, startCol: int, endRow: int, endCol: int) -> None:
        self.board[endRow][endCol] = self.board[startRow][startCol]
        self.board[endRow][endCol].row = endRow
        self.board[endRow][endCol].col = endCol
        self.board[endRow][endCol].moves += 1

        # Remove the Init Piece
        self.board[startRow][startCol] = None

        # Update the King Square
        self._updateKingSquare(endRow, endCol)

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
        self._updateKingSquare(originalRow, originalCol)

    # Validate King Safety for player after making move
    def validateKingSafety(self, cmd: MoveCommand) -> bool:
        testPlayerTurn = self.playerTurn
        removedPiece, prevEnPassant = self.movePiece(cmd)
        returnValue = self._testKingSafety(testPlayerTurn)
        self.undoMove(cmd, removedPiece, prevEnPassant)
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
