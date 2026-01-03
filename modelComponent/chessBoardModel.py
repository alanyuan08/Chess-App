# Enum 
from appEnums import PieceType, Player, MoveCommandType

# Model 
from modelComponent.moveCommand import MoveCommand
from modelComponent.chessPieceModel import ChessPieceModel

# Factory
from modelFactory.chessPieceFactory import ChessPieceFactory

# Normal
from typing import Optional

# Controller 
class ChessBoardModel():
    def __init__(self):
        # Create the Chess Board
        self.board = [[None for _ in range(8)] for _ in range(8)]

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

    # Move Piece
    def movePiece(self, cmd: MoveCommand) -> Optional[ChessPieceModel]:
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

                # Promote Piece Type is provided from above
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

    # Undo Move - Used to for Pruning
    def undoMove(self, cmd: MoveCommand, restorePiece: Optional[ChessPieceModel]) -> None:
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

    # Used for evaluating King Safety
    def _updateKingSquare(self, kingRow: int, kingCol: int) -> None:
        # Update the King Square
        if self.board[kingRow][kingCol].type == PieceType.KING:
            if self.board[kingRow][kingCol].player == Player.BLACK:
                self.blackKingSquareRow = kingRow
                self.blackKingSquareCol = kingCol

            elif self.board[kingRow][kingCol].player == Player.WHITE:
                self.whiteKingSquareRow = kingRow
                self.whiteKingSquareCol = kingCol

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
        # This is explictly defined here to avoid confusion after the move
        currentPlayer = self.playerTurn
        removedPiece = self.movePiece(cmd)

        returnValue = self._testKingSafety(currentPlayer)
        self.undoMove(cmd, removedPiece)
        return returnValue

    # Retrieve King Tuple
    def kingTuple(self, player: Player) -> tuple[int, int]:
        if player == Player.WHITE:
            return (self.whiteKingSquareRow, self.whiteKingSquareCol)
        else:
            return (self.blackKingSquareRow, self.blackKingSquareCol)

    # Test King Safety
    def _testKingSafety(self, player: Player) -> bool:
        row, col = self.kingTuple(player)

        # Test for Opponent Knights
        for dr, dc in [(2, 1), (1, 2), (-2, -1), (-1, -2), (-2, 1), (-1, 2), (2, -1), (1, -2)]:
            newRow = row + dr
            newCol = col + dc
            if newRow >= 0 and newRow < 8 and newCol >= 0 and newCol < 8:
                target = self.board[newRow][newCol] 
                if target != None and target.type == PieceType.KNIGHT \
                    and target.player != self.playerTurn:
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
                    if target.player != player:
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
                    if target.player != player:
                        if i == 1 and target.type == PieceType.KING: 
                            return False
                        if target.type in diagonalCapture:
                            return False
                    break # Path Blocked

        # Test for enemy pawns 
        pawnRow = row - 1 
        if player == Player.WHITE:
            pawnRow = row + 1

        if 0 <= pawnRow < 8:
            for columnDirection in [-1, 1]:
                newCol = col + columnDirection

                if 0 <= newCol < 8:
                    target = self.board[pawnRow][newCol]
                    if target and target.type == PieceType.PAWN \
                        and target.player != player:
                        return False

        return True

