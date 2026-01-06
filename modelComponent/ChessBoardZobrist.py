# Impoprt
import secrets
import math

# Enum
from appEnums import Player, MoveCommandType, PieceType

# Import Model
from modelComponent.moveCommand import MoveCommand
from modelComponent.chessPieceModel import ChessPieceModel
from modelComponent.chessBoardProtocal import ChessBoardProtocal

# Standard Piece Map: 6 pieces * 2 colours = 12 total
# We use indices 0-5 for White (P, N, B, R, Q, K) and 6-11 for Black
PIECE_TYPES = 12
SQUARES = 64

# Controller 
class ChessBoardZobrist():
    # 1. Piece-Square table: [piece_type][square]
    TABLE = [[secrets.randbits(64) for _ in range(SQUARES)] 
                  for _ in range(PIECE_TYPES)]
    
    # 2. Side to move (only need one for Black; if White, we don't XOR it)
    BlACK_TO_MOVE = secrets.randbits(64)
    
    # 3. Castling Rights (16 possible states: 4 bits for KQkq)
    CASTLING = [secrets.randbits(64) for _ in range(16)]
    
    # 4. En Passant (8 files where a capture can happen, +1 for "none")
    EN_PASSANT = [secrets.randbits(64) for _ in range(8)]

    @staticmethod
    def pieceIndex(chessPiece: ChessPieceModel):
        returnValue = 0
        match chessPiece.type:
            case PieceType.KING:
                returnValue = 0
            case PieceType.QUEEN:
                returnValue = 1
            case PieceType.KNIGHT:
                returnValue = 2
            case PieceType.ROOK:
                returnValue = 3
            case PieceType.PAWN:
                returnValue = 4
            case PieceType.BISHOP:
                returnValue = 5

        if chessPiece.player == Player.BLACK:
            returnValue += 6

        return returnValue

    @staticmethod
    def castleIndex(chessBoard: ChessBoardProtocal):
        returnIndex = 0 

        if chessBoard.whiteCanQueenSide:
            returnIndex += 1

        if chessBoard.whiteCanKingSide:
            returnIndex += 2

        if chessBoard.blackCanQueenSide:
            returnIndex += 4

        if chessBoard.blackCanKingSide:
            returnIndex += 8

        return returnIndex

    @staticmethod
    def computeInitValue(chessBoard: ChessBoardProtocal):
        h = 0
        # Pieces
        for row in range(0, 8):
            for col in range(0, 8):
                chessPiece = chessBoard.board[row][col]

                sq = row * 8 + col
                if chessPiece:
                    p_idx = ChessBoardZobrist.pieceIndex(chessPiece)
                    h ^= ChessBoardZobrist.TABLE[p_idx][sq]
        
        # Side to move
        if chessBoard.playerTurn == Player.BLACK:
            h ^= ChessBoardZobrist.BlACK_TO_MOVE
            
        # Castling
        castleIndex = ChessBoardZobrist.castleIndex(chessBoard)
        h ^= ChessBoardZobrist.CASTLING[castleIndex]

        # En Passant - 8 is No En Passant
        enPassantCol = 8

        if self.enPassant:
            h ^= ChessBoardZobrist.EN_PASSANT[enPassantCol]
        
        return h

    @staticmethod
    def undoMovePiece(chessBoard: ChessBoardProtocal, 
        initRow: int, initCol: int, finalRow: int, finalCol: int):

        ChessBoardZobrist.movePiece(chessBoard, finalRow, finalCol, initRow, initCol)

    @staticmethod
    def movePiece(chessBoard: ChessBoardProtocal, 
        initRow: int, initCol: int, finalRow: int, finalCol: int):

        # XOR InitRow / InitCol
        startSq = initRow * 8 + initCol
        startPiece = chessBoard.board[initRow][initCol]
        startIndex = ChessBoardZobrist.pieceIndex(startPiece)
        chessBoard.zobristHash ^= ChessBoardZobrist.TABLE[startIndex][startSq]

        # XOR finalRow / finalCol
        endSq = finalRow * 8 + finalCol
        endPiece = chessBoard.board[finalRow][finalCol]
        endIndex = ChessBoardZobrist.pieceIndex(endPiece)
        chessBoard.zobristHash ^= ChessBoardZobrist.TABLE[endIndex][endSq]

    @staticmethod
    def addOrRemovePiece(chessBoard: ChessBoardProtocal, row: int, col: int):
        # XOR int / col
        startSq = initRow * 8 + initCol
        startPiece = chessBoard.board[initRow][initCol]
        startIndex = ChessBoardZobrist.pieceIndex(startPiece)
        chessBoard.zobristHash ^= ChessBoardZobrist.TABLE[startIndex][startSq]

    @staticmethod
    def addOrRemoveQueen(chessBoard: ChessBoardProtocal, row: int, col: int):
        queenIndex = 0
        if chessBoard.playerTurn == Player.WHITE:
            queenIndex = 1
        else:
            queenIndex = 7

        endSq = row * 8 + col
        chessBoard.zobristHash ^= ChessBoardZobrist.TABLE[queenIndex][endSq]

    @staticmethod
    def forwardUpdate(chessBoard: ChessBoardProtocal, 
        cmd: MoveCommand, prevCastleIndex: int, prevEnPassant: int):
        # Forward Turn
        chessBoard.zobristHash ^= ChessBoardZobrist.BlACK_TO_MOVE

        # Remove Prev EnPassant

        match cmd.moveType:
            # Queen Side Castle
            case MoveCommandType.QUEENSIDECASTLE:
                # Determine the row of the Castle 
                row = 7 if chessBoard.playerTurn == Player.BLACK else 0

                # Move the King 2 steps to the left
                ChessBoardZobrist.undoMovePiece(chessBoard, row, 4, row, 2)

                # Move the Rook to the right of the king
                ChessBoardZobrist.undoMovePiece(chessBoard, row, 0, row, 3)

            # King Side Castle
            case MoveCommandType.KINGSIDECASTLE:
                # Determine the row of the Castle 
                row = 7 if chessBoard.playerTurn == Player.BLACK else 0

                # Move the King 2 steps to the left
                ChessBoardZobrist.undoMovePiece(chessBoard, row, 4, row, 6)

                # Move the Rook to the right of the king
                ChessBoardZobrist.undoMovePiece(chessBoard, row, 7, row, 5)

            # Move Piece
            case MoveCommandType.MOVE:
                # Move Starting Piece to Capture Point
                ChessBoardZobrist.undoMovePiece(
                    chessBoard, cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

            case MoveCommandType.CAPTURE:
                # Remove the Previous Piece
                ChessBoardZobrist.addOrRemovePiece(chessBoard, cmd.endRow, cmd.endCol)

                # Move Starting Piece to Capture Point
                ChessBoardZobrist.undoMovePiece(
                    chessBoard, cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

            # Double Pawn Move
            case MoveCommandType.PAWNOPENMOVE:
                # Move the piece from start to end
                ChessBoardZobrist.undoMovePiece(
                    chessBoard, cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

                ChessBoardZobrist.chessBoard ^= ChessBoardZobrist.en_passant[cmd.startCol]

            # Promote Pawn
            case MoveCommandType.PROMOTE:
                # Promote MAY be a capture or a move
                if chessBoard.board[cmd.endRow][cmd.endCol]:
                    ChessBoardZobrist.addOrRemovePiece(chessBoard, cmd.endRow, cmd.endCol)

                # Promote Piece Type is provided from above
                ChessBoardZobrist.addOrRemoveQueen(chessBoard, cmd.endRow, cmd.endCol)

                # Remove Pawn
                ChessBoardZobrist.addOrRemovePiece(chessBoard, cmd.startRow, cmd.startCol)

            # En Passant
            case MoveCommandType.ENPASSANT:
                # Remove En Passant Pawn
                ChessBoardZobrist.addOrRemovePiece(chessBoard, cmd.startRow, cmd.endCol)

                # Move the Pawn to the target
                ChessBoardZobrist.addOrRemovePiece(cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

        # Unset Previous Castle Zobrist
        chessBoard.zobristHash ^= ChessBoardZobrist.CASTLING[prevIndex]

        # Compute new Zobrist
        castleIndex = ChessBoardZobrist.castleIndex(chessBoard)
        chessBoard.zobristHash ^= ChessBoardZobrist.CASTLING[castleIndex]

    @staticmethod
    def backwardUpdate(chessBoard: ChessBoardProtocal, cmd: MoveCommand, prevIndex: int):
        # Rollback Turn
        chessBoard.zobristHash ^= ChessBoardZobrist.BlACK_TO_MOVE

        match cmd.moveType:
            # Queen Side Castle
            case MoveCommandType.QUEENSIDECASTLE:
                # Determine the row of the Castle 
                row = 7 if chessBoard.playerTurn == Player.BLACK else 0

                # Move the King 2 steps to the left
                ChessBoardZobrist.movePiece(chessBoard, row, 4, row, 2)

                # Move the Rook to the right of the king
                ChessBoardZobrist.movePiece(chessBoard, row, 0, row, 3)

            # King Side Castle
            case MoveCommandType.KINGSIDECASTLE:
                # Determine the row of the Castle 
                row = 7 if chessBoard.playerTurn == Player.BLACK else 0

                # Move the King 2 steps to the left
                ChessBoardZobrist.movePiece(chessBoard, row, 4, row, 6)

                # Move the Rook to the right of the king
                ChessBoardZobrist.movePiece(chessBoard, row, 7, row, 5)

            # Move Piece
            case MoveCommandType.MOVE:
                # Move Starting Piece to Capture Point
                ChessBoardZobrist.movePiece(
                    chessBoard, cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

            case MoveCommandType.CAPTURE:
                # Remove the Previous Piece
                ChessBoardZobrist.addOrRemovePiece(chessBoard, cmd.endRow, cmd.endCol)

                # Move Starting Piece to Capture Point
                ChessBoardZobrist.movePiece(
                    chessBoard, cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

            # Double Pawn Move
            case MoveCommandType.PAWNOPENMOVE:
                # Move the piece from start to end
                ChessBoardZobrist.movePiece(
                    chessBoard, cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

                ChessBoardZobrist.chessBoard ^= ChessBoardZobrist.en_passant[cmd.startCol]

            # Promote Pawn
            case MoveCommandType.PROMOTE:
                # Promote MAY be a capture or a move
                if chessBoard.board[cmd.endRow][cmd.endCol]:
                    ChessBoardZobrist.addOrRemovePiece(chessBoard, cmd.endRow, cmd.endCol)

                # Promote Piece Type is provided from above
                ChessBoardZobrist.addOrRemoveQueen(chessBoard, cmd.endRow, cmd.endCol)

                # Remove Pawn
                ChessBoardZobrist.addOrRemovePiece(chessBoard, cmd.startRow, cmd.startCol)

            # En Passant
            case MoveCommandType.ENPASSANT:
                # Remove En Passant Pawn
                ChessBoardZobrist.addOrRemovePiece(chessBoard, cmd.startRow, cmd.endCol)

                # Move the Pawn to the target
                ChessBoardZobrist.addOrRemovePiece(cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

        # Unset Current Zobrist Hash
        castleIndex = ChessBoardZobrist.castleIndex(chessBoard)
        chessBoard.zobristHash ^= ChessBoardZobrist.CASTLING[castleIndex]

        # Set to Previous Zobrit
        chessBoard.zobristHash ^= ChessBoardZobrist.CASTLING[prevIndex]
