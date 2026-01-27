# Impoprt
import secrets
import math

# Enum
from appEnums import Player, MoveCommandType, PieceType

# Import Model
from modelComponent.moveCommand import MoveCommand
from modelComponent.chessPieceModel import ChessPieceModel
from modelComponent.chessBoardProtocal import ChessBoardProtocal

# Factory
from modelFactory.chessPieceFactory import ChessPieceFactory

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
    def computeInitValue(chessBoard: ChessBoardProtocal) -> int:
        h = 0
        # Pieces
        for row in range(0, 8):
            for col in range(0, 8):
                chessPiece = chessBoard.board[row][col]

                sq = row * 8 + col
                if chessPiece:
                    p_idx = ChessBoardZobrist.pieceType(chessPiece)
                    h ^= ChessBoardZobrist.TABLE[p_idx][sq]
        
        # Side to move
        if chessBoard.playerTurn == Player.BLACK:
            h ^= ChessBoardZobrist.BlACK_TO_MOVE
            
        # Castling
        castleIndex = ChessBoardZobrist.castleIndex(chessBoard)
        h ^= ChessBoardZobrist.CASTLING[castleIndex]

        return h

    @staticmethod
    def pieceType(chessPiece: ChessPieceModel):
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
    def undoMovePiece(chessBoard: ChessBoardProtocal, 
        initRow: int, initCol: int, finalRow: int, finalCol: int):

        ChessBoardZobrist.movePiece(chessBoard, finalRow, finalCol, initRow, initCol)

    @staticmethod
    def movePiece(chessBoard: ChessBoardProtocal, 
        initRow: int, initCol: int, finalRow: int, finalCol: int):

        # XOR InitRow / InitCol
        startSq = initRow * 8 + initCol
        startPiece = chessBoard.board[initRow][initCol]
        startType = ChessBoardZobrist.pieceType(startPiece)
        chessBoard.zobristHash ^= ChessBoardZobrist.TABLE[startType][startSq]

        # XOR finalRow / finalCol
        endSq = finalRow * 8 + finalCol
        chessBoard.zobristHash ^= ChessBoardZobrist.TABLE[startType][endSq]

    @staticmethod
    def forwardCastle(chessBoard: ChessBoardProtocal, prevCastleIndex: int):
        currCastleIndex = ChessBoardZobrist.castleIndex(chessBoard)

        if currCastleIndex != prevCastleIndex:
            # Unset Previous Castle Zobrist
            chessBoard.zobristHash ^= ChessBoardZobrist.CASTLING[prevCastleIndex]

            # Compute new Zobrists
            chessBoard.zobristHash ^= ChessBoardZobrist.CASTLING[currCastleIndex]

    @staticmethod
    def removePiece(chessBoard: ChessBoardProtocal, initRow: int, initCol: int):
        # XOR int / col
        startSq = initRow * 8 + initCol
        startPiece = chessBoard.board[initRow][initCol]
        startType = ChessBoardZobrist.pieceType(startPiece)
        chessBoard.zobristHash ^= ChessBoardZobrist.TABLE[startType][startSq]

    @staticmethod
    def restorePiece(chessBoard: ChessBoardProtocal, 
        initRow: int, initCol: int, restorePieceType: int):
        # XOR int / col
        startSq = initRow * 8 + initCol
        chessBoard.zobristHash ^= ChessBoardZobrist.TABLE[restorePieceType][startSq]

    @staticmethod
    def addQueen(chessBoard: ChessBoardProtocal, row: int, col: int):
        queenIndex = 0
        if chessBoard.playerTurn == Player.WHITE:
            queenIndex = 1
        else:
            queenIndex = 7

        endSq = row * 8 + col
        chessBoard.zobristHash ^= ChessBoardZobrist.TABLE[queenIndex][endSq]

    @staticmethod
    def forwardUpdate(chessBoard: ChessBoardProtocal, cmd: MoveCommand, prevEnPassant: int):

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
                ChessBoardZobrist.removePiece(chessBoard, cmd.endRow, cmd.endCol)

                # Move Starting Piece to Capture Point
                ChessBoardZobrist.movePiece(
                    chessBoard, cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

            # Double Pawn Move
            case MoveCommandType.PAWNOPENMOVE:
                # Move the piece from start to end
                ChessBoardZobrist.movePiece(
                    chessBoard, cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

            # Promote Pawn
            case MoveCommandType.PROMOTE:
                # Promote MAY be a capture or a move
                if chessBoard.board[cmd.endRow][cmd.endCol]:
                    ChessBoardZobrist.removePiece(chessBoard, cmd.endRow, cmd.endCol)

                # Promote Piece Type is provided from above
                ChessBoardZobrist.addQueen(chessBoard, cmd.endRow, cmd.endCol)

                ChessBoardZobrist.removePiece(chessBoard, cmd.startRow, cmd.startCol)

            # En Passant
            case MoveCommandType.ENPASSANT:
                # Remove En Passant Pawn
                ChessBoardZobrist.removePiece(chessBoard, cmd.startRow, cmd.endCol)

                # Move the Pawn to the target
                ChessBoardZobrist.movePiece(chessBoard, cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

    @staticmethod
    def backwardUpdate(chessBoard: ChessBoardProtocal, 
        cmd: MoveCommand, restorePiece: Optional[ChessPieceModel], prevEnPassant: int):

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
                # Move Starting Piece to Capture Point
                ChessBoardZobrist.undoMovePiece(
                    chessBoard, cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

                # Remove the Previous Piece
                pieceType = ChessBoardZobrist.pieceType(restorePiece)
                ChessBoardZobrist.restorePiece(chessBoard, cmd.endRow, cmd.endCol, pieceType)

            # Double Pawn Move
            case MoveCommandType.PAWNOPENMOVE:
                # Move the piece from start to end
                ChessBoardZobrist.undoMovePiece(
                    chessBoard, cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

            # Promote Pawn
            case MoveCommandType.PROMOTE:
                # Remove Queen
                ChessBoardZobrist.removePiece(chessBoard, cmd.endRow, cmd.endCol)

                # Promote MAY be a capture or a move
                if restorePiece:
                    pieceType = ChessBoardZobrist.pieceType(restorePiece)
                    ChessBoardZobrist.restorePiece(chessBoard, cmd.endRow, cmd.endCol, pieceType)

                # Restore Pawn
                newPawn = ChessPieceFactory.createChessPiece(
                    PieceType.PAWN, chessBoard.playerTurn, cmd.startRow, cmd.startCol)
                pawnType = ChessBoardZobrist.pieceType(newPawn)
                ChessBoardZobrist.restorePiece(chessBoard, cmd.startRow, cmd.startCol, pawnType)

            # En Passant
            case MoveCommandType.ENPASSANT:
                # Remove En Passant Pawn
                ChessBoardZobrist.undoMovePiece(chessBoard, cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

                # Move the Pawn to the target
                pieceType = ChessBoardZobrist.pieceType(restorePiece)
                ChessBoardZobrist.restorePiece(chessBoard, cmd.startRow, cmd.endCol, pieceType)
        
