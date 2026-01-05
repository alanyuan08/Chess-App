# Impoprt
import secrets
import math

# Enum
from appEnums import Player, MoveCommandType, PieceType

# Import Model
from modelComponent.chessBoardProtocal import ChessBoardProtocal
from modelComponent.chessPieceModel import ChessPieceModel

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
    EN_PASSANT = [secrets.randbits(64) for _ in range(9)]

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
        for sq in range(0, 64):
            row = math.floor(sq/8)
            col = sq % 8 
            chessPiece = chessBoard.board[row][col]
            if chessPiece:
                # map piece type/color to 0-11
                p_idx = ChessBoardZobrist.pieceIndex(chessPiece)
                h ^= ChessBoardZobrist.TABLE[p_idx][sq]
        
        # Side to move
        if chessBoard.playerTurn == Player.BLACK:
            h ^= ChessBoardZobrist.BlACK_TO_MOVE
            
        # Castling
        castleIndex = ChessBoardZobrist.castleIndex(chessBoard)
        h ^= ChessBoardZobrist.castling[castleIndex]

        # En Passant - 8 is No En Passant
        enPassantCol = 8

        if chessBoard.playerTurn == Player.BLACK:
            if self.whiteEnPassantColumn != None:
                enPassantCol = self.whiteEnPassantColumn

        if chessBoard.playerTurn == Player.WHITE:
            if self.blackEnPassantColumn != None:
                enPassantCol = self.blackEnPassantColumn

        h ^= ChessBoardZobrist.en_passant[enPassantCol]
        
        return h

