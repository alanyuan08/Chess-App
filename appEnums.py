from enum import Enum

class PieceType(Enum):
	KING = 1
	QUEEN = 2
	KNIGHT = 3
	ROOK = 4
	PAWN = 5
	BISHOP = 6

class Player(Enum):
	WHITE = 1
	BLACK = 2

class GameState(Enum):
	PLAYING = 1
	BLACKWIN = 2
	WHITEWIN = 3
	DRAW = 4

class MoveCommandType(Enum):
    MOVE = 0
    PAWNOPENMOVE = 1
    QUEENSIDECASTLE = 2
    KINGSIDECASTLE = 3
    ENPASSANT = 4
    CAPTURE = 5
    NULL = 6

    PROMOTION_QUEEN = 7
    PROMOTION_ROOK = 8
    PROMOTION_BISHOP = 9
    PROMOTION_KNIGHT = 10
 
class TTBoundType(Enum):
    EXACT = 1
    UPPERBOUND = 2
    LOWERBOUND = 3