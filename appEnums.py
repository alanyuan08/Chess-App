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
    MOVE = 1
    QUEENSIDECASTLE = 2
    KINGSIDECASTLE = 3
    CAPTURE = 4
    ENPASSANT = 5
    PAWNOPENMOVE = 6
    PROMOTE = 7
 
class TTBoundType(Enum):
 	EXACT = 1
 	UPPERBOUND = 2
 	LOWERBOUND = 3