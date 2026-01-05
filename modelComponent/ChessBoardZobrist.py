import secrets

# Standard Piece Map: 6 pieces * 2 colours = 12 total
# We use indices 0-5 for White (P, N, B, R, Q, K) and 6-11 for Black
PIECE_TYPES = 12
SQUARES = 64

# Controller 
class ChessBoardZobrist():
    def __init__(self):
        # 1. Piece-Square table: [piece_type][square]
        self.table = [[secrets.randbits(64) for _ in range(SQUARES)] 
                      for _ in range(PIECE_TYPES)]
        
        # 2. Side to move (only need one for Black; if White, we don't XOR it)
        self.black_to_move = secrets.randbits(64)
        
        # 3. Castling Rights (16 possible states: 4 bits for KQkq)
        self.castling = [secrets.randbits(64) for _ in range(16)]
        
        # 4. En Passant (8 files where a capture can happen, +1 for "none")
        self.en_passant = [secrets.randbits(64) for _ in range(9)]