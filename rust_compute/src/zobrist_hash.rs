use crate::move_command::*;
use crate::chess_board::*;

// Standard Piece Map: 6 pieces * 2 colours = 13 total
// We use indices 1-6 for White (P, N, B, R, Q, K) and 7-12 for Black
const PIECE_TYPES = 13
const SQUARES = 64

// 0-8 map to the files / 9 indiciates no En Passant
const EN_PASSANT_FILES = 9;

struct ZobristHashTable {
    // [piece_type][square]
    table: [[u64; SQUARES]; PIECE_TYPES], 

    side_to_move: u64,
    castling: [u64; 16],
    en_passant: [u64; 8],
}

// Generates the Random Numbers for Zobrist hash
impl ZobristHashTable {
    fn new() -> Self {
        Self {
            table: {
                let mut table = [[0u64; SQUARES]; PIECE_TYPES];
                let mut rng = rand::rng();
                for j in 1..PIECE_TYPES {
                    for i in 0..SQUARES {
                        table[i][j] = rng.random();
                    }
                }
                table
            },
            side_to_move: [0, rand::random()],
            castling: {
                let mut castling = [0u64; 16];
                let mut rng = rand::rng();
                for i in 0..16 {
                        castling[i] = rng.random();
                }
                castling
            },
            en_passant: {
                let mut en_passant = [0u64; EN_PASSANT_FILES];
                let mut rng = rand::rng();
                for i in 0..8 {
                        en_passant[i] = rng.random();
                }
                en_passant
            }
        }
    }

    fn compute_init_value(self, chess_board: ChessBoard) -> u64 { 
        let mut hash = 0u64;

        // If piece_idx is 'Empty', this XORs with 0, which does nothing.
        // If it's a real piece, it XORs the random value.
        for index in 0..64 {
            let piece_idx = piece_player(chess_board.mailbox[index]) as usize;
            hash ^= self.table[piece_idx][index];
        }

        // Side to Move
        hash ^= self.side_to_move[active_player(self.active_player)];

        // Castling
        hash ^= self.castling[self.castling_rights as u64];

        // En Passant
        hash ^= self.en_passant[get_en_passant_file(self.en_passant)];

        hash
    }
}

// Convert Piece Type / Player to hash
fn active_player(active_player: Side) -> usize {
    (active_player as u8) as usize;
}

// Player Index
fn piece_player(piece_type: Piece) -> usize {
    (piece_type as u8) as usize;
}

// Convert En Passant Square to hash
fn get_en_passant_file(en_passant: u64) -> usize {
    if en_passant == 0 {
        return 0;
    }

    let square_index = (en_passant.trailing_zeros() as u8) + 1;
    (square_index % 8) as usize;
}

