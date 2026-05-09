use crate::move_command::*;
use crate::chess_board::*;

// White: Rank 1 (Indices 0-7)
// e1=4, f1=5, g1=6
const WHITE_KINGSIDE_PATH: u64 = 0x70; 

// e1=4, d1=3, c1=2
const WHITE_QUEENSIDE_PATH: u64 = 0x1C; 

// Black: Rank 8 (Indices 56-63)
// e8=60, f8=61, g8=62
const BLACK_KINGSIDE_PATH: u64 = 0x7000000000000000;

// e8=60, d8=59, c8=58
const BLACK_QUEENSIDE_PATH: u64 = 0x1C00000000000000;

 // Square e1 (bit 4)
const WHITE_KING_START: u64 = 0x10;

// Square e8 (bit 60)
const BLACK_KING_START: u64 = 0x1000000000000000;

// Check for Blocks
const WHITE_KINGSIDE_MOVE_PATH: u64 = WHITE_KINGSIDE_PATH & !WHITE_KING_START;
const WHITE_QUEENSIDE_MOVE_PATH: u64 = WHITE_QUEENSIDE_PATH & !WHITE_KING_START;

const BLACK_KINGSIDE_MOVE_PATH: u64 = BLACK_KINGSIDE_PATH & !BLACK_KING_START;
const BLACK_QUEENSIDE_MOVE_PATH: u64 = BLACK_QUEENSIDE_PATH & !BLACK_KING_START;

// Compute King Attack on Compile
pub const KING_ATTACKS: [u64; 64] = {
    let mut king_attack = [0u64; 64];

    let mut i = 0;
    while i < 64 {
        let r = (i / 8) as i8;
        let f = (i % 8) as i8;

        // The 8 directions for the king
        let directions: [(i8, i8); 8] = 
        [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)];

        let mut d = 0;
        while d < 8 {
            let (dr, df) = directions[d];
            let (cur_r, cur_f) = (r + dr, f + df);
            
            if cur_r >= 0 && cur_r <= 7 && cur_f >= 0 && cur_f <= 7 {
                king_attack[i] |= 1 << (cur_r * 8 + cur_f);
            }
            d += 1;
        }
        i += 1;
    }

    king_attack
};

pub fn king_moves(active_king: Vec<usize>, occupancy: u64, 
    opponent_attacks: u64, active_player: Side, castling_rights: u8,
    opponent_pieces: u64, moves: &mut Vec<Move>)  {

    // Check if King Capture / Move goes into Check
    for &king in &active_king {
        let king_attack_paths = KING_ATTACKS[king as usize];
        let king_safe_squares = king_attack_paths & !opponent_attacks;

        let mut king_moves = king_safe_squares & !occupancy;
        let mut king_captures = king_safe_squares & opponent_pieces;

        while king_moves != 0 {
            let target = king_moves.trailing_zeros() as usize;
            moves.push(Move { startSq: king, endSq: target, moveType: MoveFlag::MOVE });
            king_moves &= king_moves - 1;
        }

        while king_captures != 0 {
            let target = king_captures.trailing_zeros() as usize;
            moves.push(Move { startSq: king, endSq: target, moveType: MoveFlag::CAPTURE });
            king_captures &= king_captures - 1;
        }

        // Check for Queen / King Side Castle
        match active_player {
            Side::WHITE => {
                if (castling_rights & WHITE_KINGSIDE) != 0 {
                    if (WHITE_KINGSIDE_PATH & opponent_attacks) == 0 {
                        if (WHITE_KINGSIDE_MOVE_PATH & occupancy) == 0 {
                            moves.push(Move { startSq: king, endSq: king + 2, moveType: MoveFlag::KINGSIDECASTLE });
                        }
                    }
                }

                if (castling_rights & WHITE_QUEENSIDE) != 0 {
                    if (WHITE_QUEENSIDE_PATH & opponent_attacks) == 0 {   
                        if (WHITE_QUEENSIDE_MOVE_PATH & occupancy) == 0 {
                            moves.push(Move { startSq: king, endSq: king - 2, moveType: MoveFlag::QUEENSIDECASTLE });
                        }
                    }
                }
            },
            Side::BLACK => {
                if (castling_rights & BLACK_KINGSIDE) != 0 {
                    if (BLACK_KINGSIDE_PATH & opponent_attacks) == 0 {
                        if (BLACK_KINGSIDE_MOVE_PATH & occupancy) == 0 {
                            moves.push(Move { startSq: king, endSq: king + 2, moveType: MoveFlag::KINGSIDECASTLE });
                        }
                    }
                }

                if (castling_rights & BLACK_QUEENSIDE) != 0 {
                    if (BLACK_QUEENSIDE_PATH & opponent_attacks) == 0 {   
                        if (BLACK_QUEENSIDE_MOVE_PATH & occupancy) == 0 {
                            moves.push(Move { startSq: king, endSq: king - 2, moveType: MoveFlag::QUEENSIDECASTLE });
                        }
                    }
                }
            }
        }
    }
}