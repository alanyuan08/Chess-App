use crate::move_command::*;
use crate::chess_board::*;
use arrayvec::ArrayVec;

// Compute Knight Attack on Compile
pub const KNIGHT_ATTACKS: [u64; 64] = {
    let mut knight_attack = [0u64; 64];

    let mut i = 0;
    while i < 64 {
        let r = (i / 8) as i8;
        let f = (i % 8) as i8;

        // The 8 directions for the knight
        let directions: [(i8, i8); 8] = 
        [(-2, -1), (-1, -2), (1, 2), (2, 1), (1, -2), (2, -1), (-1, 2), (-2, 1)];

        let mut d = 0;
        while d < 8 {
            let (dr, df) = directions[d];
            let (cur_r, cur_f) = (r + dr, f + df);
            
            if cur_r >= 0 && cur_r <= 7 && cur_f >= 0 && cur_f <= 7 {
                knight_attack[i] |= 1u64 << (cur_r * 8 + cur_f);
            }
            d += 1;
        }
        i += 1;
    }

    knight_attack
};

pub fn knight_moves(chess_board: &mut ChessBoard, player_index: usize, opp_index: usize,
    moves: &mut ArrayVec::<ForwardMove, 256>)  {

    let mut knight_bitboard = chess_board.knights[player_index];

    while knight_bitboard != 0 {
        let knight = knight_bitboard.trailing_zeros() as usize;
        let knight_attack_paths = KNIGHT_ATTACKS[knight];

        let mut knight_moves = knight_attack_paths & !chess_board.occupied;
        let mut knight_captures = knight_attack_paths & chess_board.all_pieces[opp_index];

        while knight_moves != 0 {
            let target = knight_moves.trailing_zeros() as usize;
            moves.push(
                ForwardMove { 
                    start_sq: knight, end_sq: target, move_type: MoveFlag::MOVE, pv_score: 1000
                }
            );
            knight_moves &= knight_moves - 1;
        }

        while knight_captures != 0 {
            let target = knight_captures.trailing_zeros() as usize;

            let captured_piece_val = piece_value(chess_board.mailbox[target]);
            let pv_score = 100 - (captured_piece_val * 10) + 2; 

            moves.push(
                ForwardMove { 
                    start_sq: knight, end_sq: target, move_type: MoveFlag::CAPTURE, pv_score 
                }
            );
            knight_captures &= knight_captures - 1;
        }

        knight_bitboard &= knight_bitboard - 1;
    }
}