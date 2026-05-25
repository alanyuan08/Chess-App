use crate::bishop_mask::*;
use crate::rook_mask::*;
use crate::move_command::*;
use arrayvec::ArrayVec;

pub fn queen_moves(mut queen_bitboard: u64, occupancy: u64, 
    opponent_pieces: u64, moves: &mut ArrayVec::<ForwardMove, 256>, mailbox: [BoardPiece; 64])  {

    while queen_bitboard != 0 {
        let queen = queen_bitboard.trailing_zeros() as usize;

        let rook_attack_paths = rook_attack_paths(queen, occupancy);
        let bishop_attack_paths = bishop_attack_paths(queen, occupancy);

        let mut queen_moves = (rook_attack_paths | bishop_attack_paths) & !occupancy;
        let mut queen_captures = (rook_attack_paths | bishop_attack_paths) & opponent_pieces;

        while queen_moves != 0 {
            let target = queen_moves.trailing_zeros() as usize;
            moves.push(ForwardMove { 
                start_sq: queen, end_sq: target, move_type: MoveFlag::MOVE, pv_score: 200 
            });
            queen_moves &= queen_moves - 1;
        }

        while queen_captures != 0 {
            let target = queen_captures.trailing_zeros() as usize;

            let captured_piece_val = piece_value(mailbox[target]);
            let pv_score = 100 - (captured_piece_val * 10) + 5; 

            moves.push(ForwardMove { 
                start_sq: queen, end_sq: target, move_type: MoveFlag::CAPTURE, pv_score 
            });
            queen_captures &= queen_captures - 1;
        }

        queen_bitboard &= queen_bitboard - 1;
    }
}