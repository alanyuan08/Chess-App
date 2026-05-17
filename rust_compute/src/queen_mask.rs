use crate::bishop_mask::*;
use crate::rook_mask::*;
use crate::move_command::*;

pub fn queen_moves(mut queen_bitboard: u64, occupancy: u64, 
    opponent_pieces: u64, moves: &mut Vec<ForwardMove>)  {

    while queen_bitboard != 0 {
        let queen = queen_bitboard.trailing_zeros() as usize;

        let rook_attack_paths = rook_attack_paths(queen, occupancy);
        let bishop_attack_paths = bishop_attack_paths(queen, occupancy);

        let mut queen_moves = (rook_attack_paths | bishop_attack_paths) & !occupancy;
        let mut queen_captures = (rook_attack_paths | bishop_attack_paths) & opponent_pieces;

        while queen_moves != 0 {
            let target = queen_moves.trailing_zeros() as usize;
            moves.push(ForwardMove { startSq: queen, endSq: target, moveType: MoveFlag::MOVE });
            queen_moves &= queen_moves - 1;
        }

        while queen_captures != 0 {
            let target = queen_captures.trailing_zeros() as usize;
            moves.push(ForwardMove { startSq: queen, endSq: target, moveType: MoveFlag::CAPTURE });
            queen_captures &= queen_captures - 1;
        }

        queen_bitboard &= queen_bitboard - 1;
    }
}