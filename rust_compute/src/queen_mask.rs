use crate::bishop_mask::*;
use crate::rook_mask::*;

pub fn queen_moves(active_queen: Vec<usize>, occupancy: u64, 
    opponent_pieces: u64, moves: &mut Vec<Move>)  {

    for &queen in &active_queen {
        let rook_attack_paths = rook_attack_paths(queen, occupancy);
        let bishop_attack_paths = bishop_attack_paths(queen, occupancy);

        let mut queen_moves = (rook_attack_paths | bishop_attack_paths) & !occupancy;
        let mut queen_captures = (rook_attack_paths | bishop_attack_paths) & opponent_pieces;

        while queen_moves != 0 {
            let target = queen_moves.trailing_zeros() as usize;
            moves.push(Move { startSq: queen, endSq: target, moveType: MoveFlag::MOVE });
            queen_moves &= queen_moves - 1;
        }

        while queen_captures != 0 {
            let target = queen_captures.trailing_zeros() as usize;
            moves.push(Move { startSq: queen, endSq: target, moveType: MoveFlag::CAPTURE });
            queen_captures &= queen_captures - 1;
        }
    }
}