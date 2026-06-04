use crate::bishop_mask::*;
use crate::rook_mask::*;
use crate::move_command::*;
use crate::chess_board::*;
use arrayvec::ArrayVec;

pub fn queen_moves(chess_board: &mut ChessBoard, player_index: usize, opp_index: usize,
    moves: &mut ArrayVec::<ForwardMove, 256>)  {

    let mut queen_bitboard = chess_board.queens[player_index];

    while queen_bitboard != 0 {
        let queen = queen_bitboard.trailing_zeros() as usize;

        let rook_attack_paths = rook_attack_paths(queen, chess_board.occupied);
        let bishop_attack_paths = bishop_attack_paths(queen, chess_board.occupied);
        let total_attacks = rook_attack_paths | bishop_attack_paths;

        let mut queen_moves = total_attacks & !chess_board.occupied;
        let mut queen_captures = total_attacks & chess_board.all_pieces[opp_index];

        while queen_moves != 0 {
            let target = queen_moves.trailing_zeros() as usize;
            moves.push(ForwardMove { 
                start_sq: queen, end_sq: target, move_type: MoveFlag::MOVE, pv_score: 1000 
            });
            queen_moves &= queen_moves - 1;
        }

        while queen_captures != 0 {
            let target = queen_captures.trailing_zeros() as usize;

            let captured_piece_val = piece_value(chess_board.mailbox[target]);
            let pv_score = 100 - (captured_piece_val * 10) + 5; 

            moves.push(ForwardMove { 
                start_sq: queen, end_sq: target, move_type: MoveFlag::CAPTURE, pv_score 
            });
            queen_captures &= queen_captures - 1;
        }

        queen_bitboard &= queen_bitboard - 1;
    }
}