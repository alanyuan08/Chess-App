use crate::move_command::*;
use std::collections::HashMap;
use crate::chess_board::ChessBoard;

pub fn parse_uci(forward_move: ForwardMove) -> String {
    let map = HashMap::from([
        (0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'),
        (4, 'e'), (5, 'f'), (6, 'g'), (7, 'h'),
    ]);

    let start_file = *map.get(&(forward_move.start_sq % 8)).unwrap_or(&'a'); 
    let start_rank = (forward_move.start_sq / 8) + 1;
    let end_file = *map.get(&(forward_move.end_sq % 8)).unwrap_or(&'a'); 
    let end_rank = (forward_move.end_sq / 8) + 1;

    let promo = match forward_move.move_type {
        MoveFlag::PROMOTIONQUEEN => "q",
        MoveFlag::PROMOTIONROOK => "r",
        MoveFlag::PROMOTIONBISHOP => "b",
        MoveFlag::PROMOTIONKNIGHT => "n",
        _ => "",
    };
    format!("{}{}{}{}{}", start_file, start_rank, end_file, end_rank, promo)
}

pub fn parse_forward_move_with_board(uci: &str, board: &ChessBoard) -> ForwardMove {
    fn sq_from_uci(file: char, rank: char) -> usize {
        let file_idx = (file as u8 - b'a') as usize;
        let rank_idx = (rank as u8 - b'1') as usize;
        rank_idx * 8 + file_idx
    }

    let chars: Vec<char> = uci.chars().collect();

    let start_sq = sq_from_uci(chars[0], chars[1]);
    let end_sq   = sq_from_uci(chars[2], chars[3]);

    let start_piece = board.mailbox_piece(start_sq);
    let end_piece   = board.mailbox_piece(end_sq);

    let mut move_type = MoveFlag::MOVE;

    // --- Promotion ---
    if chars.len() == 5 {
        move_type = match chars[4] {
            'q' => MoveFlag::PROMOTIONQUEEN,
            'r' => MoveFlag::PROMOTIONROOK,
            'b' => MoveFlag::PROMOTIONBISHOP,
            'n' => MoveFlag::PROMOTIONKNIGHT,
            _   => MoveFlag::MOVE,
        };
    }

    // --- Castling ---
    if is_king(start_piece) {
        let start_file = start_sq % 8;
        let end_file   = end_sq % 8;

        if start_file == 4 && end_file == 6 {
            move_type = MoveFlag::KINGSIDECASTLE;
        }
        if start_file == 4 && end_file == 2 {
            move_type = MoveFlag::QUEENSIDECASTLE;
        }
    }

    // --- Capture ---
    if is_some(end_piece) {
        move_type = MoveFlag::CAPTURE;
    }

    // --- En Passant ---
    if is_pawn(start_piece) {
        let start_rank = start_sq / 8;
        let end_rank   = end_sq / 8;
        let start_file = start_sq % 8;
        let end_file   = end_sq % 8;

        // Pawn moved diagonally but target square is empty → en passant
        if start_file != end_file && is_none(end_piece) {
            move_type = MoveFlag::ENPASSANT;
        }

        // Double pawn push
        if (start_rank as i32 - end_rank as i32).abs() == 2 {
            move_type = MoveFlag::PAWNOPENMOVE;
        }
}

    ForwardMove {
        start_sq,
        end_sq,
        move_type,
        pv_score: 0,
    }
}
