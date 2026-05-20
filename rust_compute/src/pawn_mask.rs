use crate::move_command::*;

const NOT_A_FILE: u64 = 0xFEFE_FEFE_FEFE_FEFE;
const NOT_H_FILE: u64 = 0x7F7F_7F7F_7F7F_7F7F;

const RANK_1: u64 = 0x0000_0000_0000_00FF;
const RANK_2: u64 = 0x0000_0000_0000_FF00;

const RANK_7: u64 = 0x00FF_0000_0000_0000;
const RANK_8: u64 = 0xFF00_0000_0000_0000;

// Return Squares that could be attacked / Used for King Safety
pub fn white_pawn_attacks(white_pawns: u64) -> u64 {
    // Left capture (North-West): shift 7, but only if not on A-file
    let captures_left = (white_pawns << 7) & NOT_H_FILE;
    
    // Right capture (North-East): shift 9, but only if not on H-file
    let captures_right = (white_pawns << 9) & NOT_A_FILE;
    
    captures_left | captures_right
}

pub fn black_pawn_attacks(black_pawns: u64) -> u64 {
    // Left capture (South-West): shift 7, but only if not on A-file
    let captures_left = (black_pawns >> 9) & NOT_H_FILE;
    
    // Right capture (South-East): shift 9, but only if not on H-file
    let captures_right = (black_pawns >> 7) & NOT_A_FILE;
    
    captures_left | captures_right
}

pub fn white_pawn_moves(white_pawns: u64, occupancy: u64, black_pieces: u64, 
    en_passant_board: u64, moves: &mut Vec<ForwardMove>)  {

    // --- Aggregating Single Pushes (No Promotion) ---
    let mut one_move = ((white_pawns & !RANK_7) << 8) & !occupancy;
    while one_move != 0 {
        let target = one_move.trailing_zeros() as usize;
        moves.push(
            ForwardMove { startSq: target - 8, endSq: target, moveType: MoveFlag::MOVE }
        );
        one_move &= one_move - 1;
    }

    // --- Aggregating Single Pushes (Promotion) ---
    let mut promotion_move = ((white_pawns & RANK_7) << 8) & !occupancy;
    while promotion_move != 0 {
        let target = promotion_move.trailing_zeros() as usize;
        for promotion_flag in PROMOTION_FLAGS.iter().copied() {
            moves.push(
                ForwardMove { startSq: target - 8, endSq: target, moveType: promotion_flag }
            );
        }
        promotion_move &= promotion_move - 1;
    }

    // --- Aggregating Double Pushes ---
    let single_move = ((white_pawns & RANK_2) << 8) & !occupancy;
    let mut double_move = (single_move << 8) & !occupancy;
    while double_move != 0 {
        let target = double_move.trailing_zeros() as usize;
        moves.push(
            ForwardMove { startSq: target - 8 * 2, endSq: target, moveType: MoveFlag::PAWNOPENMOVE }
        );
        double_move &= double_move - 1;
    }

    // --- Aggregating Capture (No Promotion) ---
    let left_captures = (white_pawns << 7) & NOT_H_FILE & black_pieces;
    let right_captures = (white_pawns << 9) & NOT_A_FILE & black_pieces;

    let mut left_capture_no_promotion = left_captures & !RANK_8;
    while left_capture_no_promotion != 0 {
        let target = left_capture_no_promotion.trailing_zeros() as usize;
        moves.push(
            ForwardMove { startSq: target - 7, endSq: target, moveType: MoveFlag::CAPTURE }
        );
        left_capture_no_promotion &= left_capture_no_promotion - 1;
    }

    let mut right_capture_no_promotion = right_captures & !RANK_8;
    while right_capture_no_promotion != 0 {
        let target = right_capture_no_promotion.trailing_zeros() as usize;
        moves.push(
            ForwardMove { startSq: target - 9, endSq: target, moveType: MoveFlag::CAPTURE }
        );
        right_capture_no_promotion &= right_capture_no_promotion - 1;
    }

    // --- Aggregating Capture (Promotion) ---
    let mut left_capture_promotion = left_captures & RANK_8;
    while left_capture_promotion != 0 {
        let target = left_capture_promotion.trailing_zeros() as usize;
        for promotion_flag in PROMOTION_FLAGS.iter().copied() {
            moves.push(
                ForwardMove { startSq: target - 8, endSq: target, moveType: promotion_flag }
            );
        }
        left_capture_promotion &= left_capture_promotion - 1;
    }

    let mut right_capture_promotion = right_captures & RANK_8;
    while right_capture_promotion != 0 {
        let target = right_capture_promotion.trailing_zeros() as usize;
        for promotion_flag in PROMOTION_FLAGS.iter().copied() {
            moves.push(
                ForwardMove { startSq: target - 8, endSq: target, moveType: promotion_flag }
            );
        }
        right_capture_promotion &= right_capture_promotion - 1;
    }

    // --- Aggregating En Passant
    let mut left_attackers = (en_passant_board >> 9) & NOT_H_FILE & white_pawns;
    while left_attackers != 0 {
        let from = left_attackers.trailing_zeros() as usize;
        moves.push(
            ForwardMove { startSq: from, endSq: from + 9, moveType: MoveFlag::ENPASSANT }
        );
        left_attackers &= left_attackers - 1;
    }

    let mut right_attackers = (en_passant_board >> 7) & NOT_A_FILE & white_pawns;
    while right_attackers != 0 {
        let from = right_attackers.trailing_zeros() as usize;
        moves.push(
            ForwardMove { startSq: from, endSq: from + 7, moveType: MoveFlag::ENPASSANT }
        );
        right_attackers &= right_attackers - 1;
    }
}

pub fn black_pawn_moves(black_pawns: u64, occupancy: u64, white_pieces: u64, 
    en_passant_board: u64, moves: &mut Vec<ForwardMove>)  {

    // --- Aggregating Single Pushes (No Promotion) ---
    let mut one_move = ((black_pawns & !RANK_2) >> 8) & !occupancy;
    while one_move != 0 {
        let target = one_move.trailing_zeros() as usize;
        moves.push(
            ForwardMove { startSq: target + 8, endSq: target, moveType: MoveFlag::MOVE }
        );
        one_move &= one_move - 1;
    }

    // --- Aggregating Single Pushes (Promotion) ---
    let mut promotion_move = ((black_pawns & RANK_2) >> 8) & !occupancy;
    while promotion_move != 0 {
        let target = promotion_move.trailing_zeros() as usize;
        for promotion_flag in PROMOTION_FLAGS.iter().copied() {
            moves.push(
                ForwardMove { startSq: target - 8, endSq: target, moveType: promotion_flag }
            );
        }
        promotion_move &= promotion_move - 1;
    }

    // --- Aggregating Double Pushes ---
    let single_move = ((black_pawns & RANK_7) >> 8) & !occupancy;
    let mut double_move = (single_move >> 8) & !occupancy;
    while double_move != 0 {
        let target = double_move.trailing_zeros() as usize;
        moves.push(
            ForwardMove { startSq: target + 8 * 2, endSq: target, moveType: MoveFlag::PAWNOPENMOVE }
        );
        double_move &= double_move - 1;
    }

    // --- Aggregating Capture (No Promotion) ---
    let left_captures = (black_pawns >> 9) & NOT_H_FILE & white_pieces;
    let right_captures = (black_pawns >> 7) & NOT_A_FILE & white_pieces;

    let mut left_capture_no_promotion = left_captures & !RANK_1;
    while left_capture_no_promotion != 0 {
        let target = left_capture_no_promotion.trailing_zeros() as usize;
        moves.push(
            ForwardMove { startSq: target + 9, endSq: target, moveType: MoveFlag::CAPTURE }
        );
        left_capture_no_promotion &= left_capture_no_promotion - 1;
    }

    let mut right_capture_no_promotion = right_captures & !RANK_1;
    while right_capture_no_promotion != 0 {
        let target = right_capture_no_promotion.trailing_zeros() as usize;
        moves.push(
            ForwardMove { startSq: target + 7, endSq: target, moveType: MoveFlag::CAPTURE }
        );
        right_capture_no_promotion &= right_capture_no_promotion - 1;
    }

    // --- Aggregating Capture (Promotion) ---
    let mut left_capture_promotion = left_captures & RANK_1;
    while left_capture_promotion != 0 {
        let target = left_capture_promotion.trailing_zeros() as usize;
        for promotion_flag in PROMOTION_FLAGS.iter().copied() {
            moves.push(
                ForwardMove { startSq: target - 8, endSq: target, moveType: promotion_flag }
            );
        }
        left_capture_promotion &= left_capture_promotion - 1;
    }

    let mut right_capture_promotion = right_captures & RANK_1;
    while right_capture_promotion != 0 {
        let target = right_capture_promotion.trailing_zeros() as usize;
        for promotion_flag in PROMOTION_FLAGS.iter().copied() {
            moves.push(
                ForwardMove { startSq: target - 8, endSq: target, moveType: promotion_flag }
            );
        }
        right_capture_promotion &= right_capture_promotion - 1;
    }

    // --- Aggregating En Passant
    let mut left_attackers = (en_passant_board << 7) & NOT_H_FILE & black_pawns;
    while left_attackers != 0 {
        let target = left_attackers.trailing_zeros() as usize;
        moves.push(
            ForwardMove { startSq: target, endSq: target - 7, moveType: MoveFlag::ENPASSANT }
        );
        left_attackers &= left_attackers - 1;
    }

    let mut right_attackers = (en_passant_board << 9) & NOT_A_FILE & black_pawns;
    while right_attackers != 0 {
        let target = right_attackers.trailing_zeros() as usize;
        moves.push(
            ForwardMove { startSq: target, endSq: target - 9, moveType: MoveFlag::ENPASSANT }
        );
        right_attackers &= right_attackers - 1;
    }
}