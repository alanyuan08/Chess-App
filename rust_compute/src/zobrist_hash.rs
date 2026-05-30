use std::sync::LazyLock;
use crate::move_command::*;
use rand::prelude::*;

// Standard Piece Map: 6 pieces * 2 colours = 13 total
// We use indices 1-6 for White (P, N, B, R, Q, K) and 7-12 for Black
const PIECE_TYPES: usize = 13;
const SQUARES: usize = 64;
const CASTLING: usize = 16;

// 0-7 = Files A-H, 8 = No En Passant
const EN_PASSANT_STATES: usize = 9;

pub static ZOBRIST_TABLE_MAP: LazyLock<Box<[[u64; SQUARES]; PIECE_TYPES]>> = LazyLock::new(|| {
    let mut table = Box::new([[0u64; SQUARES]; PIECE_TYPES]);
    let mut rng = rand::rng();

    for j in 1..PIECE_TYPES {
        for i in 0..SQUARES {
            table[j][i] = rng.random();
        }
    }
    table
});

pub static ZOBRIST_EN_PASSANT: LazyLock<Box<[u64; EN_PASSANT_FILES]>> = LazyLock::new(|| {
    let mut en_passant = Box::new([0u64; EN_PASSANT_FILES]);
    let mut rng = rand::rng();

    for i in 0..EN_PASSANT_STATES{
        en_passant[i] = rng.random();
    }
    en_passant
});

pub static ZOBRIST_CASTLING: LazyLock<Box<[u64; CASTLING]>> = LazyLock::new(|| {
    let mut castling = Box::new([0u64; CASTLING]);
    let mut rng = rand::rng();

    for i in 0..CASTLING {
        castling[i] = rng.random();
    }
    castling
});

pub static ZOBRIST_SIDE_TO_MOVE: LazyLock<Box<[u64; 2]>> = LazyLock::new(|| {
    Box::new([rand::random(), rand::random()])
});

// Convert Piece Type / Player to hash
pub fn active_player_zobrist(active_player: Side) -> usize {
    active_player as usize
}

// Player Index
pub fn piece_type_zobrist(piece_type: BoardPiece) -> usize {
    piece_type as usize
}

// Convert En Passant Square to hash
pub fn en_passant_zobrist(en_passant: u64) -> usize {
    if en_passant == 0 {
        return 8;
    }

    let square_index = en_passant.trailing_zeros() as usize;
    (square_index % 8)
}
