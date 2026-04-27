use pyo3::prelude::*;
use crate::bishop_mask::*;

// 0 -> White / 1 -> Black
#[derive(Clone, Copy)]
struct ChessBoard {
    pawns: [u64; 2],
    knights: [u64; 2],
    bishops: [u64; 2],
    rooks: [u64; 2],
    queens: [u64; 2],
    kings: [u64; 2],
    
    all_pieces: [u64; 2],
    occupied: u64,

    castling_rights: u8,
    en_passant: i8,
    active_player: u8,
    total_moves: i32,
}


impl ChessBoard {
    // A constructor-like associated function
    fn new() -> Self {
        Self {
            pawns: [0, 0],
            knights: [0, 0],
            bishops: [0, 0],
            rooks: [0, 0],
            queens: [0, 0],
            kings: [0, 0],
            all_pieces: [0, 0],
            occupied: 0,
            castling_rights: 0,
            en_passant: -1,
            active_player: 0,
            total_moves: 0,
        }
    }
}

#[pyfunction]
pub fn compute_next_move(prev_moves: Vec<String>) -> bool {
    let chess_board = ChessBoard::new();

    for prev_move in &prev_moves {
        println!("{}", prev_move);
    }

    true
}