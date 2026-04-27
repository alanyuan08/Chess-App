use pyo3::prelude::*;
use crate::bishop_mask::*;

/*

// 0 -> White / 1 -> Black
struct ChessBoard {
    pawns: [u64; 2],
    knights: [u64; 2],
    bishops: [u64; 2],
    rooks: [u64; 2],
    queens: [u64; 2],
    kings: [u64; 2],
    
    all_pieces: [u64; 2],
    occupied: u64,
    player_turn: PlayerTurn,

    castling_rights: u8,
    en_passant: i8,
    active_player: u8,
}

impl ChessBoard {
    // A constructor-like associated function
    fn new(fen_notation: String) -> Self {
        Self { 
        }
    }
}

*/

#[pyfunction]
pub fn compute_next_move(fen_notation: String) -> bool {
    // let chess_board = ChessBoard::new(fen_notation);
    
    println!("{}", fen_notation);
    true
}