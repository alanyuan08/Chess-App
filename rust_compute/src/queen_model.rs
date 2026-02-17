use pyo3::prelude::*;

mod chess_piece;
use chess_piece::ChessPiece;

impl QueenModel for ChessPiece {
    fn phase_weight(&self) -> u32 {
        4
    }

    fn piece_value(&self) -> u32 {
    	900
    }
}