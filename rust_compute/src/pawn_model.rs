use pyo3::prelude::*;

mod chess_piece;
use chess_piece::ChessPiece;

impl PawnModel for ChessPiece {
    fn phase_weight(&self) -> u32 {
        0
    }

    fn piece_value(&self) -> u32 {
    	100
    }
}