use pyo3::prelude::*;

mod chess_piece;
use chess_piece::ChessPiece;

impl RookModel for ChessPiece {
    fn phase_weight(&self) -> u32 {
        2
    }

    fn piece_value(&self) -> u32 {
    	500
    }
}