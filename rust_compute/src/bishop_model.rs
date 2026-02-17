use pyo3::prelude::*;

mod chess_piece;
use chess_piece::ChessPiece;

impl BishopModel for ChessPiece {
    fn phase_weight(&self) -> u32 {
        1
    }

    fn piece_value(&self) -> u32 {
    	300
    }
}