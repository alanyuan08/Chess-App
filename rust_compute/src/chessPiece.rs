use pyo3::prelude::*;

mod chessBoard;

#[derive(Debug, PartialEq)]
enum Color {
    White,
    Black,
}

struct ChessPiece {
    row: u32,
    col: u32,
    color: Color
}

trait ChessPiece {
    fn phase_weight(&self) -> u32;

    fn piece_value(&self) -> u32;

    fn computed_value(&self, chessBoard: &chessBoard, phaseWeight: u32) -> u32:
}