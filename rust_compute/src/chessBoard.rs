use pyo3::prelude::*;

#[pyfunction]
pub fn test_print(chessBoard: &str) {
    println!("ChessBoard Setup {}", chessBoard);
}