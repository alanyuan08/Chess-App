use pyo3::prelude::*;

pub struct ChessBoard {
    width: u32,
    height: u32,
}

impl ChessBoard {
    // A constructor-like associated function
    fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    // A read-only method
    fn area(&self) -> u32 {
        self.width * self.height
    }
}

#[pyfunction]
pub fn new_chess_board(width: u32, height: u32) -> bool {
    let chess_board = ChessBoard::new(width, height);

    let chess_board_area = chess_board.area();
    println!("Area: {}", chess_board_area);
    
    if chess_board_area > 100 {
        true
    } else {
        false
    }
}