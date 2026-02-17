use pyo3::prelude::*;

struct Chessboard {
    width: u32,
    height: u32,
}

impl Chessboard {
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
pub fn new_chessBoard(width: u32, height: u32) -> bool {
    let chessBoard = Chessboard::new(width, height);

    let chessBoardArea = chessBoard.area();
    println!("Area: {}", chessBoardArea);
    
    if chessBoardArea > 100 {
        true
    } else {
        false
    }
}