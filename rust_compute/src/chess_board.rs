use pyo3::prelude::*;
use crate::bishop_mask::BISHOP_MASKS;

#[derive(Debug, PartialEq)]
enum PlayerTurn {
    White,
    Black,
}

pub struct ChessBoard {
    width: u32,
    height: u32,
    player_turn: PlayerTurn,
}

impl ChessBoard {
    // A constructor-like associated function
    fn new(width: u32, height: u32) -> Self {
        Self { 
            width, 
            height, 
            player_turn: PlayerTurn::White,
        }
    }

    // A read-only method
    fn area(&self) -> u32 {
        self.width * self.height
    }
}

#[pyfunction]
pub fn new_chess_board(width: u32, height: u32) -> bool {
    let chess_board = ChessBoard::new(width, height);

    let masks = BISHOP_MASKS;

    println!("{:?}", masks);
    
    let chess_board_area = chess_board.area();
    println!("Area: {}", chess_board_area);
    
    if chess_board_area > 100 {
        true
    } else {
        false
    }
}