use pyo3::prelude::*;

use crate::chess_piece::ChessPiece;

#[derive(Debug, PartialEq)]
enum PlayerTurn {
    White,
    Black,
}

pub struct ChessBoard {
    width: u32,
    height: u32,
    player_turn: PlayerTurn,
    chess_board: [[Option<Box<dyn ChessPiece>>; 8]; 8]
}

const EMPTY: Option<Box<dyn ChessPiece>> = None;

impl ChessBoard {
    // A constructor-like associated function
    fn new(width: u32, height: u32) -> Self {
        Self { 
            width, 
            height, 
            player_turn: PlayerTurn::White,
            chess_board: [const { [const { None }; 8] }; 8]
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

    let chess_board_area = chess_board.area();
    println!("Area: {}", chess_board_area);
    
    if chess_board_area > 100 {
        true
    } else {
        false
    }
}