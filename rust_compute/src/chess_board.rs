use pyo3::prelude::*;
use crate::bishop_mask::*;

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

    /*
    for mask in &BISHOP_MASKS {
        let mut r = 7;
        while r >= 0 {
            let mut c = 0;
            while c < 8 {
                print!("{}", ((mask >> r*8 + c) & 1));
                c += 1;
            }
            r -= 1;
            println!("");
        }
        println!("");
    }
    */

    for shift in &BISHOP_OFFSETS {
        println!("{}", shift);
    }
    
    let chess_board_area = chess_board.area();
    
    if chess_board_area > 100 {
        true
    } else {
        false
    }
}