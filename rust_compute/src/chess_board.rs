use pyo3::prelude::*;
use std::collections::HashMap;
use crate::bishop_mask::*;

// 0 -> White / 1 -> Black
#[derive(Clone, Copy)]
struct ChessBoard {
    pawns: [u64; 2],
    knights: [u64; 2],
    bishops: [u64; 2],
    rooks: [u64; 2],
    queens: [u64; 2],
    kings: [u64; 2],
    
    all_pieces: [u64; 2],
    occupied: u64,

    castling_rights: u8,
    en_passant: i8,
    active_player: u8,
    total_moves: i32,
}


impl ChessBoard {
    // A constructor-like associated function
    fn new() -> Self {
        Self {
            pawns: [0, 0],
            knights: [0, 0],
            bishops: [0, 0],
            rooks: [0, 0],
            queens: [0, 0],
            kings: [0, 0],
            all_pieces: [0, 0],
            occupied: 0,
            castling_rights: 0,
            en_passant: -1,
            active_player: 0,
            total_moves: 0,
        }
    }
    fn move_piece(&mut self, move: &String) {
        let map = HashMap::from([
            ('a', 0), ('b', 1), ('c', 2), ("d", 3),
            ("e", 4), ("f", 5), ("g", 6), ("h", 7),
        ]);
        
        let result: Vec<u32> = input.chars().map(|c| {
            if c.is_alphabetic() {
                // Fetch from map, default to 0 if not found
                *map.get(&c).unwrap_or(&0) 
            } else {
                // Convert digit char to its numeric value (e.g., '3' -> 3)
                c.to_digit(10).unwrap_or(0)
            }
        }).collect();

        println!("{:?}", result);
    }
}

#[pyfunction]
pub fn compute_next_move(prev_moves: Vec<String>) -> bool {
    let chess_board = ChessBoard::new();

    true
}