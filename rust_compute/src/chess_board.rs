use pyo3::prelude::*;
use std::collections::HashMap;
use crate::bishop_mask::*;

#[derive(Copy, Clone)]
pub enum Side {
    WHITE = 0,
    BLACK = 1,
}

#[derive(Copy, Clone)]
pub enum Piece {
    NONE = 0,

    WPAWN = 1,
    WBISHOP = 2,
    WKNIGHT = 3,
    WROOK = 4,
    WQUEEN = 5,
    WKING = 6,

    BPAWN = 7,
    BBISHOP = 8,
    BKNIGHT = 9,
    BROOK = 10,
    BQUEEN = 11,
    BKING = 12,
}

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
    active_player: Side,
    total_moves: i32,

    mailbox: [Piece; 64],
}

fn print_board(board: u64) {
    println!("PRINT BOARD");
    for r in (0..8).rev() {
        for c in 0..8 {
            print!("{}", (board >> (r * 8 + c)) & 1);
        }
        println!("");
    }
}

impl ChessBoard {
    // A constructor-like associated function
    // [0] - White / [1] - Black
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
            active_player: Side::WHITE,
            total_moves: 0,
            mailbox: [Piece::NONE; 64],
        }
    }

    fn init_board(&mut self) {
        // Black is 0 / White is 1
        for color in 0..2 {
            let offset = if color == 1 { 0 } else { 48 };

            for i in 0..8 {
                self.pawns[color] |= 1u64 << (i + offset + if color == 1 { 8 } else { 0 });
                self.mailbox[i + offset] = if color == 1 { Piece::WPAWN } else { Piece::BPAWN };
            }

            let p_off = if color == 1 { 0 } else { 56 };
            self.rooks[color] |= (1u64 << (0 + p_off)) | (1u64 << (7 + p_off));
            self.knights[color] |= (1u64 << (1 + p_off)) | (1u64 << (6 + p_off));
            self.bishops[color] |= (1u64 << (2 + p_off)) | (1u64 << (5 + p_off));
            self.queens[color] |= 1u64 << (3 + p_off);
            self.kings[color] |= 1u64 << (4 + p_off);

            if color == 1 {
                self.mailbox[0 + offset] = Piece::WROOK;
                self.mailbox[1 + offset] = Piece::WKNIGHT;
                self.mailbox[2 + offset] = Piece::WBISHOP;
                self.mailbox[3 + offset] = Piece::WQUEEN;
                self.mailbox[4 + offset] = Piece::WKING;
                self.mailbox[5 + offset] = Piece::WBISHOP;
                self.mailbox[6 + offset] = Piece::WKNIGHT;
                self.mailbox[7 + offset] = Piece::WROOK;
            } else {
                self.mailbox[0 + offset] = Piece::BROOK;
                self.mailbox[1 + offset] = Piece::BKNIGHT;
                self.mailbox[2 + offset] = Piece::BBISHOP;
                self.mailbox[3 + offset] = Piece::BQUEEN;
                self.mailbox[4 + offset] = Piece::BKING;
                self.mailbox[5 + offset] = Piece::BBISHOP;
                self.mailbox[6 + offset] = Piece::BKNIGHT;
                self.mailbox[7 + offset] = Piece::BROOK;
            }

            // Update composite bitboards
            self.all_pieces[color] = self.pawns[color] | self.rooks[color] | 
                                        self.knights[color] | self.bishops[color] | 
                                        self.queens[color] | self.kings[color];
            self.occupied |= self.all_pieces[color];
        }
    }

    fn move_piece(&mut self, hci_move: &String) {
        let map = HashMap::from([
            ('a', 0), ('b', 1), ('c', 2), ('d', 3),
            ('e', 4), ('f', 5), ('g', 6), ('h', 7),
        ]);
        
        let result: Vec<u32> = hci_move.chars().map(|c: char| {
            if c.is_alphabetic() {
                *map.get(&c).unwrap_or(&0) 
            } else {
                c.to_digit(10).unwrap_or(0)
            }
        }).collect();

        println!("{:?}", result);
    }
}

#[pyfunction]
pub fn compute_next_move(prev_moves: Vec<String>) -> bool {
    let mut chess_board = ChessBoard::new();
    chess_board.init_board();

    let bishop_attack = bishop_move_paths(2 as usize, (chess_board.all_pieces[0] | chess_board.all_pieces[1]));
    print_board(bishop_attack & !(chess_board.all_pieces[1]));
    print_board(chess_board.all_pieces[1]);

    true
}