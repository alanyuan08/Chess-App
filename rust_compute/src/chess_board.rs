use pyo3::prelude::*;
use std::collections::HashMap;
use crate::bishop_mask::*;
use crate::king_mask::*;
use crate::knight_mask::*;
use crate::pawn_mask::*;
use crate::rook_mask::*;
use crate::move_command::*;

// 0 -> White / 1 -> Black
#[derive(Clone, Copy, PartialEq, Eq)]
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
    en_passant: u64,
    active_player: Side,
    total_moves: i32,

    mailbox: [Piece; 64],

    history: [Option<UndoMove>; 1024],
    history_index: usize,
}

const WHITE_KINGSIDE: u8 = 0b0001; // 1
const WHITE_QUEENSIDE: u8 = 0b0010; // 2
const BLACK_KINGSIDE: u8 = 0b0100; // 4
const BLACK_QUEENSIDE: u8 = 0b1000; // 8


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
            castling_rights: 0b1111,
            en_passant: 0,
            active_player: Side::WHITE,
            total_moves: 0,
            mailbox: [Piece::NONE; 64],
            history: [None; 1024],
            history_index: 0,
        }
    }

    fn init_board(&mut self) {
        for color in 0..2 {
            // Ranks: White = 0 & 1, Black = 6 & 7
            let piece_rank_offset = if color == 0 { 0 } else { 56 };
            let pawn_rank_offset  = if color == 0 { 8 } else { 48 };

            // 1. Initialize Pawns
            for i in 0..8 {
                let sq = pawn_rank_offset + i;
                self.pawns[color] |= 1u64 << sq;
                self.mailbox[sq] = if color == 0 { Piece::WPAWN } else { Piece::BPAWN };
            }

            // 2. Initialize Major Pieces (Bitboards)
            self.rooks[color]   |= (1u64 << (piece_rank_offset + 0)) | (1u64 << (piece_rank_offset + 7));
            self.knights[color] |= (1u64 << (piece_rank_offset + 1)) | (1u64 << (piece_rank_offset + 6));
            self.bishops[color] |= (1u64 << (piece_rank_offset + 2)) | (1u64 << (piece_rank_offset + 5));
            self.queens[color]  |= 1u64 << (piece_rank_offset + 3);
            self.kings[color]   |= 1u64 << (piece_rank_offset + 4);

            // 3. Initialize Major Pieces (Mailbox)
            let pieces = if color == 0 {
                [Piece::WROOK, Piece::WKNIGHT, Piece::WBISHOP, Piece::WQUEEN, Piece::WKING, Piece::WBISHOP, Piece::WKNIGHT, Piece::WROOK]
            } else {
                [Piece::BROOK, Piece::BKNIGHT, Piece::BBISHOP, Piece::BQUEEN, Piece::BKING, Piece::BBISHOP, Piece::BKNIGHT, Piece::BROOK]
            };

            for i in 0..8 {
                self.mailbox[piece_rank_offset + i] = pieces[i];
            }

            // 4. Composite Bitboards
            self.all_pieces[color] = self.pawns[color] | self.rooks[color] | 
                                    self.knights[color] | self.bishops[color] | 
                                    self.queens[color] | self.kings[color];
            self.occupied |= self.all_pieces[color];
        }
    }

    fn opponent_attack_targets(&mut self) -> u64 {
        let mut attacks = 0u64;
        let opp = if self.active_player == Side::WHITE { 1 } else { 0 };
        let occ = self.occupied;

        // 1. Pawns - Print Opponent Attack Pawns
        let mut pawns = self.pawns[opp];
        if self.active_player == Side::WHITE {
            while pawns != 0 {
                attacks |= BLACK_PAWN_ATTACKS[pawns.trailing_zeros() as usize];
                pawns &= pawns - 1;
            }
        } else {
            while pawns != 0 {
                attacks |= WHITE_PAWN_ATTACKS[pawns.trailing_zeros() as usize];
                pawns &= pawns - 1;
            }
        }

        // 2. Knights
        let mut knights = self.knights[opp];
        while knights != 0 {
            attacks |= KNIGHT_ATTACKS[knights.trailing_zeros() as usize];
            knights &= knights - 1;
        }

        // 3. Kings
        let mut kings = self.kings[opp];
        while kings != 0 {
            attacks |= KING_ATTACKS[kings.trailing_zeros() as usize];
            kings &= kings - 1;
        }

        // 4. Sliders (Bishops, Rooks, Queens)
        let mut bishops = self.bishops[opp] | self.queens[opp];
        while bishops != 0 {
            attacks |= bishop_move_paths(bishops.trailing_zeros() as usize, occ);
            bishops &= bishops - 1;
        }

        let mut rooks = self.rooks[opp] | self.queens[opp];
        while rooks != 0 {
            attacks |= rook_attack_paths(rooks.trailing_zeros() as usize, occ);
            rooks &= rooks - 1;
        }

        attacks
    }

    fn generate_moves(&mut self) -> Vec<Move> {
        let mut generate_moves = Vec::new();
        let mut player_index = if self.active_player == Side::WHITE { 0 } else { 1 }; 

        let pawn_position = get_lsb_indices(self.pawns[player_index]);
        println!("{:?}", pawn_position);

        let rook_position = get_lsb_indices(self.rooks[player_index]);
        println!("{:?}", rook_position);

        let knight_positon = get_lsb_indices(self.knights[player_index]);
        println!("{:?}", knight_positon);

        let bishop_position = get_lsb_indices(self.bishops[player_index]);
        println!("{:?}", bishop_position);

        let queen_position = get_lsb_indices(self.queens[player_index]);
        println!("{:?}", queen_position);

        let king_positon = get_lsb_indices(self.kings[player_index]);
        println!("{:?}", king_positon);

        let opponent_attack_targets = self.opponent_attack_targets();
        print_board(opponent_attack_targets);

        generate_moves
    }

    fn move_piece(&mut self, uci_move: &String) {
        let map = HashMap::from([
            ('a', 0), ('b', 1), ('c', 2), ('d', 3),
            ('e', 4), ('f', 5), ('g', 6), ('h', 7),
        ]);
        
        let result: Vec<u32> = uci_move.chars().map(|c: char| {
            if c.is_alphabetic() {
                *map.get(&c).unwrap_or(&0) 
            } else {
                c.to_digit(10).unwrap_or(0)
            }
        }).collect();

        println!("{:?}", result);
    }
}

fn get_lsb_indices(board: u64) -> Vec<usize> {
    let mut bitboard = board;
    let mut indices = Vec::new();
    
    while bitboard != 0 {
        let lsb_index = bitboard.trailing_zeros();
        indices.push(lsb_index as usize);
        
        bitboard &= bitboard - 1;
    }
    
    indices
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

#[pyfunction]
pub fn compute_next_move(prev_moves: Vec<String>) -> bool {
    let mut chess_board = ChessBoard::new();
    chess_board.init_board();

    chess_board.generate_moves();

    true
}