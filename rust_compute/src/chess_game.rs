use std::collections::HashMap;
use pyo3::prelude::*;

use crate::chess_board::*;
use crate::move_command::*;

use crate::bishop_mask::*;
use crate::rook_mask::*;

#[derive(Clone)]
struct ChessGame {
    history: [Option<UndoMove>; 1024],
    history_index: usize,

    chess_board: ChessBoard,
    traversed_positions: HashMap<u64, i32>,
}

impl ChessGame {
    fn new() -> Self {
        Self {
            history: [None; 1024],
            history_index: 0,

            chess_board: {
                let mut chess_board = ChessBoard::new();
                chess_board.init_board();
                chess_board
            },
            traversed_positions: {
                HashMap::new()
            },
        }
    }

    fn process_moves(&mut self, prev_moves: Vec<String>) {
        for prev_move in &prev_moves {
            if self.history_index >= 1024 { break; } 

            let move_command: ForwardMove = parse_forward_move(prev_move);
            self.process_forward_move(move_command);
        }
    }

    fn process_forward_move(&mut self, forward_move: ForwardMove) {
        let uci_command: String = parse_uci(forward_move);

        // Store Value prior to Executing Move
        let prev_castle_rights = self.chess_board.castle_rights(); 
        let prev_en_passant = self.chess_board.en_passant();

        let remove_piece = self.chess_board.execute_move(forward_move);
        let undo_move = UndoMove {
            startSq: forward_move.startSq,
            endSq: forward_move.endSq,
            moveType: forward_move.moveType,
            capturedPiece: remove_piece,
            prevCastleRights: prev_castle_rights,
            prevEnPassant: prev_en_passant,
        };
        
        let hash = self.chess_board.zobrist_hash();
        let count = self.traversed_positions.entry(hash).or_insert(0);
        *count += 1;

        self.history[self.history_index] = Some(undo_move);
        self.history_index += 1;

        // Update Time Cat Move
        self.chess_board.timecat_push_move(uci_command);
    }

    fn process_backward_move(&mut self) {
        if self.history_index == 0 {
            return;
        }

        self.history_index -= 1;
        if let Some(undo_move) = self.history[self.history_index].take() {
            let hash = self.chess_board.zobrist_hash();
            if let Some(count) = self.traversed_positions.get_mut(&hash) {
                if *count > 1 {
                    *count -= 1;
                } else {
                    self.traversed_positions.remove(&hash);
                }
            }

            self.chess_board.timecat_pop_move();
            self.chess_board.unexecute_move(undo_move);
        }
    }

    // DEBUG
    fn undo_moves(&mut self) {
        while self.history_index > 0 {
            self.history_index -= 1;

            if let Some(undo_command) = self.history[self.history_index].take() {
                self.chess_board.unexecute_move(undo_command);
            }
        }
    }

    // Process Next Best Move
    fn get_move_priority(&self, cmd: &ForwardMove) -> i32 {
        match cmd.moveType {
            // 1. Promotions (High Priority)
            MoveFlag::PROMOTION => 90000,

            // 2. Captures (MVV-LVA)
            MoveFlag::CAPTURE => {
                let captured_piece_val=  self.chess_board.index_piece_value(cmd.endSq);
                let starting_piece_val = self.chess_board.index_piece_value(cmd.startSq);

                (captured_piece_val * 10) - starting_piece_val
            },

            MoveFlag::ENPASSANT => 100, 

            // 3. Castling (Mid Priority)
            MoveFlag::KINGSIDECASTLE | MoveFlag::QUEENSIDECASTLE => 50,

            // 4. Standard Moves (Low Priority)
            _ => 0,
        }
    }
}

fn parse_uci(forward_move: ForwardMove) -> String {
    let map = HashMap::from([
        (0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'),
        (4, 'e'), (5, 'f'), (6, 'g'), (7, 'h'),
    ]);

    let start_file = *map.get(&(forward_move.startSq % 8)).unwrap_or(&'a'); 
    let start_rank = (forward_move.startSq / 8) + 1;
    let end_file = *map.get(&(forward_move.endSq % 8)).unwrap_or(&'a'); 
    let end_rank = (forward_move.endSq / 8) + 1;

    let promo = match forward_move.moveType {
        MoveFlag::PROMOTION => "q",
        _ => "",
    };
    format!("{}{}{}{}{}", start_file, start_rank, end_file, end_rank, promo)
}

fn parse_forward_move(raw_move: &String) -> ForwardMove {    
    let result: Vec<u32> = raw_move.chars().map(|c: char| {
        c.to_digit(10).unwrap_or(0)
    }).collect();

    ForwardMove { 
        startSq: (result[1] * 8 + result[0]) as usize, 
        endSq: (result[3] * 8 + result[2]) as usize, 
        moveType: MoveFlag::try_from(result[4]).expect("Corrupted move data"),
    }
}

#[pyfunction]
pub fn compute_next_move(prev_moves: Vec<String>) {
    let mut chess_game = ChessGame::new();

    chess_game.process_moves(prev_moves);
    chess_game.chess_board.generate_moves();

    // Print
    println!("{}", chess_game.chess_board.eval());
}

#[pyfunction]
pub fn init_attack_tables() {
    let _ = *BISHOP_ATTACKS;
    let _ = *ROOK_ATTACKS;
}
