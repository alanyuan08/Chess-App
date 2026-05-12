use std::collections::HashMap;
use pyo3::prelude::*;

use crate::chess_board::*;
use crate::move_command::*;

use crate::bishop_mask::*;
use crate::rook_mask::*;

#[derive(Clone, Copy, PartialEq, Eq)]
struct ChessGame {
    history: [Option<UndoMove>; 1024],
    history_index: usize,

    chess_board: ChessBoard,
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
            }

        }
    }

    fn process_moves(&mut self, prev_moves: Vec<String>) {
        for prev_move in &prev_moves {
            let move_command: Move = parse_move(prev_move);
            let remove_piece = self.chess_board.execute_move(move_command);

            let undo_move = UndoMove {
                startSq: move_command.startSq,
                endSq: move_command.endSq,
                moveType: move_command.moveType,
                capturedPiece: remove_piece,
                prevCastleRights: self.chess_board.castle_rights(),
                prevEnPassant: self.chess_board.en_passant(),
            };

            self.history_index += 1;
            self.history[self.history_index] = Some(undo_move);
        }
    }

    fn undo_moves(&mut self) {
        for i in (0..self.history_index).rev() {
            let undo_command_history = self.history[i];

            if let Some(undo_command) = undo_command_history {
                self.chess_board.unexecute_move(undo_command);

                self.history_index -= 1;
            }
        }
    }
}

#[pyfunction]
pub fn compute_next_move(prev_moves: Vec<String>) {
    let mut chess_game = ChessGame::new();

    chess_game.process_moves(prev_moves);
    chess_game.chess_board.generate_moves();

    chess_game.undo_moves();
}

#[pyfunction]
pub fn init_attack_tables() {
    let _ = *BISHOP_ATTACKS;
    let _ = *ROOK_ATTACKS;
}

fn parse_move(uci_move: &String) -> Move {
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

    Move { 
        startSq: (result[1] * 8 + result[0]) as usize, 
        endSq: (result[3] * 8 + result[2]) as usize, 
        moveType: MoveFlag::try_from(result[4]).expect("Corrupted move data"),
    }
}
