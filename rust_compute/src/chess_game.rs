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

            let (move_command, uci_move): (ForwardMove, String) = parse_move_uci(prev_move);

            // Store Value prior to Executing Move
            let prev_castle_rights = self.chess_board.castle_rights(); 
            let prev_en_passant = self.chess_board.en_passant();

            let remove_piece = self.chess_board.execute_move(move_command);
            let undo_move = UndoMove {
                startSq: move_command.startSq,
                endSq: move_command.endSq,
                moveType: move_command.moveType,
                capturedPiece: remove_piece,
                prevCastleRights: prev_castle_rights,
                prevEnPassant: prev_en_passant,
            };
            
            let hash = self.chess_board.zobrist_hash();
            let count = self.traversed_positions.entry(hash).or_insert(0);
            *count += 1;

            self.history[self.history_index] = Some(undo_move);
            self.history_index += 1;

            // Update Move
            self.chess_board.timecat_push_move(uci_move);
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
}

fn parse_move_uci(raw_move: &String) -> (ForwardMove, String) {
    let map = HashMap::from([
        (0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'),
        (4, 'e'), (5, 'f'), (6, 'g'), (7, 'h'),
    ]);
    
    let result: Vec<u32> = raw_move.chars().map(|c: char| {
        c.to_digit(10).unwrap_or(0)
    }).collect();

    let forward_move: ForwardMove = ForwardMove { 
        startSq: (result[1] * 8 + result[0]) as usize, 
        endSq: (result[3] * 8 + result[2]) as usize, 
        moveType: MoveFlag::try_from(result[4]).expect("Corrupted move data"),
    };

    let start_file = *map.get(&result[0]).unwrap_or(&'a'); 
    let start_rank = result[1] + 1;
    let end_file = *map.get(&result[2]).unwrap_or(&'a'); 
    let end_rank = result[3] + 1;

    let uci_string = format!("{}{}{}{}", start_file, start_rank, end_file, end_rank);

    // 3. Return both values as a tuple
    (forward_move, uci_string)

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
