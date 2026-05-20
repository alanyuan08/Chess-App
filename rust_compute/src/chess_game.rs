use std::collections::HashMap;
use pyo3::prelude::*;

use crate::chess_board::*;
use crate::move_command::*;

use crate::bishop_mask::*;
use crate::rook_mask::*;

pub const DEPTH: i32 = 6;
pub const MATE_VALUE: i32 = 3000000;

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
        // Store Value prior to Executing Move
        let prev_castle_rights = self.chess_board.castle_rights(); 
        let prev_en_passant = self.chess_board.en_passant();

        let hash = self.chess_board.zobrist_hash();
        let count = self.traversed_positions.entry(hash).or_insert(0);
        *count += 1;

        let remove_piece = self.chess_board.execute_move(forward_move);
        let undo_move = UndoMove {
            startSq: forward_move.startSq,
            endSq: forward_move.endSq,
            moveType: forward_move.moveType,
            capturedPiece: remove_piece,
            prevCastleRights: prev_castle_rights,
            prevEnPassant: prev_en_passant,
        };

        self.history[self.history_index] = Some(undo_move);
        self.history_index += 1;
    }

    fn process_time_cat_forward(&mut self, forward_move: ForwardMove) {
        let uci_command: String = parse_uci(forward_move);
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

            self.chess_board.unexecute_move(undo_move);
        }
    }

    pub fn check_three_move_repetition(&self) -> bool {
        let hash = self.chess_board.zobrist_hash();
        
        // Check if the current position has been reached 3 times
        self.traversed_positions.get(&hash).copied().unwrap_or(0) == 3
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

    // Root Entrypoint
    pub fn root_search(&mut self, depth: i32) -> Option<ForwardMove> {
        let result = self.negamax(depth, i32::MIN + 1, i32::MAX - 1);
        result.best_move
    }
 
    // Process Negamax
    fn negamax(&mut self, depth: i32, mut alpha: i32, beta: i32) -> SearchResult {
        // Three Move Repetition Draw
        if self.check_three_move_repetition() {
            return SearchResult {
                score: 0,
                best_move: None,
            };
        }

        // Leaf Node Condition -> Drop into Quiescence Search
        if depth == 0 {
            return SearchResult {
                score: self.quiescence_search(alpha, beta, 0),
                best_move: None,
            }
        }

        let mut best_move = None;
        let mut max_score = i32::MIN;
        let mut legal_moves_played = 0;

        for forward_move in self.all_pseudo_legal_moves() {
            // Push move (handles UCI, board state, hash, and history internally)
            self.process_forward_move(forward_move);

            // Psuedo legal move exposes check, undo move
            if self.chess_board.is_previous_player_king_in_check() {
                self.process_backward_move();
                continue;
            }

            // Move is Legal, Forward Move Time Cat
            legal_moves_played += 1;
            self.process_time_cat_forward(forward_move);

            // Recursive Negamax Call
            let negamax_result = self.negamax(depth - 1, -beta, -alpha);
            let score = negamax_result.score;
            let result = -score;

            // Undo Move + TimeCat
            self.process_backward_move();
            self.chess_board.timecat_pop_move();

            // Track maximum evaluations
            if score > max_score {
                max_score = score;
                best_move = Some(forward_move);
            }

            if max_score > alpha {
                alpha = max_score;
            }

            // Alpha-Beta Pruning Cutoff
            if alpha >= beta {
                break;
            }
        }

        // 4. Handle terminal nodes cleanly if no legal moves exist
        if legal_moves_played == 0 {
            if self.chess_board.is_in_check() {
                // Checkmate
                return SearchResult { 
                    score: -MATE_VALUE + depth, 
                    best_move: None };
            } else {
                // Stalemate
                return SearchResult { 
                    score: 0, 
                    best_move: None 
                };
            }
        }

        SearchResult { score: max_score, best_move }
    }

    // Quiescence Search 
    fn quiescence_search(&mut self, mut alpha: i32, beta: i32, depth: i32) -> i32 {
        let static_eval = self.chess_board.eval();

        if depth > 50 {
            return static_eval;
        }

        // Three Move Repetition Draw
        if self.check_three_move_repetition() {
            return 0;
        }

        // Beta cutoff
        if static_eval >= beta {
            return beta;
        }

        // Update alpha (standing pat)
        if static_eval > alpha {
            alpha = static_eval;
        }

        // Filter and iterate through quiescence moves
        for forward_move in self.all_psuedo_legal_quiescence_moves() {
            // Push move (handles UCI, board state, hash, and history internally)
            self.process_forward_move(forward_move);
            
            // Psuedo legal move exposes check, undo move
            if self.chess_board.is_previous_player_king_in_check() {
                self.process_backward_move();
                continue;
            }

            // Move is Legal, Forward Move Time Cat
            self.process_time_cat_forward(forward_move);

            // Negamax search call
            let score = -self.quiescence_search(-beta, -alpha, depth + 1);
            
            // Undo Move + TimeCat
            self.process_backward_move();
            self.chess_board.timecat_pop_move();

            // Alpha-Beta pruning checks
            if score >= beta {
                return beta; 
            }
            if score > alpha {
                alpha = score;
            }
        }

        alpha
    }

    // Sort used for Principal Variation Search
    fn get_move_priority(&self, cmd: &ForwardMove) -> i32 {
        match cmd.moveType {
            MoveFlag::PROMOTIONQUEEN | MoveFlag::PROMOTIONROOK |
            MoveFlag::PROMOTIONBISHOP | MoveFlag::PROMOTIONKNIGHT => 90000,
            MoveFlag::CAPTURE => {
                let captured_piece_val=  self.chess_board.index_piece_value(cmd.endSq);
                let starting_piece_val = self.chess_board.index_piece_value(cmd.startSq);

                (captured_piece_val * 10) - starting_piece_val
            },
            MoveFlag::ENPASSANT => 100, 
            MoveFlag::KINGSIDECASTLE | MoveFlag::QUEENSIDECASTLE => 50,
            _ => 0,
        }
    }

    // Filter all Capture & Promotion Moves
    fn all_psuedo_legal_quiescence_moves(&mut self) -> Vec<ForwardMove> {
        let pseudo_legal_moves = self.all_pseudo_legal_moves();
        
        pseudo_legal_moves
            .into_iter()
            .filter(|cmd| {
                matches!(
                    cmd.moveType,
                    MoveFlag::PROMOTIONQUEEN | MoveFlag::PROMOTIONROOK |
                    MoveFlag::PROMOTIONBISHOP | MoveFlag::PROMOTIONKNIGHT | 
                    MoveFlag::CAPTURE | MoveFlag::ENPASSANT
                )
            })
            .collect()
    }

    // The move does not confirm if it introduces a discovered check
    fn all_pseudo_legal_moves(&mut self) -> Vec<ForwardMove> {
        let mut valid_moves = self.chess_board.generate_moves();
        valid_moves.sort_by_cached_key(|mov| -self.get_move_priority(mov));

        valid_moves
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
        MoveFlag::PROMOTIONQUEEN => "q",
        MoveFlag::PROMOTIONROOK => "r",
        MoveFlag::PROMOTIONBISHOP => "b",
        MoveFlag::PROMOTIONKNIGHT => "n",
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
    println!("{:?}", chess_game.root_search(DEPTH));
}

#[pyfunction]
pub fn init_attack_tables() {
    let _ = *BISHOP_ATTACKS;
    let _ = *ROOK_ATTACKS;
}
