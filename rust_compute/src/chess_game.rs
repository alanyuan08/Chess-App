use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use std::time::Duration;
use std::collections::HashMap;
use pyo3::prelude::*;
use arrayvec::ArrayVec;

use crate::chess_board::*;
use crate::move_command::*;

use crate::bishop_mask::*;
use crate::rook_mask::*;

pub const PV_DEPTH: i32 = 8;
pub const MATE_VALUE: i32 = 3000000;

struct ChessGame {
    history: [Option<UndoMove>; 1024],
    history_index: usize,

    chess_board: ChessBoard,
    traversed_positions: HashMap<u64, i32>,

    nodes_processed: AtomicUsize,
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
            nodes_processed: AtomicUsize::new(0),
        }
    }

    fn process_moves(&mut self, prev_moves: Vec<String>) {
        for prev_move in &prev_moves {
            if self.history_index >= 1024 { break; } 

            let move_command: ForwardMove = parse_forward_move(prev_move);
            self.process_forward_move(move_command);
            self.process_time_cat_forward(move_command);
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
            start_sq: forward_move.start_sq,
            end_sq: forward_move.end_sq,
            move_type: forward_move.move_type,
            captured_piece: remove_piece,
            prev_castle_rights: prev_castle_rights,
            prev_en_passant: prev_en_passant,
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

    // Search Entry Point
    pub fn root_search(&mut self) -> Option<ForwardMove> {
        // Search Time
        let start_time = Instant::now();
        let time_limit = Duration::from_secs(20);

        let mut best_move_overall: Option<ForwardMove> = None;

        // Iterative Deepening
        for depth in 1..=PV_DEPTH {
            if start_time.elapsed() >= time_limit {
                break;
            }
            
            let result = self.negamax(depth, 0, i32::MIN + 1, i32::MAX - 1, best_move_overall);
            best_move_overall = result.best_move.clone();
        }

        let elapsed_time = start_time.elapsed();
        let node_procesed =  self.nodes_processed.load(Ordering::Relaxed);
        println!("{} Nodes Procesed in {} milliseconds", node_procesed, elapsed_time.as_millis());

        best_move_overall
    }
 
    // Process Negamax
    fn negamax(&mut self, depth: i32, ply: i32, mut alpha: i32, beta: i32, 
        pv_move_hint: Option<ForwardMove>) -> SearchResult {
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

        let mut gen_moves = ArrayVec::<ForwardMove, 256>::new();
        self.chess_board.generate_moves(&mut gen_moves, pv_move_hint);

        for forward_move in &gen_moves {
            // Push move (handles UCI, board state, hash, and history internally)
            self.process_forward_move(*forward_move);

            // Psuedo legal move exposes check, undo move
            if self.chess_board.is_previous_player_king_in_check() {    
                self.process_backward_move();
                continue;
            }

            // Move is Legal, Forward Move Time Cat
            legal_moves_played += 1;
            self.process_time_cat_forward(*forward_move);

            // Recursive Negamax Call
            let negamax_result = self.negamax(depth - 1, ply + 1, -beta, -alpha, None);
            let score = -negamax_result.score;

            // Undo Move + TimeCat
            self.chess_board.timecat_pop_move();
            self.process_backward_move();

            // Track maximum evaluations
            if score > max_score {
                max_score = score;
                best_move = Some(*forward_move);
            }

            if score > alpha {
                alpha = score;
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
                    score: -MATE_VALUE + ply, 
                    best_move: None 
                };
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

    fn board_eval(&mut self) -> i32 {
        let static_eval = self.chess_board.eval();
        self.nodes_processed.fetch_add(1, Ordering::Relaxed);
        static_eval
    }

    // Quiescence Search 
    fn quiescence_search(&mut self, mut alpha: i32, beta: i32, depth: i32) -> i32 {
        let mut static_eval = self.board_eval();
        if self.chess_board.active_player() == Side::BLACK {
            static_eval = -static_eval;
        }

        // Three Move Repetition Draw
        if self.check_three_move_repetition() {
            return 0;
        }

        if depth > 50 {
            return static_eval;
        }

        // Beta cutoff
        if static_eval >= beta {
            return beta;
        }

        // Update alpha (standing pat)
        if static_eval > alpha {
            alpha = static_eval;
        }
        
        // Generate strictly legal tactical moves directly onto the global stack
        let mut gen_moves = ArrayVec::<ForwardMove, 256>::new();
        self.filter_psuedo_legal_quiescence_moves(&mut gen_moves);

        // Quiscence Search
        for forward_move in &gen_moves {
            // Push move (handles UCI, board state, hash, and history internally)
            self.process_forward_move(*forward_move);
            
            // Psuedo legal move exposes check, undo move
            if self.chess_board.is_previous_player_king_in_check() {
                self.process_backward_move();
                continue;
            }

            // Move is Legal, Forward Move Time Cat
            self.process_time_cat_forward(*forward_move);

            // Negamax search call
            let score = -self.quiescence_search(-beta, -alpha, depth + 1);
            
            // Undo Move + TimeCat
            self.chess_board.timecat_pop_move();
            self.process_backward_move();

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

    // Filter all Capture & Promotion Moves
    fn filter_psuedo_legal_quiescence_moves(&mut self, 
        gen_moves: &mut ArrayVec::<ForwardMove, 256>
    ) {
        self.chess_board.generate_moves(gen_moves, None);

        gen_moves.retain(|cmd| {
            matches!(
                cmd.move_type,
                MoveFlag::PROMOTIONQUEEN | MoveFlag::PROMOTIONROOK |
                MoveFlag::PROMOTIONBISHOP | MoveFlag::PROMOTIONKNIGHT | 
                MoveFlag::CAPTURE | MoveFlag::ENPASSANT
            )
        });
    }

}

fn parse_uci(forward_move: ForwardMove) -> String {
    let map = HashMap::from([
        (0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'),
        (4, 'e'), (5, 'f'), (6, 'g'), (7, 'h'),
    ]);

    let start_file = *map.get(&(forward_move.start_sq % 8)).unwrap_or(&'a'); 
    let start_rank = (forward_move.start_sq / 8) + 1;
    let end_file = *map.get(&(forward_move.end_sq % 8)).unwrap_or(&'a'); 
    let end_rank = (forward_move.end_sq / 8) + 1;

    let promo = match forward_move.move_type {
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
        start_sq: (result[1] * 8 + result[0]) as usize, 
        end_sq: (result[3] * 8 + result[2]) as usize, 
        move_type: MoveFlag::try_from(result[4]).expect("Corrupted move data"),
        pv_score: 0,
    }
}

#[pyfunction]
pub fn compute_next_move<'py>(py: Python<'py>, prev_moves: Vec<String>) -> PyResult<Bound<'py, PyAny>> {
    let mut chess_game = ChessGame::new();
    chess_game.process_moves(prev_moves);

    // chess_game.chess_board.timecat_print_fen();
    // println!("{:?}", chess_game.all_pseudo_legal_moves());
    let best_move = chess_game.root_search();
    
    let module = py.import("modelComponent.moveCommand")?;
    let move_command_class = module.getattr("MoveCommand")?;

    let enum_mod = py.import("appEnums")?;
    let move_command_type_enum = enum_mod.getattr("MoveCommandType")?;

    if let Some(mv) = best_move {
        let move_type_value = mv.move_type as i32;
        let enum_instance = move_command_type_enum.call1((move_type_value,))?;

        let args = (
            (mv.start_sq / 8) as i32,
            (mv.start_sq % 8) as i32,
            (mv.end_sq / 8) as i32,
            (mv.end_sq % 8) as i32,
            enum_instance,
        ).into_pyobject(py)?;

        let move_command_instance = move_command_class.call1(args)?;
        Ok(move_command_instance)
    } else {
        Ok(py.None().into_bound(py))
    }
}

#[pyfunction]
pub fn init_attack_tables() {
    let _ = *BISHOP_ATTACKS;
    let _ = *ROOK_ATTACKS;
}
