use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use std::time::{Instant, Duration};
use std::sync::Arc;
use crate::transposition_table::*;
use std::collections::HashMap;
use std::cmp;

use crate::move_command::*;
use crate::chess_game::*;

use crate::parser::*;
use crate::chess_board::*;
use arrayvec::ArrayVec;

pub struct SearchWorker {
    nodes_processed: Arc<AtomicUsize>,
    transposition_table: Arc<TranspositionTable>,

    history: [Option<UndoMove>; 1024],
    history_index: usize,

    chess_board: ChessBoard,
    traversed_positions: HashMap<u64, i32>
}

impl SearchWorker {
    pub fn new(
        nodes_processed: Arc<AtomicUsize>,
        transposition_table: Arc<TranspositionTable>,
    ) -> Self {
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
            nodes_processed,
            transposition_table
        }
    }

    // Search Entry Point
    pub fn root_search(&mut self, thread_id: i32, 
        max_depth: i32, stop_signal: &AtomicBool) -> Option<ForwardMove> {
        // Start the timer
        let start_time = Instant::now();
        let time_limit = Duration::from_secs(60);

        // Thread_id = 0 is the main thread, the rest are helper threads
        let start_depth = if thread_id == 0 { 
            1 
        } else { 
            2 + (thread_id % 4)
        };

        // --- ITERATIVE DEEPENING LOOP ---
        let mut best_move_overall: Option<ForwardMove> = None;
        for depth in start_depth..=max_depth {
            if stop_signal.load(Ordering::Relaxed) || start_time.elapsed() >= time_limit {
                if thread_id == 0 {
                    stop_signal.store(true, Ordering::Relaxed);
                }
                break;
            }
            
            let result = self.negamax(depth, 0, -INFINITY, INFINITY, 
                best_move_overall, stop_signal);
            
            if !stop_signal.load(Ordering::Relaxed){
                best_move_overall = result.best_move;

                if thread_id == 0 {
                    println!("[Master Thread] Completed Depth: {} | Best Move Score: {:?}", depth, result.best_move);
                    if depth >= PV_DEPTH { 
                        stop_signal.store(true, Ordering::Relaxed);
                        break;
                    }
                }
            } else {
                break;
            }
        }
            
        best_move_overall
    }

    pub fn process_moves(&mut self, prev_moves: Vec<String>) {
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

    fn process_time_cat_forward(&mut self, forward_move: ForwardMove) {
        let uci_command: String = parse_uci(forward_move);
        self.chess_board.timecat_push_move(uci_command);
    }

    fn process_time_cat_backward(&mut self) {
        self.chess_board.timecat_pop_move();
    }

    // Check if the current position has been reached 3 times
    fn check_three_move_repetition(&self) -> bool {
        let hash = self.chess_board.zobrist_hash();
        self.traversed_positions.get(&hash).copied().unwrap_or(0) == 3
    }

    // Process Negamax
    fn negamax(&mut self, depth: i32, ply: i32, mut alpha: i32, mut beta: i32, 
        mut pv_move_hint: Option<ForwardMove>, stop_signal: &AtomicBool) -> SearchResult {
        
        if stop_signal.load(Ordering::Relaxed) {
            return SearchResult { score: 0, best_move: None };
        }

        // Original Alpha for Transposition Table
        let original_alpha = alpha;
        
        // Three Move Repetition Draw
        if self.check_three_move_repetition() {
            return SearchResult {
                score: 0,
                best_move: None,
            };
        }

        let hash = self.chess_board.zobrist_hash();
        if let Some(tt_entry) = self.transposition_table.probe(hash, ply) {
            let retrieved_score: i32 = tt_entry.score as i32;
            let retrieved_depth: i32 = tt_entry.depth as i32;
            
            if tt_entry.move_id != 0 {
                let mut mv = ForwardMove::unpack(tt_entry.move_id);
                mv.pv_score = -2_000_000;
                pv_move_hint = Some(mv);
            };

            if retrieved_depth >= depth {

                // EXACT: The true minimax value was found; return it immediately.
                if tt_entry.flag == HashFlag::EXACT {
                    return SearchResult {
                        score: retrieved_score,
                        best_move: pv_move_hint,
                    };
                }
                    
                // LOWER BOUND: The true score is AT LEAST this high. 
                else if tt_entry.flag == HashFlag::LOWERBOUND {
                    alpha = cmp::max(alpha, retrieved_score);
                }
                // UPPER BOUND: The true score is AT MOST this high.
                else if tt_entry.flag == HashFlag::UPPERBOUND {
                    beta = cmp::min(beta, retrieved_score);
                }  

                // If the bounds adjusted alpha/beta enough to cause a cutoff, return early
                if alpha >= beta {
                    return SearchResult {
                        score: retrieved_score,
                        best_move: pv_move_hint,
                    };
                }
            }
        }

        // Leaf Node Condition -> Drop into Quiescence Search
        if depth == 0 {
            return SearchResult {
                score: self.quiescence_search(alpha, beta, ply, -1),
                best_move: None,
            };
        }

        let mut best_move = None;
        let mut legal_moves_played = 0;
        let mut best_score = -INFINITY;

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
            let negamax_result = self.negamax(depth - 1, ply + 1, -beta, -alpha, None, stop_signal);
            let score = -negamax_result.score;

            // Undo Move + TimeCat
            self.process_time_cat_backward();
            self.process_backward_move();

            // Track maximum evaluations
            if score > best_score {
                best_score = score;
                best_move = Some(*forward_move);
            }

            // Alpha-Beta Cutoff
            if best_score >= beta {
                // Adjust mate scores to absolute bounds before saving
                self.transposition_table.store(
                    hash, best_score, ply, best_move, depth, HashFlag::LOWERBOUND
                );

                return SearchResult { score: best_score, best_move };
            }

            if score > alpha {
                alpha = score;
            }
        }

        // 4. Handle terminal nodes cleanly if no legal moves exist
        if legal_moves_played == 0 {
            if self.chess_board.is_in_check() {
                let mate_score = -MATE_VALUE + ply;

                self.transposition_table.store(
                    hash, mate_score, ply, None, MAX_DEPTH, HashFlag::EXACT
                );

                // Checkmate
                return SearchResult { 
                    score: mate_score, 
                    best_move: None 
                };
            } else {
                self.transposition_table.store(
                    hash, 0, ply, None, MAX_DEPTH, HashFlag::EXACT
                );

                // Stalemate
                return SearchResult { 
                    score: 0, 
                    best_move: None 
                };
            }
        }

        let flag = if best_score > original_alpha {
            HashFlag::EXACT
        } else {
            HashFlag::UPPERBOUND
        };

        // Adjust mate scores to absolute bounds before saving final loop results
        self.transposition_table.store(hash, best_score, ply, best_move, depth, flag);
        SearchResult { score: best_score, best_move }
    }

    fn board_eval(&mut self) -> i32 {
        let static_eval = self.chess_board.eval();
        self.nodes_processed.fetch_add(1, Ordering::Relaxed);
        static_eval
    }

    // Quiescence Search 
    fn quiescence_search(&mut self, mut alpha: i32, mut beta: i32, ply: i32, depth: i32) -> i32 {
        // Three Move Repetition Draw
        if self.check_three_move_repetition() {
            return 0;
        }

        let mut pv_move_hint = None;

        let hash = self.chess_board.zobrist_hash();
        if let Some(tt_entry) = self.transposition_table.probe(hash, ply) {
            let retrieved_score: i32 = tt_entry.score as i32;
            let retrieved_depth: i32 = tt_entry.depth as i32;
            
            if tt_entry.move_id != 0 {
                let mut mv = ForwardMove::unpack(tt_entry.move_id);
                mv.pv_score = -2_000_000;
                pv_move_hint = Some(mv);
            };

            if retrieved_depth >= depth {

                // EXACT: The true minimax value was found; return it immediately.
                if tt_entry.flag == HashFlag::EXACT {
                    return retrieved_score;
                }
                    
                // LOWER BOUND: The true score is AT LEAST this high. 
                else if tt_entry.flag == HashFlag::LOWERBOUND {
                    alpha = cmp::max(alpha, retrieved_score);
                }
                // UPPER BOUND: The true score is AT MOST this high.
                else if tt_entry.flag == HashFlag::UPPERBOUND {
                    beta = cmp::min(beta, retrieved_score);
                }  

                // If the bounds adjusted alpha/beta enough to cause a cutoff, return early
                if alpha >= beta {
                    return retrieved_score;
                }
            }
        }

        let mut static_eval = self.board_eval();
        if self.chess_board.active_player() == Side::BLACK {
            static_eval = -static_eval;
        }

        if ply > MAX_DEPTH {
            return static_eval;
        }

        let king_in_check = self.chess_board.is_in_check(); 
        let mut best_score = if king_in_check { -INFINITY } else { static_eval };

        if !king_in_check {
            // Only allow standing pat if your king is perfectly safe
            if best_score >= beta {
                return best_score;
            }
            if best_score > alpha {
                alpha = best_score;
            }
        }

        let mut legal_moves_played = 0;
        let mut best_move = None;
        let mut hash_flag = HashFlag::UPPERBOUND;

        // Generate strictly legal tactical moves directly onto the global stack
        let mut gen_moves = ArrayVec::<ForwardMove, 256>::new();

        if king_in_check {
            // King IS in check: Generate all Moves
            self.chess_board.generate_moves(&mut gen_moves, pv_move_hint);
        } else {
            // King is NOT in check: Only generate captures, promotions, etc.
            self.filter_psuedo_legal_quiescence_moves(&mut gen_moves);
        }

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
            legal_moves_played += 1;
            self.process_time_cat_forward(*forward_move);

            // Negamax search call
            let score = -self.quiescence_search(-beta, -alpha, ply + 1, depth - 1);
            
            // Undo Move + TimeCat
            self.process_time_cat_backward();
            self.process_backward_move();

            // Fail-soft updates
            if score > best_score {
                best_score = score;
                best_move = Some(*forward_move);

                if score > alpha {
                    alpha = score;
                    hash_flag = HashFlag::EXACT;

                    if score >= beta { 
                        hash_flag = HashFlag::LOWERBOUND;
                        break;
                    }
                }
            }
        }

        if legal_moves_played == 0 && king_in_check {
            return -MATE_VALUE + ply;
        }

        // Store evaluation state inside the Transposition Table
        self.transposition_table.store(hash, best_score, ply, best_move, depth, hash_flag);
        best_score
    }   

    // Filter all Capture & Promotion Moves
    fn filter_psuedo_legal_quiescence_moves(&mut self, 
        gen_moves: &mut ArrayVec::<ForwardMove, 256>
    ) {
        self.chess_board.generate_moves(gen_moves, None);
        gen_moves.retain(|cmd| {
            matches!(
                cmd.move_type,
                MoveFlag::PROMOTIONQUEEN | 
                MoveFlag::CAPTURE | MoveFlag::ENPASSANT
            )
        });
    }
}