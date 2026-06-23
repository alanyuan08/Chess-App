use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Instant, Duration};
use crate::transposition_table::*;
use std::cmp;

use crate::move_command::*;
use crate::chess_game::*;
use crate::lmr_table::*;

use crate::parser::*;
use crate::chess_board::*;
use arrayvec::ArrayVec;

#[derive(Clone)] 
pub struct SearchWorker<'a>  {
    transposition_table: &'a TranspositionTable,
    nodes_processed: usize,

    history: [Option<UndoMove>; 1024],
    history_index: usize,

    chess_board: ChessBoard,

    traversed_positions: [u64; 1024],
    position_stack_len: usize,

    killer_move_table: [[Option<ForwardMove>; MAX_DEPTH as usize]; 2],
    thread_id: i32
}

impl<'a> SearchWorker<'a> {
    pub fn new(
        transposition_table: &'a TranspositionTable, 
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
                [0; 1024]
            },
            position_stack_len: 0, 
            
            nodes_processed: 0,
            transposition_table,

            killer_move_table: [[None; MAX_DEPTH as usize]; 2],
            thread_id: 0,
        }
    }

    pub fn from_game_state(
        transposition_table: &'a TranspositionTable, 
        search_worker: &SearchWorker,
        thread_id: i32
    ) -> Self {
        Self {
            // History move records are left blank as they aren't needed for future searches
            history: [None; 1024],
            history_index: search_worker.history_index,

            // INSTANTLY copy the fully evaluated board configuration (Blazing fast bitboard clone)
            chess_board: search_worker.chess_board.clone(),

            // Populate our repetition detection array
            traversed_positions: search_worker.traversed_positions,
            position_stack_len: search_worker.position_stack_len, 
            
            nodes_processed: 0,
            transposition_table,

            killer_move_table: [[None; MAX_DEPTH as usize]; 2],
            thread_id
        }
    }

    // Search Entry Point
    pub fn root_search(&mut self,
        max_depth: i32, stop_signal: &AtomicBool) -> (Option<ForwardMove>, usize){
        // Start the timer
        let start_time = Instant::now();
        let time_limit = Duration::from_secs(DEPTH_SEARCH_LIMIT);

        // Thread_id = 0 is the main thread, the rest are helper threads
        let start_depth = if self.thread_id == 0 {
            1
        } else {
            // Subtle offset: Helpers start at depth 2 or 3, but move up 1 depth at a time.
            2 + (self.thread_id % 2)
        };

        // --- ITERATIVE DEEPENING LOOP ---
        let mut best_move_overall: Option<ForwardMove> = None;
        let mut depth = start_depth;

        while depth <= max_depth {
            if stop_signal.load(Ordering::Relaxed) || start_time.elapsed() >= time_limit {
                stop_signal.store(true, Ordering::Relaxed);
                break;
            }
            
            let result = self.negamax(depth, 0, -INFINITY, INFINITY, 
                best_move_overall, stop_signal);
            
            if !stop_signal.load(Ordering::Relaxed){
                best_move_overall = result.best_move;

                if self.thread_id == 0 {
                    println!("[Master Thread] Completed Depth: {} | Best Move Score: {:?}", 
                        depth, result.best_move);
                    if depth >= PV_DEPTH { 
                        stop_signal.store(true, Ordering::Relaxed);
                        break;
                    }
                }
            } else {
                break;
            }

            depth += 1;
        }
            
        (best_move_overall, self.nodes_processed)
    }

    // Call this when entering a node (making a move)
    fn push_position(&mut self) {
        let hash = self.chess_board.zobrist_hash();
        self.traversed_positions[self.position_stack_len] = hash;
        self.position_stack_len += 1;
    }

    // Call this when backing out of a node (undoing a move)
    fn pop_position(&mut self) {
        self.position_stack_len -= 1;
    }

    // Fast linear scan inside the L1 CPU cache stepping by 2
    fn is_three_move_repetition(&self) -> bool {
        let current_hash = self.chess_board.zobrist_hash();
        
        // A repetition requires at least 4 plies to pass (2 full moves) 
        // for the same position to occur a second time on your turn.
        if self.position_stack_len < 4 {
            return false;
        }

        // Initialize to 1 because the current position is the 1st occurrence
        let mut curr_count = 1;

        // Start 2 plies back (the last time it was this player's turn)
        let mut i = self.position_stack_len - 2;

        loop {
            if self.traversed_positions[i] == current_hash {
                curr_count += 1;
                if curr_count == 3 {
                    return true;
                }
            }

            // Clean underflow protection and loop exit
            if i >= 2 {
                i -= 2;
            } else {
                break;
            }
        }

        false
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

        self.push_position();

        let remove_piece = self.chess_board.execute_move(forward_move);
        let undo_move = UndoMove {
            start_sq: forward_move.start_sq,
            end_sq: forward_move.end_sq,
            move_type: forward_move.move_type,
            captured_piece: remove_piece,
            prev_castle_rights,
            prev_en_passant,
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
            self.pop_position();

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

    // A simple, linear-logarithmic approximation using integer division:
    // R increases slowly as depth and move count grow.
    fn calculate_lmr_reduction(&mut self, depth: i32, moves_tried: i32) -> i32 {
        LMR_TABLE[depth as usize][moves_tried as usize]
    }

    // Store Killer Move - No Captures
    fn store_killer_move(&mut self, new_killer_move: ForwardMove, depth: i32) {
        // Store Non-Captures for Killer Move
        if matches!(new_killer_move.move_type, 
            MoveFlag::CAPTURE | MoveFlag::ENPASSANT | MoveFlag::PROMOTIONQUEEN
        ) {
            return;
        }

        let depth_idx = depth as usize;

        // If this move is already our primary killer, do nothing
        if self.killer_move_table[0][depth_idx] == Some(new_killer_move) {
            return;
        }
        
        // Later Moves would case a stronger beta cutoff
        self.killer_move_table[1][depth_idx] = self.killer_move_table[0][depth_idx];
        self.killer_move_table[0][depth_idx] = Some(new_killer_move);
    }

    // Process Negamax
    fn negamax(&mut self, depth: i32, ply: i32, mut alpha: i32, mut beta: i32, 
        mut pv_move_hint: Option<ForwardMove>, stop_signal: &AtomicBool) -> SearchResult {
        
        // Original Alpha for Transposition Table
        let original_alpha = alpha;
        
        // Three Move Repetition Draw
        if self.is_three_move_repetition() {
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

        // Halt Signal
        if (self.nodes_processed & 2047) == 0 && stop_signal.load(Ordering::Relaxed) {
            return SearchResult { score: 0, best_move: None };
        }

        // Leaf Node Condition -> Drop into Quiescence Search
        if depth == 0 {
            return SearchResult {
                score: self.quiescence_search(alpha, beta, ply, -1, stop_signal),
                best_move: None,
            };
        }

        let mut best_move = None;   
        let mut legal_moves_played = 0;
        let mut best_score = -INFINITY;

        let mut gen_moves = ArrayVec::<ForwardMove, 256>::new();
        self.chess_board.generate_moves(&mut gen_moves, pv_move_hint, 
            depth, &self.killer_move_table);

        // Late Move Reduction 
        let mut lmr_eligibility;
        let mut moves_tried: i32 = 0;
        let king_in_check = self.chess_board.is_in_check(); 

        for forward_move in &gen_moves {
            // Check LMR Eligibility
            lmr_eligibility = false;
            if depth >= 3 && moves_tried > 2 && !king_in_check && matches!(forward_move.move_type, MoveFlag::MOVE) {
                lmr_eligibility = true;
            }

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

            // LMR Reduction
            let mut negamax_result;
            if lmr_eligibility {
                let mut reduction = self.calculate_lmr_reduction(depth, moves_tried);
                // Introduce Lazy SMP Divergence
                if self.thread_id > 0 {
                    // Even threads search slightly deeper on quiet lines, odd threads prune harder
                    if self.thread_id % 2 == 0 {
                        reduction += 1; // Prune deeper paths aggressively
                    } else {
                        // Skip certain reductions entirely to look for hidden tactical refutations
                        if moves_tried > 6 { reduction = reduction.saturating_sub(1); }
                    }
                }

                let reduced_depth = (depth - 1 - reduction).max(1);

                negamax_result = self.negamax(reduced_depth, ply + 1,  -alpha - 1, -alpha, None, stop_signal);

                if -negamax_result.score > alpha {
                    negamax_result = self.negamax(depth - 1, ply + 1, -beta, -alpha, None, stop_signal);
                }
            } else {
                negamax_result = self.negamax(depth - 1, ply + 1, -beta, -alpha, None, stop_signal);
            }

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

                // Move Triggered a Beta Cutoff - Store as Killer Move
                self.store_killer_move(*forward_move, depth);

                return SearchResult { score: best_score, best_move };
            }

            if score > alpha {
                alpha = score;
            }

            moves_tried += 1;
        }

        // 4. Handle terminal nodes cleanly if no legal moves exist
        if legal_moves_played == 0 {
            if king_in_check {
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
        self.nodes_processed += 1;
        static_eval
    }

    // Quiescence Search 
    fn quiescence_search(&mut self, mut alpha: i32, mut beta: i32, ply: i32, 
        depth: i32, stop_signal: &AtomicBool) -> i32 {
            
        // Three Move Repetition Draw
        if self.is_three_move_repetition() {
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

        // Halt Signal
        if (self.nodes_processed & 2047) == 0 && stop_signal.load(Ordering::Relaxed) {
            return 0; 
        }

        let mut legal_moves_played = 0;
        let mut best_move = None;
        let mut hash_flag = HashFlag::UPPERBOUND;

        // Generate strictly legal tactical moves directly onto the global stack
        let mut gen_moves = ArrayVec::<ForwardMove, 256>::new();

        if king_in_check {
            // King IS in check: Generate all Moves
            self.chess_board.generate_moves(&mut gen_moves, pv_move_hint, 
                depth, &self.killer_move_table);
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
            let score = -self.quiescence_search(-beta, -alpha, ply + 1, depth - 1, stop_signal);
            
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
        self.chess_board.generate_moves(gen_moves, None, 
            -1, &self.killer_move_table);
        gen_moves.retain(|cmd| {
            matches!(
                cmd.move_type,
                MoveFlag::PROMOTIONQUEEN | 
                MoveFlag::CAPTURE | MoveFlag::ENPASSANT
            )
        });
    }
}