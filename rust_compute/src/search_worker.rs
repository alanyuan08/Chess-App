use std::sync::atomic::{AtomicBool, Ordering};
use crate::transposition_table::*;
use crate::board_accumlator::*;
use arrayvec::ArrayVec;
use std::sync::Arc;
use std::cmp;

use crate::move_command::*;
use crate::chess_game::*;
use crate::lmr_table::*;

use crate::parser::*;
use crate::chess_board::*;
use crate::nnue_network::*;
use crate::search_command::*;

#[derive(Clone)] 
pub struct SearchWorker  {
    transposition_table: Arc<TranspositionTable>,
    nodes_processed: usize,

    history: [UndoMove; MAX_GAME_MOVES],
    history_index: usize,

    chess_board: ChessBoard,

    traversed_positions: [u64; MAX_GAME_MOVES],
    position_stack_len: usize,

    killer_move_table: [[ForwardMove; 2]; 128],
    thread_id: usize,

    thread_buffer: NnueInferenceBuffer,
    board_accumulator: AccumulatorHistory,
    
    stop_signal: Arc<AtomicBool>,
    age: i8,
}

impl SearchWorker {
    pub fn new(
        transposition_table: Arc<TranspositionTable>,
        nnue_network: &'static NnueNetwork,
        stop_signal: Arc<AtomicBool>,
        thread_id: usize,
    ) -> Self {

        let mut board_accumulator = AccumulatorHistory::default();
        let mut chess_board = ChessBoard::new(nnue_network);
        chess_board.init_board(&mut board_accumulator);

        Self {
            history: [UndoMove::NULL_UNDO_MOVE; MAX_GAME_MOVES],
            history_index: 0,

            chess_board,
            traversed_positions: {
                [0; MAX_GAME_MOVES]
            },
            position_stack_len: 0, 
            
            nodes_processed: 0,
            transposition_table,

            killer_move_table: [[ForwardMove::NULL_MOVE; 2]; 128],
            thread_id,

            thread_buffer: NnueInferenceBuffer::default(),
            board_accumulator: AccumulatorHistory::default(),

            stop_signal,
            age: 0,
        }
    }

    // Search Entry Point
    pub fn root_search(&mut self, engine_age: i8) -> (ForwardMove, usize, usize) {
        // Start the timer / reset counters
        self.nodes_processed = 0;

        // Reset the age
        self.age = engine_age;

        // --- ITERATIVE DEEPENING LOOP ---
        let mut best_move_overall: ForwardMove = ForwardMove::NULL_MOVE;
        let mut depth = 1;

        // Aspiration window configuration
        let mut alpha = -INFINITY;
        let mut beta = INFINITY;
        let window_delta = 50;

        while depth <= MAX_DEPTH {
            let mut result = self.negamax(depth, 0, alpha, beta, best_move_overall, true);

            // Check for fail-low or fail-high (if search wasn't aborted)
            if !result.was_aborted && !self.stop_signal.load(Ordering::Relaxed) {
                
                // Pure Fail-Soft Widen Loop: Keep re-searching until the score falls cleanly inside the window
                while result.score <= alpha || result.score >= beta {
                    if result.score <= alpha {
                        // Fail-low: The true score is worse than expected. 
                        beta = result.score;
                        alpha = -INFINITY;
                    } else if result.score >= beta {
                        // Fail-high: The true score is better than expected.
                        alpha = result.score;
                        beta = INFINITY;
                    }

                    // Re-search this exact depth with the properly adjusted bounds
                    result = self.negamax(depth, 0, alpha, beta, best_move_overall, true);

                    // Break immediately if a re-search hits the timeout/halt barrier
                    if result.was_aborted || self.stop_signal.load(Ordering::Relaxed) {
                        break;
                    }
                }
            }
            
            // Post-depth assessment and window initialization for the subsequent depth
            if !self.stop_signal.load(Ordering::Relaxed) {
                if self.thread_id == 0 {
                    if !result.was_aborted {
                        best_move_overall = result.best_move;
                        
                        // Verified: Scores are positive/correct relative to root. No negation needed.
                        println!("[Master Thread] Completed Depth: {} | Best Move: {:?} | Evaluation: {}", 
                            depth, best_move_overall, result.score);
                        
                        // Initialize a narrow aspiration window around our newly verified score for the next depth layer
                        alpha = result.score - window_delta;
                        beta = result.score + window_delta;
                    }

                    if depth >= MAX_DEPTH { 
                        self.stop_signal.store(true, Ordering::Relaxed);
                        break;
                    }
                }
            } else {
                break;
            }

            depth += 1;
        }
        
        (best_move_overall, self.nodes_processed, self.thread_id)
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

    pub fn process_move(&mut self, uci_move: String) {
        if self.history_index >= 512 { 
            eprintln!("History Index too long");
            return;
         } 

        let move_command: ForwardMove = 
            parse_forward_move_with_board(uci_move, &self.chess_board);
        self.process_forward_move(move_command);
    }

    fn process_forward_move(&mut self, forward_move: ForwardMove) {
        // Store Value prior to Executing Move
        let prev_castle_rights = self.chess_board.castle_rights(); 
        let prev_en_passant = self.chess_board.en_passant();
        self.chess_board.increment_ply();

        let mut captured_piece = BoardPiece::NONE;
        let move_type = forward_move.move_type;

        if forward_move.move_type == MoveFlag::NULL {
            self.chess_board.execute_move(forward_move); 
            self.chess_board.null_move_accumulator(&mut self.board_accumulator);
        } else {
            if self.chess_board.does_move_cross_king_bucket(forward_move) {
                captured_piece = self.chess_board.execute_move(forward_move);
                self.chess_board.create_accumulator_from_scratch(&mut self.board_accumulator);
            } else {
                self.chess_board.make_move(forward_move, &mut self.board_accumulator);
                captured_piece = self.chess_board.execute_move(forward_move);
            }
        }

        let undo_state = UndoMove {
            start_sq: forward_move.start_sq,
            end_sq: forward_move.end_sq,
            capture_sq: forward_move.capture_sq,
            prev_castle_rights,
            prev_en_passant,
            move_type,
            captured_piece,
        };
        self.history[self.history_index] = undo_state;

        self.push_position();
        self.history_index += 1;
    }

    fn process_backward_move(&mut self) {
        if self.history_index == 0 {
            return;
        }

        self.history_index -= 1;
        let undo_move = self.history[self.history_index];
        self.chess_board.unmake_move();

        // ChessBoard Undo Move
        self.chess_board.unexecute_move(undo_move);
        self.pop_position();
    }

    // A simple, linear-logarithmic approximation using integer division:
    // R increases slowly as depth and move count grow.
    #[inline]
    fn get_reduced_depth(&self, depth: i8, moves_tried: i8) -> i8 {
        // 1. Safe array bounds clamping
        let d = depth.clamp(0, 63) as usize;
        let m = moves_tried.clamp(0, 63) as usize;

        let base_reduction = LMR_TABLE[d][m] as i16;

        // 2. Branchless Thread Diversity Calculation
        let is_helper_thread = (self.thread_id != 0) as i16;
        let is_odd_thread = ((self.thread_id % 2) != 0) as i16;
        let is_even_thread = 1 - is_odd_thread;
        
        let depth_gt_4 = (depth > 4) as i16;
        let even_offset = (-1 * (1 - depth_gt_4)) + (-2 * depth_gt_4);

        // Combine offsets branchlessly based on the active thread ID properties
        let thread_offset = (1 * is_odd_thread) + (even_offset * is_even_thread);
        
        // Apply thread diversity adjustment, clamped at a minimum of 0
        let adjusted_reduction = base_reduction + (thread_offset * is_helper_thread);
        let reduction = adjusted_reduction * (adjusted_reduction > 0) as i16;

        // 3. Calculate raw reduced depth
        let raw_reduced_depth = (depth as i16) - 1 - reduction;

        // 4. Branchless Safety Clamp (Forces result between 0 and depth - 1)
        // If raw < 0, it clamps up to 0. If raw >= depth, it clamps down to depth - 1.
        let is_less_than_zero = (raw_reduced_depth < 0) as i16;
        let is_greater_equal_depth = (raw_reduced_depth >= depth as i16) as i16;
        let is_valid = 1 - is_less_than_zero - is_greater_equal_depth;

        let reduced_depth = (0 * is_less_than_zero) 
            + (((depth as i16) - 1) * is_greater_equal_depth) 
            + (raw_reduced_depth * is_valid);

        reduced_depth as i8
    }

    // Store Killer Move - No Captures (Entirely Branchless)
    fn store_killer_move(&mut self, new_killer_move: ForwardMove, ply: i16) {
        let ply_idx = (ply as usize) & 127;

        let k0 = self.killer_move_table[ply_idx][0];
        let k1 = self.killer_move_table[ply_idx][1];

        // Evaluate match statuses
        let match_k0 = k0 == new_killer_move;
        let match_k1 = k1 == new_killer_move;

        // Determine the next state for Slot 0 and Slot 1 
        // These clean conditional selections compile to zero-branch hardware instructions (CMOV)
        let next_slot0 = if match_k0 { k0 } else { new_killer_move };
        let next_slot1 = if match_k0 { k1 } else if match_k1 { k0 } else { k0 };

        // Save states back down into the table
        self.killer_move_table[ply_idx][0] = next_slot0;
        self.killer_move_table[ply_idx][1] = next_slot1;
    }

    // StockFish NMP Reduction algorithm
    #[inline]
    fn calculate_nmp_reduction(&self, depth: i8, static_eval: i16, beta: i16) -> i8 {
        let base_reduction = 3 + (depth / 4);
        
        let eval_bonus = ((static_eval - beta) / 300).clamp(0, 3);
        
        base_reduction + (eval_bonus as i8)
    }

    // Static Eval
    #[inline]
    fn static_eval(&mut self) -> i16 {
        self.chess_board.evaluate(&self.board_accumulator, &mut self.thread_buffer)
    }

    // Process Negamax
    // Transposition Table Summary
    // Exact -> Preivous Function call found a value between Alpha and Beta
    // - Pass Value to parent

    // LOWER BOUND (Fail-High / Beta Cutoff): 
    // - Previous search found a value that exceeded or met Beta.
    // - The true value is AT LEAST this high. 
    // - Action: alpha = cmp::max(alpha, retrieved_score)

    // UPPER BOUND (Fail-Low): 
    // - Previous search couldn't find any move that beat Alpha.
    // - The true value is AT MOST this low. 
    // - Action: beta = cmp::min(beta, retrieved_score)

    // True Value -> If we ran Min-Max all the way down with zero pruning
    // Alpha - Best value a Maximizer can guarantee, hence true value is greater than or equal to this
    // Beta - Worst value a Minimizer can guarantee, hence true value is else than or equal to this

    // Fail-Soft -> Allows the function to return a value completely outside the [alpha, beta] window
    // Hard CutOff -> Once a value causes alpha >= beta, stop the search

    // !self.stop_signal.load(Ordering::Relaxed) should be called prior to save as the search
    // is likely incompelte
    fn negamax(&mut self, depth: i8, ply: i16, mut alpha: i16, mut beta: i16, 
        mut pv_move_hint: ForwardMove, allow_null: bool) -> SearchResult {

        // 1. Halt Signal
        if self.nodes_processed & 0x3FFF == 0 && self.stop_signal.load(Ordering::Relaxed) {
            return SearchResult { 
                score: 0, 
                best_move: ForwardMove::NULL_MOVE, 
                was_aborted: true 
            };
        }

        self.nodes_processed += 1;
        
        // 2. Three Move Repetition Draw
        if self.is_three_move_repetition() {
            return SearchResult {
                score: 0,
                best_move: ForwardMove::NULL_MOVE,
                was_aborted: false,
            };
        }

        let hash = self.chess_board.zobrist_hash();
        let king_in_check = self.chess_board.is_in_check(); 

        let original_alpha = alpha; 
        let original_beta = beta;

        // 3. Fail-Soft Transposition Table
        let tt_entry = self.transposition_table.probe(hash, ply);
        if tt_entry.flag != HashFlag::EMPTY {
            let retrieved_score = tt_entry.score;
            let retrieved_depth = tt_entry.depth;
            pv_move_hint = tt_entry.forward_move;

            if retrieved_depth >= depth {
                // EXACT: The true minimax value was found; return it immediately.
                if tt_entry.flag == HashFlag::EXACT {
                    return SearchResult {
                        score: retrieved_score,
                        best_move: pv_move_hint,
                        was_aborted: false,
                    };
                }
                    
                // LOWER BOUND: There is a retrieved score higher than beta
                // Stop search as the miminizer will not return retrieved_score
                else if tt_entry.flag == HashFlag::LOWERBOUND {
                    // Fail-Soft wants the true Value
                    if retrieved_score >= beta {
                        return SearchResult {
                            score: retrieved_score,
                            best_move: pv_move_hint,
                            was_aborted: false,
                        };
                    }
                    // True Value is at least this high, Maximizer is interested 
                    alpha = cmp::max(alpha, retrieved_score);
                }

                // UPPER BOUND: There was a previous searched that searched all values
                // and they are all below alpha 
                else if tt_entry.flag == HashFlag::UPPERBOUND {
                    // Fail-Soft wants the true Value
                    if retrieved_score <= alpha {
                        return SearchResult {
                            score: retrieved_score,
                            best_move: pv_move_hint,
                            was_aborted: false,
                        };
                    }
                    
                    // True Vlaue is at most this high, Minimizer is interested
                    beta = cmp::min(beta, retrieved_score);
                }  
            }
        }

        // Leaf Node Condition -> Drop into Quiescence Search
        // Q-Search uses Hard-Cut Off and the value is bounded between alpha - beta
        if depth <= 0 {
            return self.quiescence_search(alpha, beta, ply, pv_move_hint);
        }

        let static_eval = self.static_eval();

        // 4. Late Futility Pruning
        if depth <= 3 && !king_in_check && beta < MATE_THRESHOLD && beta > -MATE_THRESHOLD {
            // Highly accurate NNUE allows a significantly tighter linear slope
            let base_margin = 40;
            let rfp_margin = (40 * depth as i16) + base_margin;
            
            if static_eval - rfp_margin >= beta {
                let aborted = self.stop_signal.load(Ordering::Relaxed);
                let fail_soft_score = static_eval;

                if !aborted && beta == original_beta {
                    self.transposition_table.store(
                        hash, fail_soft_score, 
                        depth, ply, self.age,
                        ForwardMove::NULL_MOVE, HashFlag::LOWERBOUND
                    );
                }

                return SearchResult {
                    score: if aborted { 0 } else { fail_soft_score },
                    best_move: ForwardMove::NULL_MOVE,
                    was_aborted: aborted,
                };
            }
        }

        // 5. Pure Fail-Soft Null Move Pruning
        if allow_null && !matches!(pv_move_hint.move_type, MoveFlag::NULL) && 
            !king_in_check && depth >= 3 && 
            self.chess_board.has_major_pieces() && ply > 0 &&
            beta < MATE_THRESHOLD && beta > -MATE_THRESHOLD && 
            static_eval >= beta {

            // Calculate Reduction
            let reduction = self.calculate_nmp_reduction(depth, static_eval, beta);
            let next_depth = (depth - reduction).max(0); 

            // Make the null move (switch sides, update en-passant/hash keys)
            let null_move = ForwardMove::NULL_MOVE;
            self.process_forward_move(null_move);
            
            let null_result = self.negamax(next_depth, ply + 1, -beta, -beta + 1, ForwardMove::NULL_MOVE, false);
            let null_score = -null_result.score;
            
            self.process_backward_move();

            if null_result.was_aborted {
                return SearchResult { 
                    score: 0, 
                    best_move: ForwardMove::NULL_MOVE, 
                    was_aborted: true 
                };
            }
            
            if null_score >= beta {
                if beta == original_beta {
                    self.transposition_table.store(
                        hash, null_score, 
                        depth, ply, self.age,
                        ForwardMove::NULL_MOVE, HashFlag::LOWERBOUND
                    );
                }

                return SearchResult { 
                    score: null_score, 
                    best_move: ForwardMove::NULL_MOVE, 
                    was_aborted: false 
                };
            }
        }

        let mut legal_moves_played = 0;
        let mut moves_tried: i8 = 0;
        let mut best_move = ForwardMove::NULL_MOVE;   
        let mut best_score = -MATE_VALUE;

        let mut gen_moves = ArrayVec::<ForwardMove, 256>::new();
        self.chess_board.generate_moves(&mut gen_moves, pv_move_hint, 
            ply, &self.killer_move_table, false);

        // 6. Move Search Loop
        for forward_move in &gen_moves {
            // --- PRE-MOVE TACTICAL PRUNING GATES ---
            if !king_in_check && depth <= 4
                && matches!(forward_move.move_type, 
                    MoveFlag::CAPTURE | MoveFlag::ENPASSANT | MoveFlag::PROMOTIONQUEEN) 
            {
                let exact_see_score = 
                    self.chess_board.static_exchange_evaluation(*forward_move);

                if exact_see_score < 0 {
                    continue; 
                }

                if static_eval + exact_see_score + 200 < alpha {
                    continue; 
                }
            }

            self.process_forward_move(*forward_move);

            // Pseudo legal move exposes check, undo move
            if self.chess_board.is_previous_player_king_in_check() {    
                self.process_backward_move();
                continue;
            }

            legal_moves_played += 1;
            moves_tried += 1;

            let mut score;

            // 1. FIRST MOVE (PV Move): Search with full window immediately
            if legal_moves_played == 1 {
                let search_result = self.negamax(depth - 1, ply + 1, -beta, -alpha, ForwardMove::NULL_MOVE, true);
                score = -search_result.score;

                if search_result.was_aborted {
                    self.process_backward_move();
                    return SearchResult { score: 0, best_move: ForwardMove::NULL_MOVE, was_aborted: true };
                }
            } else {
                // 2. SUBSEQUENT MOVES: Run LMR or Scout verification
                let mut lmr_eligibility = false;
                if depth >= 3 && moves_tried > 4 && !king_in_check && matches!(forward_move.move_type, MoveFlag::MOVE)
                    && !self.chess_board.is_in_check() {
                    
                    lmr_eligibility = true;
                }

                if lmr_eligibility {
                    let reduced_depth = self.get_reduced_depth(depth, moves_tried);
                    let search_result = self.negamax(reduced_depth, ply + 1, -alpha - 1, -alpha, ForwardMove::NULL_MOVE, true);
                    score = -search_result.score;

                    if search_result.was_aborted {
                        self.process_backward_move();
                        return SearchResult { score: 0, best_move: ForwardMove::NULL_MOVE, was_aborted: true };
                    }

                    if score > alpha && reduced_depth < depth - 1 {
                        let research_result = self.negamax(depth - 1, ply + 1, -alpha - 1, -alpha, ForwardMove::NULL_MOVE, true);
                        score = -research_result.score;

                        if research_result.was_aborted {
                            self.process_backward_move();
                            return SearchResult { score: 0, best_move: ForwardMove::NULL_MOVE, was_aborted: true };
                        }
                    }
                } else {
                    // Non-PV Move Node: Scout with a narrow null-window search tracking alpha
                    let search_result = self.negamax(depth - 1, ply + 1, -alpha - 1, -alpha, ForwardMove::NULL_MOVE, true);
                    score = -search_result.score;

                    if search_result.was_aborted {
                        self.process_backward_move();
                        return SearchResult { score: 0, best_move: ForwardMove::NULL_MOVE, was_aborted: true };
                    }
                }

                // If a scout/LMR search reveals a better move, research with the full window
                if score > alpha && score < beta {
                    let search_result = self.negamax(depth - 1, ply + 1, -beta, -alpha, ForwardMove::NULL_MOVE, true);
                    score = -search_result.score;

                    if search_result.was_aborted {
                        self.process_backward_move();
                        return SearchResult { score: 0, best_move: ForwardMove::NULL_MOVE, was_aborted: true };
                    }
                }
            }

            // Always undo the move exactly once per iteration at the bottom of the loop lifecycle
            self.process_backward_move();

             if score > best_score {
                best_score = score;
                best_move = *forward_move;
            }

            if score >= beta {
                if matches!(
                    best_move.move_type, 
                    MoveFlag::MOVE | MoveFlag::PAWNOPENMOVE | MoveFlag::KINGSIDECASTLE | MoveFlag::QUEENSIDECASTLE
                ) {
                    self.store_killer_move(best_move, ply); 
                }
                break; 
            }

            if score > alpha {
                alpha = score;
            }
        }

        let aborted = self.stop_signal.load(Ordering::Relaxed);
        // 4. Handle terminal nodes cleanly if no legal moves exist
        if legal_moves_played == 0 {
            // Check if the search was halted BEFORE anything else
            if aborted {
                return SearchResult { 
                    score: 0, 
                    best_move: ForwardMove::NULL_MOVE, 
                    was_aborted: true 
                };
            }

            let terminal_score = if king_in_check {
                -MATE_VALUE + ply
            } else {
                0
            };

            self.transposition_table.store(
                hash, terminal_score,
                depth, ply, self.age,
                ForwardMove::NULL_MOVE, HashFlag::EXACT
            );

            return SearchResult { 
                score: terminal_score, 
                best_move: ForwardMove::NULL_MOVE,
                was_aborted: false
            };
        }

        if aborted {
            return SearchResult { 
                score: 0, 
                best_move: ForwardMove::NULL_MOVE, 
                was_aborted: true 
            };
        }

        // Pure Fail-Soft TT Storage and Return Contract
        let final_hash_flag = if best_score <= original_alpha {
            HashFlag::UPPERBOUND // Fail-Low: Checked everything, all moves sucked
        } else if best_score >= original_beta {
            HashFlag::LOWERBOUND // Fail-High: Pruned early, found a refutation
        } else {
            HashFlag::EXACT
        };
        
        self.transposition_table.store(
            hash, best_score, 
            depth, ply, self.age,
            best_move, final_hash_flag
        );

        SearchResult { 
            score: best_score, 
            best_move, 
            was_aborted: false 
        }
    }

    // Quiescence Search -> Q-Search only uses Score and was_aborted. 
    // Fail-Soft -> Allows the function to return a value completely outside the [alpha, beta] window
    // Hard Cut-Off -> Once a value causes alpha >= beta, stop the search
    fn quiescence_search(&mut self, mut alpha: i16, beta: i16, ply: i16, mut pv_move_hint: ForwardMove) -> SearchResult {        
        // 1. Halt Signal
        if self.nodes_processed & 0x3FFF == 0 && self.stop_signal.load(Ordering::Relaxed) {
            return SearchResult { 
                score: 0, 
                best_move: ForwardMove::NULL_MOVE,
                was_aborted: true,
            };
        }

        // Nodes Processed
        self.nodes_processed += 1;

        // 2. Three Move Repetition Draw
        if self.is_three_move_repetition() {
            return SearchResult { 
                score: 0, 
                best_move: ForwardMove::NULL_MOVE,
                was_aborted: false,
            };
        }

        let hash = self.chess_board.zobrist_hash();
        const Q_DEPTH_MARKER: i8 = -1;
        let original_alpha = alpha;
        let original_beta = beta;

        // 3. Tranposition Table
        let tt_entry = self.transposition_table.probe(hash, ply);
        if tt_entry.flag != HashFlag::EMPTY {
            let retrieved_score = tt_entry.score;
            pv_move_hint = tt_entry.forward_move;

            // EXACT: The true minimax value was found; return it immediately.
            if tt_entry.flag == HashFlag::EXACT {
                return SearchResult { 
                    score: retrieved_score, 
                    best_move: pv_move_hint,
                    was_aborted: false,
                };
            }

            // LOWER BOUND: The true score is AT LEAST this high. 
            else if tt_entry.flag == HashFlag::LOWERBOUND {
                if retrieved_score >= beta {
                    return SearchResult {
                        score: retrieved_score,
                        best_move: pv_move_hint,
                        was_aborted: false,
                    };
                }
                alpha = cmp::max(alpha, retrieved_score);
            }
            // UPPER BOUND: The true score is AT MOST this high.
            else if tt_entry.flag == HashFlag::UPPERBOUND &&
                retrieved_score <= alpha {

                return SearchResult {
                    score: retrieved_score,
                    best_move: pv_move_hint,
                    was_aborted: false,
                };
            } 
        }

        // 4. Standing Pat 
        let king_in_check = self.chess_board.is_in_check(); 
        let mut best_score;
        let mut best_move = ForwardMove::NULL_MOVE;
        let pruning_baseline; 

        if !king_in_check {
            // Evaluated relative to the side-to-move
            let static_eval = self.static_eval();
            best_score = static_eval;
            pruning_baseline = static_eval;

            if best_score >= beta {
                let aborted = self.stop_signal.load(Ordering::Relaxed);
                
                if !aborted {
                    self.transposition_table.store(
                        hash, best_score, 
                        Q_DEPTH_MARKER, ply, self.age, 
                        ForwardMove::NULL_MOVE, HashFlag::LOWERBOUND
                    );
                }

                return SearchResult {
                    score: if aborted { 0 } else { best_score }, 
                    best_move: ForwardMove::NULL_MOVE,
                    was_aborted: aborted,
                };
            }

            if best_score > alpha {
                alpha = best_score;
            }
        } else {
            // Evasion baseline tracking bounds prevents masking mate strings
            best_score = -MATE_VALUE + ply;
            pruning_baseline = alpha - 200; 
        }

        // Generate strictly legal tactical moves directly onto the global stack
        let mut gen_moves = ArrayVec::<ForwardMove, 256>::new();
        let mut legal_moves_played = 0;

        let only_tactical = !king_in_check;
        self.chess_board.generate_moves(&mut gen_moves, pv_move_hint, 
            ply, &self.killer_move_table, only_tactical);

        // 5. Quiscence Search
        for forward_move in &gen_moves {
            // --- PRE-MOVE TACTICAL PRUNING GATES ---
            if !king_in_check {
                // Calculate the exact transaction value from our current perspective (pre-move)
                let exact_see_score = self.chess_board.static_exchange_evaluation(*forward_move);

                // 1. Strict Tactical Gate: If the exchange immediately loses material, drop it
                if exact_see_score < 0 {
                    continue; 
                }

                // 2. Delta Pruning Gate using check-safe baseline
                if pruning_baseline + exact_see_score + 200 < alpha {
                    continue; 
                }
            }

            // --- STATE COMMITMENT & LEGALITY FILTER ---
            self.process_forward_move(*forward_move);
            if self.chess_board.is_previous_player_king_in_check() {
                self.process_backward_move();
                continue;
            }

            legal_moves_played += 1;

            // --- RECURSIVE SEARCH CASCADE ---
            let search_result = self.quiescence_search(-beta, -alpha, ply + 1, ForwardMove::NULL_MOVE);
            self.process_backward_move();

            if search_result.was_aborted {
                return SearchResult { 
                    score: 0, 
                    best_move: ForwardMove::NULL_MOVE,
                    was_aborted: true,
                };
            }

            // Bring child score back to our perspective
            let score = -search_result.score;
            
            if score > best_score {
                best_score = score;
                best_move = *forward_move;
            }

            if score >= beta { 
                break; 
            }

            if score > alpha {
                alpha = score;
            }
        }

        let aborted = self.stop_signal.load(Ordering::Relaxed);
        // 6. Checkmate/Stalemate Terminal Node Resolutions
        if legal_moves_played == 0 {
            if aborted {
                return SearchResult { 
                    score: 0, 
                    best_move: ForwardMove::NULL_MOVE, 
                    was_aborted: true 
                };
            }

            let terminal_score = if king_in_check {
                -MATE_VALUE + ply
            } else {
                0
            };

            self.transposition_table.store(
                hash, terminal_score,
                Q_DEPTH_MARKER, ply, self.age,
                ForwardMove::NULL_MOVE, HashFlag::EXACT,
            );

            return SearchResult { 
                score: terminal_score,
                best_move: ForwardMove::NULL_MOVE, 
                was_aborted: false, 
            };
        }

        // Global halt catch before finalizing the Q-search node evaluation
        if aborted {
            return SearchResult { 
                score: 0, 
                best_move: ForwardMove::NULL_MOVE, 
                was_aborted: true 
            };
        }

        // Pure Fail-Soft TT Storage and Return Contract
        let final_hash_flag = if best_score <= original_alpha {
            HashFlag::UPPERBOUND
        } else if best_score >= original_beta {
            HashFlag::LOWERBOUND
        } else {
            HashFlag::EXACT
        };

        self.transposition_table.store(
            hash, best_score,
            Q_DEPTH_MARKER, ply, self.age,
            best_move, final_hash_flag,
        );

        SearchResult { 
            score: best_score, 
            best_move,
            was_aborted: false,
        }
    }
}