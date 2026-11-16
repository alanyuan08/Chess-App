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

    #[inline(always)]
    fn get_aspiration_delta(score: i16) -> i16 {
        let abs_score = score.saturating_abs();
        if abs_score <= 250 {
            15
        } else if abs_score <= 400 {
            25
        } else {
            45
        }
    }

    // Search Entry Point
    pub fn root_search(&mut self, engine_age: i8) -> (ForwardMove, usize, usize) {
        self.nodes_processed = 0;
        self.age = engine_age;

        let mut best_move_overall: ForwardMove = ForwardMove::NULL_MOVE;
        let mut depth = 1;

        let mut alpha = -INFINITY;
        let mut beta = INFINITY;

        while depth <= MAX_DEPTH {
            // Seed delta from the active window half-width (or 15 cp fallback)
            let mut delta = if alpha > -INFINITY && beta < INFINITY {
                (beta - alpha) / 2
            } else {
                15
            };

            let mut result = self.negamax(depth, 0, alpha, beta, best_move_overall, true);

            // Fail-soft gradual aspiration widening loop
            if !result.was_aborted && !self.stop_signal.load(Ordering::Relaxed) {
                while result.score <= alpha || result.score >= beta {
                    if result.score <= alpha {
                        // Fail-low: pull beta down to midpoint, widen alpha downward
                        beta = alpha.saturating_add(beta) / 2;
                        alpha = if delta >= 200 {
                            -INFINITY
                        } else {
                            result.score.saturating_sub(delta)
                        };
                    } else if result.score >= beta {
                        // Fail-high: lock alpha at the exact proven score, widen beta upward
                        alpha = result.score;
                        beta = if delta >= 200 {
                            INFINITY
                        } else {
                            result.score.saturating_add(delta)
                        };
                    }

                    delta = delta.saturating_mul(2);
                    result = self.negamax(depth, 0, alpha, beta, best_move_overall, true);

                    if result.was_aborted || self.stop_signal.load(Ordering::Relaxed) {
                        break;
                    }
                }
            }

            // Post-depth bookkeeping and window setup for the next depth
            if !self.stop_signal.load(Ordering::Relaxed) && !result.was_aborted {
                best_move_overall = result.best_move;

                if self.thread_id == 0 {
                    println!(
                        "[Master Thread] Completed Depth: {} | Best Move: {:?} | Evaluation: {}",
                        depth, best_move_overall, result.score
                    );
                }

                if depth >= MAX_DEPTH {
                    self.stop_signal.store(true, Ordering::Relaxed);
                    break;
                }

                // Engage tight score-calibrated aspiration window once depth >= 4 stabilizes tactical noise
                if depth >= 3 && result.score.abs() < MATE_THRESHOLD - 100 {
                    let next_delta = Self::get_aspiration_delta(result.score);
                    alpha = result.score.saturating_sub(next_delta);
                    beta = result.score.saturating_add(next_delta);
                } else {
                    alpha = -INFINITY;
                    beta = INFINITY;
                }
            } else {
                break;
            }

            depth += 1;
        }

        (best_move_overall, self.nodes_processed, self.thread_id)
    }

    // Call this when entering a node (making a move)
    #[inline(always)]
    fn push_position(&mut self) {
        let hash = self.chess_board.zobrist_hash();
        self.traversed_positions[self.position_stack_len] = hash;
        self.position_stack_len += 1;
    }

    #[inline(always)]
    fn push_null_position(&mut self) {
        self.traversed_positions[self.position_stack_len] = 0;
        self.position_stack_len += 1;
    }
    
    #[inline(always)]
    fn pop_position(&mut self) {
        self.position_stack_len -= 1;
    }

    // Fast linear scan inside the L1 CPU cache stepping by 2
    #[inline(always)]
    fn is_three_move_repetition(&self, ply: i16) -> bool {
        let len = self.position_stack_len;
        if len < 5 {
            return false;
        }

        let current_hash = self.chess_board.zobrist_hash();
        let target_repeats = if ply > 0 { 2 } else { 3 };
        let mut count = 1;
        let mut i = len - 3;

        while self.traversed_positions[i] != 0 {
            if self.traversed_positions[i] == current_hash {
                count += 1;
                if count == target_repeats {
                    return true;
                }
            }

            if i < 2 {
                break;
            }
            i -= 2;
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

            self.push_null_position(); 
        } else {
            if self.chess_board.does_move_cross_king_bucket(forward_move) {
                captured_piece = self.chess_board.execute_move(forward_move);
                self.chess_board.create_accumulator_from_scratch(&mut self.board_accumulator);
            } else {
                let prev_castle_rights = self.chess_board.castle_rights();
                captured_piece = self.chess_board.make_move_with_accumulator(
                    forward_move,
                    prev_castle_rights,
                    &mut self.board_accumulator,
                );
            }
            self.push_position();
        }

        self.history[self.history_index] = UndoMove {
            start_sq: forward_move.start_sq,
            end_sq: forward_move.end_sq,
            capture_sq: forward_move.capture_sq,
            prev_castle_rights,
            prev_en_passant,
            move_type,
            captured_piece,
        };
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
    pub fn get_reduced_depth(
        &self,
        depth: i8,
        moves_tried: i8,
        is_pv: bool,
        is_killer: bool,
    ) -> i8 {
        // 1. Safe array bounds clamping
        let d = depth.clamp(0, 63) as usize;
        let m = moves_tried.clamp(0, 63) as usize;
        let base_reduction = LMR_TABLE[d][m] as i16;

        // 2. Branchless Lazy SMP Thread Diversity
        let is_helper = (self.thread_id != 0) as i16;
        let is_odd = ((self.thread_id & 1) != 0) as i16;
        let is_even = 1 - is_odd;
        let even_offset = -((depth <= 4) as i16);

        let thread_offset = is_odd + (even_offset * is_even);

        // 3. Protect PV nodes and killer moves from over-reduction (-1 ply each)
        let pv_bonus = is_pv as i16;
        let killer_bonus = is_killer as i16;

        let reduction = (base_reduction + (thread_offset * is_helper) - pv_bonus - killer_bonus).max(0);

        // 4. Branchless clamp to [0, depth - 1]
        let max_reduced = ((depth as i16) - 1).max(0);
        ((depth as i16) - 1 - reduction).clamp(0, max_reduced) as i8
    }

    // Store Killer Move - No Captures (Entirely Branchless)
    pub fn store_killer_move(&mut self, new_killer_move: ForwardMove, ply: i16) {
        let ply_idx = (ply as usize) & 127;

        let k0 = self.killer_move_table[ply_idx][0];
        let k1 = self.killer_move_table[ply_idx][1];

        let match_k0 = k0 == new_killer_move;
        let match_k1 = k1 == new_killer_move;

        let next_slot0 = if match_k0 { k0 } else if match_k1 { k1 } else { new_killer_move };
        let next_slot1 = if match_k0 { k1 } else { k0 };

        // Save states back down into the table
        self.killer_move_table[ply_idx][0] = next_slot0;
        self.killer_move_table[ply_idx][1] = next_slot1;
    }

    // StockFish NMP Reduction algorithm
    #[inline]
    fn calculate_nmp_reduction(&self, depth: i8, static_eval: i16, beta: i16) -> i8 {
        let base_reduction = 3 + (depth / 4);
        let eval_bonus = ((static_eval - beta) / 200).clamp(0, 3);

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
        pv_move_hint: ForwardMove, allow_null: bool) -> SearchResult {

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
        if self.is_three_move_repetition(ply) {
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
        let mut tt_hint = ForwardMove::NULL_MOVE;   

        // 3. Fail-Soft Transposition Table
        let tt_entry = self.transposition_table.probe(hash, ply);
        if tt_entry.flag != HashFlag::EMPTY {
            let retrieved_score = tt_entry.score;
            let retrieved_depth = tt_entry.depth;
            tt_hint = tt_entry.forward_move;

            if retrieved_depth >= depth {
                // EXACT: The true minimax value was found; return it immediately.
                if tt_entry.flag == HashFlag::EXACT {
                    return SearchResult {
                        score: retrieved_score,
                        best_move: tt_hint,
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
                            best_move: tt_hint,
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
                            best_move: tt_hint,
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
            return self.quiescence_search(alpha, beta, ply);
        }

        let static_eval = self.static_eval();

        // 4. Reverse Futility Pruning (Non-PV nodes, standard linear margin)
        let is_pv = beta - alpha > 1;
        if !is_pv
            && depth <= 3
            && !king_in_check
            && beta.abs() < MATE_THRESHOLD
        {
            let rfp_margin = (65 * depth as i16) + 15;

            if static_eval - rfp_margin >= beta {
                let aborted = self.stop_signal.load(Ordering::Relaxed);
                let fail_soft_score = static_eval - rfp_margin;

                if !aborted {
                    self.transposition_table.store(
                        hash,
                        fail_soft_score,
                        depth,
                        ply,
                        self.age,
                        ForwardMove::NULL_MOVE,
                        HashFlag::LOWERBOUND,
                    );
                }

                return SearchResult {
                    score: if aborted { 0 } else { fail_soft_score },
                    best_move: ForwardMove::NULL_MOVE,
                    was_aborted: aborted,
                };
            }
        }

        // 5. Pure Fail-Soft Null Move Pruning (Non-PV nodes only)
        let is_pv = beta - alpha > 1;
        if !is_pv
            && allow_null
            && !king_in_check
            && depth >= 3
            && ply > 0
            && self.chess_board.has_non_pawn_material_for_active_side()
            && beta.abs() < MATE_THRESHOLD
            && static_eval >= beta
        {
            // Calculate reduction (ensure reduction accounts for the null-move ply)
            let reduction = self.calculate_nmp_reduction(depth, static_eval, beta);
            let next_depth = depth.saturating_sub(reduction).max(0);

            // Execute null move
            let null_move = ForwardMove::NULL_MOVE;
            self.process_forward_move(null_move);

            let null_result = self.negamax(
                next_depth,
                ply + 1,
                -beta,
                -beta + 1,
                ForwardMove::NULL_MOVE,
                false,
            );
            let null_score = -null_result.score;

            self.process_backward_move();

            if null_result.was_aborted {
                return SearchResult {
                    score: 0,
                    best_move: ForwardMove::NULL_MOVE,
                    was_aborted: true,
                };
            }

            if null_score >= beta {
                // Never return or store unverified mate scores from a null-move search
                let verified_score = if null_score >= MATE_THRESHOLD {
                    beta
                } else {
                    null_score
                };

                self.transposition_table.store(
                    hash,
                    verified_score,
                    depth,
                    ply,
                    self.age,
                    ForwardMove::NULL_MOVE,
                    HashFlag::LOWERBOUND,
                );

                return SearchResult {
                    score: verified_score,
                    best_move: ForwardMove::NULL_MOVE,
                    was_aborted: false,
                };
            }
        }

        let mut legal_moves_played = 0;
        let mut moves_tried: i8 = 0;
        let mut best_move = ForwardMove::NULL_MOVE;   
        let mut best_score = -MATE_VALUE;

        let mut gen_moves = ArrayVec::<ForwardMove, 256>::new();
        self.chess_board.generate_moves(
            &mut gen_moves, pv_move_hint, tt_hint,
            ply, &self.killer_move_table, false)
        ;

        // 6. Move Search Loop
        let see_threshold = if is_pv { -50 * depth as i16 } else { 0 };
        let can_tactical_prune = !king_in_check && depth <= 4;
        let tactical_margin = 35 + (50 * depth as i16);

        let can_lmp = !is_pv && !king_in_check && depth <= 3;
        let lmp_limit = 3 + (depth * depth);
        let ply_idx = (ply as usize) & 127;

        for forward_move in &gen_moves {
            let is_quiet = matches!(forward_move.move_type, MoveFlag::MOVE);
            let is_killer = *forward_move == self.killer_move_table[ply_idx][0]
                || *forward_move == self.killer_move_table[ply_idx][1];

            // --- LATE MOVE PRUNING (LMP): Skip hopeless late quiet moves near the horizon ---
            if can_lmp
                && is_quiet
                && !is_killer
                && moves_tried > lmp_limit
                && alpha.abs() < MATE_THRESHOLD - 256
            {
                continue;
            }

            // --- PRE-MOVE TACTICAL CAPTURE PRUNING ---
            if can_tactical_prune
                && moves_tried > 0
                && alpha.abs() < MATE_THRESHOLD - 256
                && matches!(forward_move.move_type, MoveFlag::CAPTURE | MoveFlag::ENPASSANT)
            {
                let exact_see_score = self.chess_board.static_exchange_evaluation(*forward_move);

                // 1. Strict Tactical Gate: Drop losing exchanges on non-PV nodes (or severe blunders on PV)
                if exact_see_score < see_threshold {
                    continue;
                }

                // 2. Linear Tactical Futility Gate
                if static_eval + exact_see_score + tactical_margin < alpha {
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
                let lmr_eligibility = depth >= 3
                    && moves_tried > 3
                    && !king_in_check
                    && is_quiet
                    && !self.chess_board.is_in_check();

                if lmr_eligibility {
                    let reduced_depth = self.get_reduced_depth(depth, moves_tried, is_pv, is_killer);
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
    fn quiescence_search(&mut self, mut alpha: i16, beta: i16, ply: i16) -> SearchResult {        
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
        if self.is_three_move_repetition(ply) {
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
        let mut tt_hint = ForwardMove::NULL_MOVE;   

        // 3. Tranposition Table
        let tt_entry = self.transposition_table.probe(hash, ply);
        if tt_entry.flag != HashFlag::EMPTY {
            let retrieved_score = tt_entry.score;
            tt_hint = tt_entry.forward_move;

            // EXACT: The true minimax value was found; return it immediately.
            if tt_entry.flag == HashFlag::EXACT {
                return SearchResult { 
                    score: retrieved_score, 
                    best_move: tt_hint,
                    was_aborted: false,
                };
            }

            // LOWER BOUND: The true score is AT LEAST this high. 
            else if tt_entry.flag == HashFlag::LOWERBOUND {
                if retrieved_score >= beta {
                    return SearchResult {
                        score: retrieved_score,
                        best_move: tt_hint,
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
                    best_move: tt_hint,
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
            let static_eval = self.static_eval();
            best_score = static_eval;
            pruning_baseline = static_eval;

            if best_score >= beta {
                let aborted = self.stop_signal.load(Ordering::Relaxed);

                if !aborted {
                    self.transposition_table.store(
                        hash,
                        best_score,
                        Q_DEPTH_MARKER,
                        ply,
                        self.age,
                        ForwardMove::NULL_MOVE,
                        HashFlag::LOWERBOUND,
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
            best_score = -MATE_VALUE + ply;
            pruning_baseline = alpha;
        }

        // Generate tactical moves (or all evasions if in check) onto the stack
        let mut gen_moves = ArrayVec::<ForwardMove, 256>::new();
        let mut legal_moves_played = 0;

        let only_tactical = !king_in_check;
        self.chess_board.generate_moves(
            &mut gen_moves,
            ForwardMove::NULL_MOVE,
            tt_hint,
            ply,
            &self.killer_move_table,
            only_tactical,
        );

        // 5. Quiescence Search (Wider Delta Pruning Margin)
        for forward_move in &gen_moves {
            // --- PRE-MOVE TACTICAL PRUNING GATES ---
            if !king_in_check
                && !matches!(forward_move.move_type, MoveFlag::PROMOTIONQUEEN)
            {
                // 1. Strict Tactical Gate: Static exchange evaluation is exact and NNUE-independent
                let exact_see_score = self.chess_board.static_exchange_evaluation(*forward_move);
                if exact_see_score < 0 {
                    continue;
                }

                // 2. Wider Delta Pruning Gate
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
            let search_result = self.quiescence_search(-beta, -alpha, ply + 1);
            self.process_backward_move();

            if search_result.was_aborted {
                return SearchResult { 
                    score: 0, 
                    best_move: ForwardMove::NULL_MOVE,
                    was_aborted: true,
                };
            }

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

            if king_in_check {
                let terminal_score = -MATE_VALUE + ply;
                
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