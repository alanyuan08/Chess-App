use std::sync::atomic::{AtomicBool, Ordering};
use crate::transposition_table::*;
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

    history: [UndoMove; 1024],
    history_index: usize,

    chess_board: ChessBoard,

    traversed_positions: [u64; 1024],
    position_stack_len: usize,

    killer_move_table: [[ForwardMove; 2]; 128],
    thread_id: usize,

    thread_buffer: NnueInferenceBuffer,
    stop_signal: Arc<AtomicBool>
}

impl SearchWorker {
    pub fn new(
        transposition_table: Arc<TranspositionTable>,
        nnue_network: &'static NnueNetwork,
        stop_signal: Arc<AtomicBool>,
        thread_id: usize
    ) -> Self {
        Self {
            history: [UndoMove::NULL_UNDO_MOVE; 1024],
            history_index: 0,

            chess_board: {
                let mut chess_board = ChessBoard::new(
                    nnue_network
                );
                chess_board.init_board();
                chess_board
            },

            traversed_positions: {
                [0; 1024]
            },
            position_stack_len: 0, 
            
            nodes_processed: 0,
            transposition_table,

            killer_move_table: [[ForwardMove::NULL_MOVE; 2]; 128],
            thread_id,

            thread_buffer: NnueInferenceBuffer::default(),
            stop_signal,
        }
    }

    // Search Entry Point
    pub fn root_search(&mut self) -> (ForwardMove, usize, usize){
        // Start the timer
        self.nodes_processed = 0;

        // --- ITERATIVE DEEPENING LOOP ---
        let mut best_move_overall: ForwardMove = ForwardMove::NULL_MOVE;
        let mut depth = 1;

        while depth <= MAX_DEPTH {
            let result = self.negamax(depth, 0, -INFINITY, INFINITY, best_move_overall, true);
            
            if !self.stop_signal.load(Ordering::Relaxed){
                if self.thread_id == 0 {
                    best_move_overall = result.best_move;
                    println!("[Master Thread] Completed Depth: {} | Best Move Score: {:?}", 
                        depth, best_move_overall);
                    if depth >= PV_DEPTH { 
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
        if self.history_index >= 1024 { 
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
            self.chess_board.null_move_accumulator();
        } else {
            let move_piece = self.chess_board.mailbox_piece(forward_move.start_sq);
            let is_king_or_castle = (move_piece == BoardPiece::WKING || move_piece == BoardPiece::BKING)
                || (forward_move.move_type == MoveFlag::KINGSIDECASTLE || forward_move.move_type == MoveFlag::QUEENSIDECASTLE);

            if !is_king_or_castle {
                self.chess_board.make_move(forward_move);
            }

            captured_piece = self.chess_board.execute_move(forward_move);

            // Forced Recompute due to King
            if is_king_or_castle {
                self.chess_board.create_accumlator_from_scratch();
            }
        }

        let undo_state = UndoMove {
            start_sq: forward_move.start_sq,
            end_sq: forward_move.end_sq,
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
    fn get_reduced_depth(&self, depth: i32, moves_tried: i32) -> i32 {
        // 1. Safe array bounds clamping
        let d = depth.clamp(0, 63) as usize;
        let m = moves_tried.clamp(0, 63) as usize;

        let base_reduction = LMR_TABLE[d][m];
        let mut reduction = base_reduction;

        // 2. Apply thread diversity (helps different threads explore different paths)
        if self.thread_id != 0 {
            let thread_offset = if self.thread_id % 2 == 1 { 1 } else { -1 };
            reduction = (base_reduction + thread_offset).max(0);
        }

        // 3. Calculate the target depth
        let mut reduced_depth = depth - 1 - reduction;

        // 4. Safety lock: Ensure we always search at least depth 1,
        // and never search deeper than a standard depth step (depth - 1)
        if reduced_depth < 0 {
            reduced_depth = 0;
        } else if reduced_depth >= depth {
            reduced_depth = depth - 1;
        }

        reduced_depth
    }

    // Store Killer Move - No Captures
    fn store_killer_move(&mut self, new_killer_move: ForwardMove, ply: i32) {
        // Convert the incoming ply directly into our safe array index context
        let ply_idx = ply as usize;

        // Safety guard to prevent out-of-bounds array crashes if your search goes extremely deep
        if ply_idx >= 128 {
            return;
        }

        // If this move is already our primary killer slot for this ply, do nothing
        if self.killer_move_table[ply_idx][0] == new_killer_move {
            return;
        }
        
        // Shift slot 0 down into slot 1 to preserve it, then make this the primary killer move
        self.killer_move_table[ply_idx][1] = self.killer_move_table[ply_idx][0];
        self.killer_move_table[ply_idx][0] = new_killer_move;
    }

    // StockFish NMP Reduction algorithm
    #[inline]
    fn calculate_nmp_reduction(&self, depth: i32, static_eval: i32, beta: i32) -> i32 {
        let base_reduction = 3 + (depth / 4);
        
        let eval_bonus = ((static_eval - beta) / 300).clamp(0, 3);
        
        base_reduction + eval_bonus
    }

    // Static Eval
    #[inline]
    fn static_eval(&mut self) -> i32 {
        self.chess_board.evaluate(&mut self.thread_buffer)
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

    // Fail-Soft -> Return the first value that breaches alpha - beta -> Not Guaranteed to be best Value
    // Hard Cut-Off -> Once a value causes alpha >= beta, stop the search
    fn negamax(&mut self, depth: i32, ply: i32, mut alpha: i32, mut beta: i32, 
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
            let retrieved_score: i32 = tt_entry.score as i32;
            let retrieved_depth: i32 = tt_entry.depth as i32;
            
            // Transposition Move > PV Hint
            if tt_entry.move_id != 0 {
                let mut tt_move = ForwardMove::unpack(tt_entry.move_id);
                tt_move.pv_score = -2_000_000;
                pv_move_hint = tt_move;
            };

            if retrieved_depth >= depth {
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
                    alpha = cmp::max(alpha, retrieved_score);

                    // Fail-Soft Cutoff
                    if alpha >= beta {
                        return SearchResult {
                            score: alpha,
                            best_move: pv_move_hint,
                            was_aborted: false,
                        };
                    }
                }
                // UPPER BOUND: The true score is AT MOST this high.
                else if tt_entry.flag == HashFlag::UPPERBOUND {
                    beta = cmp::min(beta, retrieved_score);

                    // Fail-Soft Cutoff
                    if alpha >= beta {
                        return SearchResult {
                            score: beta,
                            best_move: pv_move_hint,
                            was_aborted: false,
                        };
                    }
                }  
            }
        }

        // Leaf Node Condition -> Drop into Quiescence Search
        // Q-Search uses Hard-Cut Off and the value is bounded between alpha - beta
        if depth == 0 {
            return self.quiescence_search(alpha, beta, ply);
        }

        let static_eval = self.static_eval();

        // 4. Late Futility Pruning
        if depth <= 3 && !king_in_check && beta < MATE_THRESHOLD && beta > -MATE_THRESHOLD {
            // Flat evaluation tuning factor: 40-70 centipawns per depth works well for Stockfish scale
            let rfp_margin = 60 * depth;
            let computed_val = static_eval - rfp_margin;
            if computed_val >= beta {
                if beta == original_beta {
                    self.transposition_table.store(
                        hash, computed_val, ply, ForwardMove::NULL_MOVE, depth, HashFlag::LOWERBOUND
                    );
                }

                return SearchResult {
                    score: computed_val,
                    best_move: ForwardMove::NULL_MOVE,
                    was_aborted: false,
                };
            }
        }

        // 5. Pure Fail-Soft Null Move Pruning
        let mut lmr_eligibility;
        let mut moves_tried: i32 = 0;
        if allow_null && !king_in_check && depth >= 3 && 
            self.chess_board.has_major_pieces() && ply > 0 &&
            beta < MATE_THRESHOLD && beta > -MATE_THRESHOLD {

            let static_eval = self.static_eval();
            if static_eval >= beta {
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
                
                // True Fail-Soft NMP Break
                if null_score >= beta {
                    if self.stop_signal.load(Ordering::Relaxed) {
                         return SearchResult { 
                            score: 0, 
                            best_move: ForwardMove::NULL_MOVE, 
                            was_aborted: true 
                        };
                    }

                    if beta == original_beta {
                        self.transposition_table.store(
                            hash, null_score, ply, ForwardMove::NULL_MOVE, depth, HashFlag::LOWERBOUND
                        );
                    }

                    return SearchResult { 
                        score: null_score, 
                        best_move: ForwardMove::NULL_MOVE, 
                        was_aborted: false 
                    };
                }
            }
        }

        let mut legal_moves_played = 0;
        let mut best_move = ForwardMove::NULL_MOVE;   
        let mut best_score = -INFINITY;

        let mut gen_moves = ArrayVec::<ForwardMove, 256>::new();
        self.chess_board.generate_moves(&mut gen_moves, pv_move_hint, 
            ply, &self.killer_move_table, false);

        // 6. Move Search Loop
        for forward_move in &gen_moves {
            self.process_forward_move(*forward_move);

            // Psuedo legal move exposes check, undo move
            if self.chess_board.is_previous_player_king_in_check() {    
                self.process_backward_move();
                continue;
            }

            legal_moves_played += 1;

            // Late Move Reduction
            moves_tried += 1;
            lmr_eligibility = false;
            if depth >= 3 && moves_tried > 4 && !king_in_check && matches!(forward_move.move_type, MoveFlag::MOVE) {
                lmr_eligibility = true;
            }

            let mut negamax_result;
            if lmr_eligibility {
                let reduced_depth = self.get_reduced_depth(depth, moves_tried);
                negamax_result = self.negamax(reduced_depth, ply + 1, -alpha - 1, -alpha, ForwardMove::NULL_MOVE, true);
                
                // If the reduced search failed high, we must re-search at full depth
                if !negamax_result.was_aborted && -negamax_result.score >= alpha {
                    negamax_result = self.negamax(depth - 1, ply + 1, -beta, -alpha, ForwardMove::NULL_MOVE, true);
                }
            } else {
                // Normal search without reduction
                negamax_result = self.negamax(depth - 1, ply + 1, -beta, -alpha, ForwardMove::NULL_MOVE, true);
            }

            let score = -negamax_result.score;
            self.process_backward_move();

            if negamax_result.was_aborted {
                return SearchResult { 
                    score: 0, 
                    best_move: ForwardMove::NULL_MOVE, 
                    was_aborted: true 
                };
            }

            if score > best_score {
                best_score = score;
                best_move = *forward_move;

                if score > alpha {
                    alpha = score;

                    if score >= beta {
                        if matches!(best_move.move_type, MoveFlag::MOVE) {
                            self.store_killer_move(best_move, ply); 
                        }
                        break;
                    }
                }
            }
        }

        // 4. Handle terminal nodes cleanly if no legal moves exist
        if legal_moves_played == 0 {
            let terminal_score = if king_in_check {
                -MATE_VALUE + ply
            } else {
                0
            };

            self.transposition_table.store(
                hash, terminal_score, ply, ForwardMove::NULL_MOVE, depth, HashFlag::EXACT
            );

            return SearchResult { 
                score: terminal_score, 
                best_move: ForwardMove::NULL_MOVE,
                was_aborted: false
            };
        }

        if self.stop_signal.load(Ordering::Relaxed) {
            return SearchResult { 
                score: 0, 
                best_move: ForwardMove::NULL_MOVE, 
                was_aborted: true 
            };
        }

        // Pure Fail-Soft TT Storage and Return Contract
        let final_hash_flag = if best_score >= original_beta {
            HashFlag::LOWERBOUND
        } else if best_score > original_alpha {
            // Fail-Soft / Hard-Cut off can only be 100% Certain if alpha / beta didn't change. 
            if alpha != original_alpha || beta != original_beta {
                if beta != original_beta { 
                    HashFlag::LOWERBOUND 
                } else { 
                    HashFlag::UPPERBOUND 
                }
            } else {
                HashFlag::EXACT
            }
        } else {
            HashFlag::UPPERBOUND
        };
        self.transposition_table.store(
            hash, best_score, ply, best_move, depth, final_hash_flag
        );
        SearchResult { 
            score: best_score, 
            best_move, 
            was_aborted: false 
        }
    }

    // Quiescence Search -> Q-Search only uses Score and was_aborted. 
    // Fail-Soft Framework 
    fn quiescence_search(&mut self, mut alpha: i32, mut beta: i32, ply: i32) -> SearchResult {        
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
        let mut pv_move_hint = ForwardMove::NULL_MOVE;
        const Q_DEPTH_MARKER: i32 = -1;
        let original_alpha = alpha;
        let original_beta = beta;

        // 3. Tranposition Table
        let tt_entry = self.transposition_table.probe(hash, ply);
        if tt_entry.flag != HashFlag::EMPTY {
            let retrieved_score: i32 = tt_entry.score as i32;
            let retrieved_depth: i32 = tt_entry.depth as i32;
            
            if tt_entry.move_id != 0 {
                let mut tt_move = ForwardMove::unpack(tt_entry.move_id);
                tt_move.pv_score = -2_000_000;
                pv_move_hint = tt_move;
            };

            if retrieved_depth >= Q_DEPTH_MARKER {
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
                    alpha = cmp::max(alpha, retrieved_score);

                    // Fail-Soft Cutoff
                    if alpha >= beta {
                        return SearchResult {
                            score: alpha,
                            best_move: pv_move_hint,
                            was_aborted: false,
                        };
                    }
                }
                // UPPER BOUND: The true score is AT MOST this high.
                else if tt_entry.flag == HashFlag::UPPERBOUND {
                    beta = cmp::min(beta, retrieved_score);

                    // Fail-Soft Cutoff
                    if alpha >= beta {
                        return SearchResult {
                            score: beta,
                            best_move: pv_move_hint,
                            was_aborted: false,
                        };
                    }
                }  
            }
        }

        // 4. Standing Pat 
        let king_in_check = self.chess_board.is_in_check(); 
        let mut best_score;
        let mut best_move = ForwardMove::NULL_MOVE;

        if !king_in_check {
            let static_eval = self.static_eval();
            best_score = static_eval;

            // Fail-Soft Standing Pat Cutoff: Return the actual score that broke beta
            if best_score >= original_beta {
                self.transposition_table.store(
                    hash, best_score, ply, ForwardMove::NULL_MOVE, Q_DEPTH_MARKER, HashFlag::LOWERBOUND
                );
                return SearchResult {
                    score: best_score, 
                    best_move: ForwardMove::NULL_MOVE,
                    was_aborted: false,
                };
            }

            if best_score > alpha {
                alpha = best_score;
                if alpha >= beta {
                    if beta == original_beta {
                        self.transposition_table.store(
                            hash, best_score, ply, ForwardMove::NULL_MOVE, Q_DEPTH_MARKER, HashFlag::LOWERBOUND
                        );
                    }

                    return SearchResult {
                        score: best_score,
                        best_move: ForwardMove::NULL_MOVE,
                        was_aborted: false,
                    };
                }
            }
        } else {
            // Safe checkmate buffer baseline prevents the engine from masking mate strings
            best_score = -MATE_VALUE + ply;
        }   

        // Generate strictly legal tactical moves directly onto the global stack
        let mut gen_moves = ArrayVec::<ForwardMove, 256>::new();
        let mut legal_moves_played = 0;

        let only_tactical = !king_in_check;
        self.chess_board.generate_moves(&mut gen_moves, pv_move_hint, 
            ply, &self.killer_move_table, only_tactical);

        // 5. Quiscence Search
        for forward_move in &gen_moves {
            if !king_in_check {
                // 1. Fetch victim value (0 if it's a quiet promotion)
                let victim_type = self.chess_board.mailbox_piece(forward_move.end_sq);
                let victim_value = if victim_type == BoardPiece::NONE { 0 } else { piece_value(victim_type) };
                
                // 2. Add the massive material bonus if a pawn transforms into a Queen
                let promotion_bonus = if forward_move.move_type == MoveFlag::PROMOTIONQUEEN {
                    800 
                } else {
                    0
                };

                // 3 Compute Gain
                let max_possible_gain = victim_value + promotion_bonus;
                if best_score + max_possible_gain + 200 < alpha {
                    continue; 
                }
            }

            // Push move (handles UCI, board state, hash, and history internally)
            self.process_forward_move(*forward_move);
            
            // Psuedo legal move exposes check, undo move
            if self.chess_board.is_previous_player_king_in_check() {
                self.process_backward_move();
                continue;
            }

            legal_moves_played += 1;
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

                if score > alpha {
                    alpha = score;

                    // Fail-Soft Cutoff Loop
                    if score >= beta { 
                        break;
                    }
                }
            }
        }

        // 6. Checkmate/Stalemate Terminal Node Resolutions
        if legal_moves_played == 0 {
            let terminal_score = if king_in_check {
                -MATE_VALUE + ply
            } else {
                0
            };

            self.transposition_table.store(
                hash, terminal_score, ply, ForwardMove::NULL_MOVE, Q_DEPTH_MARKER, HashFlag::EXACT
            );

            return SearchResult { 
                score: terminal_score,
                best_move: ForwardMove::NULL_MOVE, 
                was_aborted: false 
            };
        }

        if self.stop_signal.load(Ordering::Relaxed) {
            return SearchResult { 
                score: 0, 
                best_move: ForwardMove::NULL_MOVE, 
                was_aborted: true 
            };
        }

        // 7. Pure Fail-Soft Transposition Storage and Return Contract
        let final_hash_flag = if best_score >= original_beta {
            // We found a Score that exceeds the original_beta - We also want to use original_beta not beta to set a tigher LowerBound
            HashFlag::LOWERBOUND
        } else if best_score > original_alpha {
            // Fail-Soft / Hard-Cut off can only be 100% Certain if alpha / beta didn't change. 
            if alpha != original_alpha || beta != original_beta {
                if beta != original_beta { 
                    HashFlag::LOWERBOUND 
                } else { 
                    HashFlag::UPPERBOUND 
                }
            } else {
                HashFlag::EXACT
            }
        } else {
            HashFlag::UPPERBOUND
        };

        self.transposition_table.store(
            hash, best_score, ply, best_move, Q_DEPTH_MARKER, final_hash_flag
        );

        SearchResult { 
            score: best_score, 
            best_move,
            was_aborted: false,
        }
    }
}