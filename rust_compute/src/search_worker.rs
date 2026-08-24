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

    history: [Option<UndoMove>; 1024],
    history_index: usize,

    chess_board: ChessBoard,

    traversed_positions: [u64; 1024],
    position_stack_len: usize,

    killer_move_table: [[Option<ForwardMove>; MAX_DEPTH as usize]; 2],
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
            history: [None; 1024],
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

            killer_move_table: [[None; MAX_DEPTH as usize]; 2],
            thread_id,

            thread_buffer: NnueInferenceBuffer::default(),
            stop_signal,
        }
    }

    // Search Entry Point
     pub fn root_search(&mut self) -> (Option<ForwardMove>, usize, usize){
        // Start the timer
        self.nodes_processed = 0;

        // --- ITERATIVE DEEPENING LOOP ---
        let mut best_move_overall: Option<ForwardMove> = None;
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

        if forward_move.move_type == MoveFlag::NULL {
            self.chess_board.execute_move(forward_move); 

            self.chess_board.null_move_accumulator();

            // Push Move History
            self.push_position();
            self.history[self.history_index] = Some(UndoMove {
                start_sq: 0,
                end_sq: 0,
                move_type: MoveFlag::NULL,
                captured_piece: None,
                prev_castle_rights,
                prev_en_passant,
            });
            self.history_index += 1;
        } else {
            let move_piece = self.chess_board.mailbox_piece(forward_move.start_sq);
            let is_king_or_castle = move_piece == BoardPiece::WKING 
                || move_piece == BoardPiece::BKING 
                || forward_move.move_type == MoveFlag::KINGSIDECASTLE 
                || forward_move.move_type == MoveFlag::QUEENSIDECASTLE;

            if !is_king_or_castle {
                self.chess_board.make_move(forward_move);
            }

            let remove_piece = self.chess_board.execute_move(forward_move);

            // Forced Recompute due to King
            if is_king_or_castle {
                self.chess_board.create_accumlator_from_scratch();
            }

            // Push Move History
            self.push_position();
            self.history[self.history_index] = Some(UndoMove {
                    start_sq: forward_move.start_sq,
                    end_sq: forward_move.end_sq,
                    move_type: forward_move.move_type,
                    captured_piece: remove_piece,
                    prev_castle_rights,
                    prev_en_passant,
                });
            self.history_index += 1;
        }
    }

    fn process_backward_move(&mut self) {
        if self.history_index == 0 {
            return;
        }

        self.history_index -= 1;
        if let Some(undo_move) = self.history[self.history_index].take() {
            // Timecat Undo
            self.chess_board.unmake_move();

            // ChessBoard Undo Move
            self.chess_board.unexecute_move(undo_move);
            self.pop_position();
        }
    }

    // A simple, linear-logarithmic approximation using integer division:
    // R increases slowly as depth and move count grow.
    fn calculate_lmr_reduction(&mut self, depth: i32, moves_tried: i32) -> i32 {
        // Clamp depth to 0..=64
        let d = depth.clamp(0, 63) as usize;
        
        // Clamp moves_tried to 0..=64
        let m = moves_tried.clamp(0, 63) as usize;

        let base_reduction = LMR_TABLE[d][m];

        if self.thread_id == 0 {
            base_reduction
        } else {
            let thread_offset = if self.thread_id % 2 == 1 { 1 } else { -1 };
            let total_reduction = (base_reduction + thread_offset).max(0);

            if total_reduction >= depth {
                depth - 1
            } else {
                total_reduction
            }
        }
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

    // StockFish NMP Reduction algorithm
    fn calculate_nmp_reduction(&self, depth: i32, static_eval: i32, beta: i32) -> i32 {
        let base_reduction = 3 + (depth / 4);
        let eval_bonus = ((beta - static_eval) / 200).clamp(0, 2);
        
        base_reduction + eval_bonus
    }

    // NMP Static Eval to determine cuttoff eligability
    fn static_eval(&self) -> i32 {
        let static_white = (self.chess_board.rooks[0].count_ones() as i32 * 500)
            + (self.chess_board.knights[0].count_ones() as i32 * 300) 
            + (self.chess_board.bishops[0].count_ones() as i32 * 300) 
            + (self.chess_board.pawns[0].count_ones() as i32 * 100) 
            + (self.chess_board.queens[0].count_ones() as i32 * 900);

        let static_black = (self.chess_board.rooks[1].count_ones() as i32 * 500)
            + (self.chess_board.knights[1].count_ones() as i32 * 300) 
            + (self.chess_board.bishops[1].count_ones() as i32 * 300) 
            + (self.chess_board.pawns[1].count_ones() as i32 * 100) 
            + (self.chess_board.queens[1].count_ones() as i32 * 900);
            
        // 3. Return score relative to whoever's turn it is right now
        match self.chess_board.active_player() {
            Side::WHITE => static_white - static_black,
            Side::BLACK => static_black - static_white,
        }
    }

    // Process Negamax
    fn negamax(&mut self, depth: i32, ply: i32, mut alpha: i32, mut beta: i32, 
        mut pv_move_hint: Option<ForwardMove>, allow_null: bool) -> SearchResult {

        // Halt Signal
        if self.nodes_processed & 0x3FFF == 0 && self.stop_signal.load(Ordering::Relaxed) {
            return SearchResult { score: 0, best_move: None, was_aborted: true };
        }

        // Nodes Processed
        self.nodes_processed += 1;

        // Original Alpha for Transposition Table
        let original_alpha = alpha;
        
        // Three Move Repetition Draw
        if self.is_three_move_repetition() {
            return SearchResult {
                score: 0,
                best_move: None,
                was_aborted: false,
            };
        }

        let hash = self.chess_board.zobrist_hash();
        let mut tt_move = None;

        if let Some(tt_entry) = self.transposition_table.probe(hash, ply) {
            let retrieved_score: i32 = tt_entry.score as i32;
            let retrieved_depth: i32 = tt_entry.depth as i32;
            
            if tt_entry.move_id != 0 {
                let mut mv = ForwardMove::unpack(tt_entry.move_id);
                mv.pv_score = -2_000_000;
                tt_move = Some(mv);

                if pv_move_hint.is_none() {
                    pv_move_hint = tt_move;
                }
            };

            if retrieved_depth >= depth {
                // EXACT: The true minimax value was found; return it immediately.
                if tt_entry.flag == HashFlag::EXACT {
                    return SearchResult {
                        score: retrieved_score,
                        best_move: tt_move.or(pv_move_hint),
                        was_aborted: false,
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
                        best_move: tt_move.or(pv_move_hint),
                        was_aborted: false,
                    };
                }
            }
        }

        // Leaf Node Condition -> Drop into Quiescence Search
        if depth == 0 {
            let q_score = self.quiescence_search(alpha, beta, ply, -1);
            let aborted = self.stop_signal.load(Ordering::Relaxed);

            return SearchResult {
                score: if aborted { 0 } else { q_score },
                best_move: None,
                was_aborted: aborted,
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

        // Null Move Pruning
        if allow_null && !king_in_check && depth >= 3 && 
            self.chess_board.has_major_pieces() && ply > 0 &&
            beta < MATE_THRESHOLD && beta > -MATE_THRESHOLD {

            let static_eval = self.static_eval();
            if static_eval >= beta {
                
                // Calculate Reduction
                let reduction = self.calculate_nmp_reduction(depth, static_eval, beta);
                let next_depth = (depth - reduction).max(0); 

                // Make the null move (switch sides, update en-passant/hash keys)
                let null_move = ForwardMove { 
                    start_sq: 0, end_sq: 0, 
                    move_type: MoveFlag::NULL, pv_score: 0
                };
                
                self.process_forward_move(null_move);
                
                let null_result = self.negamax(next_depth, ply + 1, -beta, -beta + 1, None, false);
                let mut null_score = -null_result.score;
                
                self.process_backward_move();

                // Was Aborted
                if null_result.was_aborted {
                    return SearchResult { score: 0, best_move: None, was_aborted: true };
                }

                if null_score >= MATE_THRESHOLD {
                    null_score = beta;
                }
                
                // Fail-high cutoff: The position is so good we can prune it completely
                if null_score >= beta {
                    return SearchResult { score: beta, best_move: None, was_aborted: false };
                }
            }
        }

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
            moves_tried += 1;

            // Check LMR Eligibility
            lmr_eligibility = false;
            if depth >= 3 && moves_tried > 2 && !king_in_check && matches!(forward_move.move_type, MoveFlag::MOVE) {
                lmr_eligibility = true;
            }

            // LMR Reduction
            let mut negamax_result;
            if lmr_eligibility {
                let reduced_depth = self.calculate_lmr_reduction(depth, moves_tried);

                negamax_result = self.negamax(reduced_depth, ply + 1,  -alpha - 1, -alpha, None, true);

                if -negamax_result.score > alpha {
                    negamax_result = self.negamax(depth - 1, ply + 1, -beta, -alpha, None, true);
                }
            } else {
                negamax_result = self.negamax(depth - 1, ply + 1, -beta, -alpha, None, true);
            }

            let score = -negamax_result.score;

            // Undo Move + TimeCat
            self.process_backward_move();

            // Was Aborted
            if negamax_result.was_aborted {
                return SearchResult { score: 0, best_move: None, was_aborted: true };
            }

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

                return SearchResult { score: best_score, best_move, was_aborted: false };
            }

            if score > alpha {
                alpha = score;
            }

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
                    best_move: None,
                    was_aborted: false
                };
            } else {
                self.transposition_table.store(
                    hash, 0, ply, None, MAX_DEPTH, HashFlag::EXACT
                );

                // Stalemate
                return SearchResult { 
                    score: 0, 
                    best_move: None,
                    was_aborted: false,
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
        SearchResult { score: best_score, best_move, was_aborted: false }
    }

    fn board_eval(&mut self) -> i32 {
        self.chess_board.evaluate(
            &mut self.thread_buffer
        )
    }

    // Quiescence Search 
    fn quiescence_search(&mut self, mut alpha: i32, mut beta: i32, ply: i32, depth: i32) -> i32 {        
        if self.nodes_processed & 0x3FFF == 0 && self.stop_signal.load(Ordering::Relaxed) {
            return alpha; 
        }

        // Nodes Processed
        self.nodes_processed += 1;

        // Three Move Repetition Draw
        if self.is_three_move_repetition() {
            return 0;
        }

        let mut pv_move_hint = None;
        let hash = self.chess_board.zobrist_hash();

        const Q_DEPTH_MARKER: i32 = -1;
        if let Some(tt_entry) = self.transposition_table.probe(hash, ply) {
            let retrieved_score: i32 = tt_entry.score as i32;
            let retrieved_depth: i32 = tt_entry.depth as i32;
            
            if tt_entry.move_id != 0 {
                let mut mv = ForwardMove::unpack(tt_entry.move_id);
                mv.pv_score = -2_000_000;
                pv_move_hint = Some(mv);
            };

            if retrieved_depth >= Q_DEPTH_MARKER {
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

        let static_eval = self.board_eval();

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

            // Negamax search call
            let score = -self.quiescence_search(-beta, -alpha, ply + 1, depth - 1);
            
            // Undo Move + TimeCat
            self.process_backward_move();

            if self.stop_signal.load(Ordering::Relaxed) {
                return alpha;
            }

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

        if self.stop_signal.load(Ordering::Relaxed) {
            return alpha; 
        }

        // Store evaluation state inside the Transposition Table
        self.transposition_table.store(hash, best_score, ply, best_move, Q_DEPTH_MARKER, hash_flag);
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