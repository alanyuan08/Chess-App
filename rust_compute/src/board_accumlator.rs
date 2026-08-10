use crate::nnue_network::*;
use crate::move_command::*;

// Retain White / Black Accumulator values across Positions
#[derive(Debug, Clone, Copy)]
pub struct Accumulator {
    pub vals: [i16; 256],
}

#[derive(Debug, Clone, Copy)]
pub struct BoardAccumulators {
    pub white: Accumulator,
    pub black: Accumulator,
}

impl BoardAccumulators {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            white: Accumulator { vals: [0i16; 256] },
            black: Accumulator { vals: [0i16; 256] },
        }
    }

    #[inline(always)]
    fn raw_logit_to_centipawns_i32(self, final_normalized: i32) -> i32 {
        let alpha: f32 = 0.6;
        
        // 1. Calculate the raw float score in centipawns
        let score_f32 = (final_normalized as f32 / alpha) * 100.0;
        
        // 2. Round to the nearest mathematical integer and cast to i32
        score_f32.round() as i32
    }
        
    /// Computes the forward evaluation pass using the current accumulator states.
    /// Returns the final centipawn assessment.
    pub fn evaluate(
        &self, 
        active_player: Side,
        nn: &NnueNetwork, 
        buffer: &mut NnueInferenceBuffer
    ) -> i32 {
        // Clear Buffer 
        buffer.clear();

        // --- PERSPECTIVE ROUTING ---
        // Side to move (US) always fills the first 256 inputs.
        // Opponent (THEM) always fills the second 256 inputs.
        let (active_acc, opp_acc) = match active_player {
            Side::WHITE => (&self.white, &self.black),
            Side::BLACK => (&self.black, &self.white),
        };

        // --- STEP 1: CONCATENATION & ACTIVATION (L1 -> L2) ---
        // Weights multiplied by 128 in Python. 
        // We shift down right here by >> 7 (divide by 128) to reset input baseline scale to 1.0.
        // Clamped at 1 to match Python's SCALE_MAX = 1.0.
        for i in 0..256 {
            let active_val = (active_acc.vals[i] as i32) >> 7;
            buffer.l2_inputs[i] = active_val.clamp(0, 1) as i8;

            let opp_val = (opp_acc.vals[i] as i32) >> 7;
            buffer.l2_inputs[i + 256] = opp_val.clamp(0, 1) as i8;
        }

        // --- STEP 2: HIDDEN LAYER 2 (512 -> 64) ---
        // Weights multiplied by 32 in Python. 
        // Input Scale (1) * Weight Scale (32) = Sum Scale (32).
        for neuron in 0..64 {
            let mut sum: i32 = nn.l2_biases[neuron];
            let row = &nn.l2_weights[neuron];

            for i in 0..512 {
                sum += (buffer.l2_inputs[i] as i32) * (row[i] as i32);
            }

            // Shift down by >> 5 (divide by 32) to reset output baseline scale back to 1.0.
            let activated = sum >> 5;
            buffer.l3_inputs[neuron] = activated.clamp(0, 1) as i8;
        }

        // --- STEP 3: HIDDEN LAYER 3 (64 -> 32) ---
        // Weights multiplied by 32 in Python.
        // Input Scale (1) * Weight Scale (32) = Sum Scale (32).
        for neuron in 0..32 {
            let mut sum: i32 = nn.l3_biases[neuron];
            let row = &nn.l3_weights[neuron];

            for i in 0..64 {
                sum += (buffer.l3_inputs[i] as i32) * (row[i] as i32);
            }

            // Shift down by >> 5 (divide by 32) to reset output baseline scale back to 1.0.
            let activated = sum >> 5; 
            buffer.l4_inputs[neuron] = activated.clamp(0, 1) as i8;
        }

        // --- STEP 4: OUTPUT LAYER (32 -> 1) ---
        // Weights multiplied by 128 in Python.
        // Input Scale (1) * Weight Scale (128) = Sum Scale (128).
        let mut final_sum: i32 = nn.output_bias[0];
        let row = &nn.output_weights[0];

        for i in 0..32 {
            final_sum += (buffer.l4_inputs[i] as i32) * (row[i] as i32);
        }

        // Shift down by >> 7 (divide by 128) to resolve the final output layer scale back to 1.0.
        let final_normalized = final_sum >> 7;

        // Convert the normalized integer score directly into engine centipawns.
        self.raw_logit_to_centipawns_i32(final_normalized)
    }

    /// Re-reads the entire board layout from scratch to perform a full baseline refresh
    pub fn refresh_from_scratch(
        &mut self, 
        nn: &NnueNetwork, 
        mailbox: &[BoardPiece; 64], 
        w_king_sq: usize, 
        b_king_sq: usize
    ) {
        // Reset both accumulators to the initial layer 1 baseline biases
        self.white.vals.copy_from_slice(&nn.l1_biases);
        self.black.vals.copy_from_slice(&nn.l1_biases);

        // Loop through every square on the board and add active pieces
        for sq in 0..64 {
            let piece = mailbox[sq];
            if piece != BoardPiece::NONE {
                let w_idx = get_feature_index(w_king_sq, piece, sq, false);
                let b_idx = get_feature_index(b_king_sq, piece, sq, true);
                
                let w_row = &nn.l1_weights[w_idx];
                let b_row = &nn.l1_weights[b_idx];

                for neuron in 0..256 {
                    // FIXED: Changed 'i' to 'neuron' and mapped cleanly to [feature][neuron]
                    self.white.vals[neuron] += w_row[neuron];
                    self.black.vals[neuron] += b_row[neuron];
                }
            }
        }
    }

    /// Progresses the network forward incrementally during move making
    #[inline(always)]
    pub fn make_move(
        &mut self, 
        nn: &NnueNetwork, 
        mv: ForwardMove,
        mailbox: &[BoardPiece; 64], 
        w_king_sq: usize, 
        b_king_sq: usize
    ) {
        let move_piece: BoardPiece = mailbox[mv.start_sq];

        // --- 1. Identify Target Added Piece (Handles Promotions) ---
        // Default assume the moving piece arrives at the destination unchanged.    
        // For promotion flags, replace the piece with its upgraded target.
        let mut added_piece = move_piece;
        match mv.move_type {
            MoveFlag::PROMOTIONQUEEN => {
                added_piece = if move_piece == BoardPiece::WPAWN { BoardPiece::WQUEEN } else { BoardPiece::BQUEEN };
            }
            MoveFlag::PROMOTIONROOK => {
                added_piece = if move_piece == BoardPiece::WPAWN { BoardPiece::WROOK } else { BoardPiece::BROOK };
            }
            MoveFlag::PROMOTIONBISHOP => {
                added_piece = if move_piece == BoardPiece::WPAWN { BoardPiece::WBISHOP } else { BoardPiece::BBISHOP };
            }
            MoveFlag::PROMOTIONKNIGHT => {
                added_piece = if move_piece == BoardPiece::WPAWN { BoardPiece::WKNIGHT } else { BoardPiece::BKNIGHT };
            }
            _ => {}
        }

        // --- 2. Identify Captured Piece & Coordinate (Handles En Passant) ---
        let (captured_sq, captured_piece) = if mv.move_type == MoveFlag::ENPASSANT {
            let sq = if move_piece == BoardPiece::WPAWN {
                mv.end_sq - 8 // White captures Black pawn hanging behind it
            } else {
                mv.end_sq + 8 // Black captures White pawn hanging behind it
            };
            (sq, mailbox[sq])
        } else {
            (mv.end_sq, mailbox[mv.end_sq])
        };

        // --- 3. Compute Sparse Feature Indices ---
        // Remove old piece location
        let w_remove = get_feature_index(w_king_sq, move_piece, mv.start_sq, false);
        let b_remove = get_feature_index(b_king_sq, move_piece, mv.start_sq, true);

        // Add new piece location
        let w_add = get_feature_index(w_king_sq, added_piece, mv.end_sq, false);
        let b_add = get_feature_index(b_king_sq, added_piece, mv.end_sq, true);

       // Extract raw references to the row arrays for maximum compiler aliasing optimizations
        let w_rem_row = &nn.l1_weights[w_remove];
        let b_rem_row = &nn.l1_weights[b_remove];
        let w_add_row = &nn.l1_weights[w_add];
        let b_add_row = &nn.l1_weights[b_add];

        // --- 4. High-Density Auto-Vectorized Parallel Loop Block ---
        if captured_piece != BoardPiece::NONE {
            let w_cap = get_feature_index(w_king_sq, captured_piece, captured_sq, false);
            let b_cap = get_feature_index(b_king_sq, captured_piece, captured_sq, true);
            
            let w_cap_row = &nn.l1_weights[w_cap];
            let b_cap_row = &nn.l1_weights[b_cap];

            // Branchless, loop-unroll-friendly capture processing
            for neuron in 0..256 {
                self.white.vals[neuron] += w_add_row[neuron] - w_rem_row[neuron] - w_cap_row[neuron];
                self.black.vals[neuron] += b_add_row[neuron] - b_rem_row[neuron] - b_cap_row[neuron];
            }
        } else {
            // Quiet / Non-captures loop (Eliminates branch inside the loop completely)
            for neuron in 0..256 {
                self.white.vals[neuron] += w_add_row[neuron] - w_rem_row[neuron];
                self.black.vals[neuron] += b_add_row[neuron] - b_rem_row[neuron];
            }
        }
    }

    #[inline(always)]
    pub fn unmake_move(
        &mut self, 
        nn: &NnueNetwork, 
        mv: UndoMove,
        mailbox: &[BoardPiece; 64], 
        w_king_sq: usize, 
        b_king_sq: usize
    ) {
        // Since bitboards haven't rolled back yet, the piece at the end square is the added piece
        let added_piece: BoardPiece = mailbox[mv.end_sq];

        // --- 1. Identify Original Moving Piece (Reverse Promotion Logic) ---
        let move_piece = match mv.move_type {
            MoveFlag::PROMOTIONQUEEN | MoveFlag::PROMOTIONROOK | MoveFlag::PROMOTIONBISHOP | MoveFlag::PROMOTIONKNIGHT => {
                if added_piece == BoardPiece::WQUEEN || added_piece == BoardPiece::WROOK || added_piece == BoardPiece::WBISHOP || added_piece == BoardPiece::WKNIGHT {
                    BoardPiece::WPAWN
                } else {
                    BoardPiece::BPAWN
                }
            }
            _ => added_piece,
        };

        // --- 2. Identify Captured Piece Coordinate (Handles En Passant) ---
        let captured_sq = if mv.move_type == MoveFlag::ENPASSANT {
            if move_piece == BoardPiece::WPAWN {
                mv.end_sq - 8 
            } else {
                mv.end_sq + 8 
            }
        } else {
            mv.end_sq
        };

        // --- 3. Compute Sparse Feature Indices ---
        // To undo: we ADD back the original piece to its starting square, 
        // ADD back the captured piece to its capture square, and REMOVE the piece that reached the destination.
        let w_add_origin = get_feature_index(w_king_sq, move_piece, mv.start_sq, false);
        let b_add_origin = get_feature_index(b_king_sq, move_piece, mv.start_sq, true);

        let w_remove_dest = get_feature_index(w_king_sq, added_piece, mv.end_sq, false);
        let b_remove_dest = get_feature_index(b_king_sq, added_piece, mv.end_sq, true);

        // Extract raw references to the row arrays for maximum compiler aliasing optimizations
        let w_add_row = &nn.l1_weights[w_add_origin];
        let b_add_row = &nn.l1_weights[b_add_origin];
        let w_rem_row = &nn.l1_weights[w_remove_dest];
        let b_rem_row = &nn.l1_weights[b_remove_dest];

        // --- 4. High-Density Auto-Vectorized Parallel Loop Block ---
        // Safely extract the optional inner enum capture property
        let cap_piece = mv.captured_piece.unwrap_or(BoardPiece::NONE);

        if cap_piece != BoardPiece::NONE {
            let w_cap = get_feature_index(w_king_sq, cap_piece, captured_sq, false);
            let b_cap = get_feature_index(b_king_sq, cap_piece, captured_sq, true);
            
            let w_cap_row = &nn.l1_weights[w_cap];
            let b_cap_row = &nn.l1_weights[b_cap];

            // Branchless, loop-unroll-friendly capture undo processing
            for neuron in 0..256 {
                self.white.vals[neuron] += w_add_row[neuron] + w_cap_row[neuron] - w_rem_row[neuron];
                self.black.vals[neuron] += b_add_row[neuron] + b_cap_row[neuron] - b_rem_row[neuron];
            }
        } else {
            // Quiet / Non-captures rollback (Eliminates branch inside the loop completely)
            for neuron in 0..256 {
                self.white.vals[neuron] += w_add_row[neuron] - w_rem_row[neuron];
                self.black.vals[neuron] += b_add_row[neuron] - b_rem_row[neuron];
            }
        }
    }
}