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
    
    /// Computes the forward evaluation pass using the current accumulator states.
    /// Returns the final centipawn assessment.
    pub fn evaluate(
        &self, 
        active_player: Side,
        nn: &NnueNetwork, 
        buffer: &mut NnueInferenceBuffer
    ) -> i32 {
        // --- PERSPECTIVE ROUTING ---
        // Side to move (US) always fills the first 256 inputs.
        // Opponent (THEM) always fills the second 256 inputs.
        let (active_acc, opp_acc) = match active_player {
            Side::WHITE => (&self.white, &self.black),
            Side::BLACK => (&self.black, &self.white),
        };

        // --- STEP 1: CONCATENATION & ACTIVATION (L1 -> L2) ---
        // Python Layer 1 scale = 128. Accumulator inputs are 0 or 1.
        for i in 0..256 {
            buffer.l2_inputs[i] = active_acc.vals[i].clamp(0, 127) as i8;
            buffer.l2_inputs[i + 256] = opp_acc.vals[i].clamp(0, 127) as i8;
        }

        // --- STEP 2: HIDDEN LAYER 2 (512 -> 64) ---
        // Row-per-neuron transposed lookup layout maps to 256-bit AVX2 vector pipelines.
        for neuron in 0..64 {
            let mut sum: i32 = nn.l2_biases[neuron];
            let row = &nn.l2_weights[neuron];

            // Process all 512 concatenated inputs across the active board space
            for i in 0..512 {
                sum += (buffer.l2_inputs[i] as i32) * (row[i] as i32);
            }

            // Layer 2 internal sum scale = 4096 (128 * 32).
            // Shift right by 7 (divide by 128) results in a Layer 3 input scale of 32 (4096 / 128).
            let activated = sum >> 7;
            buffer.l3_inputs[neuron] = activated.clamp(0, 127) as i8;
        }

        // --- STEP 3: HIDDEN LAYER 3 (64 -> 32) ---
        for neuron in 0..32 {
            let mut sum: i32 = nn.l3_biases[neuron];
            let row = &nn.l3_weights[neuron];

            for i in 0..64 {
                sum += (buffer.l3_inputs[i] as i32) * (row[i] as i32);
            }

            // Layer 3 internal sum scale = 1024 (32 * 32).
            // Shift right by 5 (divide by 32) preserves precision without clipping signals.
            let activated = sum >> 5; 
            buffer.l4_inputs[neuron] = activated.clamp(0, 127) as i8;
        }

        // --- STEP 4: OUTPUT LAYER (32 -> 1) ---
        let mut final_sum: i32 = nn.output_bias[0];
        let row = &nn.output_weights[0];

        for i in 0..32 {
            final_sum += (buffer.l4_inputs[i] as i32) * (row[i] as i32);
        }

        // Convert this integer range into a standard centipawn metric (where 1.0 pawn = 100 cp):
        // Evaluation = (final_sum * 100) / 4064
        (final_sum * 100) / 4064
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
                // Accumulate features for White's perspective (No rotation)
                let w_idx = get_feature_index(w_king_sq, piece, sq, false);
                for i in 0..256 {
                    self.white.vals[i] += nn.l1_weights[w_idx][i];
                }

                // Accumulate features for Black's perspective (180° rotated perspective)
                let b_idx = get_feature_index(b_king_sq, piece, sq, true);
                for i in 0..256 {
                    self.black.vals[i] += nn.l1_weights[b_idx][i];
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

        // Optional: Capture tracking
        let mut w_cap_idx = None;
        let mut b_cap_idx = None;
        if captured_piece != BoardPiece::NONE {
            w_cap_idx = Some(get_feature_index(w_king_sq, captured_piece, captured_sq, false));
            b_cap_idx = Some(get_feature_index(b_king_sq, captured_piece, captured_sq, true));
        }

        // --- 4. High-Density Auto-Vectorized Parallel Loop Block ---
        for i in 0..256 {
            // White Accumulator Lane Updates
            self.white.vals[i] -= nn.l1_weights[w_remove][i];
            if let Some(w_cap) = w_cap_idx {
                self.white.vals[i] -= nn.l1_weights[w_cap][i];
            }
            self.white.vals[i] += nn.l1_weights[w_add][i];

            // Black Accumulator Lane Updates
            self.black.vals[i] -= nn.l1_weights[b_remove][i];
            if let Some(b_cap) = b_cap_idx {
                self.black.vals[i] -= nn.l1_weights[b_cap][i];
            }
            self.black.vals[i] += nn.l1_weights[b_add][i];
        }
    }

    #[inline(always)]
    pub fn unmake_move(&mut self, 
        nn: &NnueNetwork, 
        mv: UndoMove,
        mailbox: &[BoardPiece; 64], 
        w_king_sq: usize, 
        b_king_sq: usize
    ) {
        // Since bitboards haven't rolled back yet, the piece at the end square is the added piece
        let added_piece: BoardPiece = mailbox[mv.end_sq];

        // --- 1. Identify Original Moving Piece (Reverse Promotion Logic) ---
        // Work backward from the added piece and move flag to find what the piece originally was (always a pawn)
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
        // Reconstruct where the captured piece was located spatially on the matrix.
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
        // To undo, we ADD back the original piece to its starting square, 
        // ADD back the captured piece to its capture square, and REMOVE the piece that reached the destination.
        let w_add_origin = get_feature_index(w_king_sq, move_piece, mv.start_sq, false);
        let b_add_origin = get_feature_index(b_king_sq, move_piece, mv.start_sq, true);

        let w_remove_dest = get_feature_index(w_king_sq, added_piece, mv.end_sq, false);
        let b_remove_dest = get_feature_index(b_king_sq, added_piece, mv.end_sq, true);

        let mut w_cap_idx = None;
        let mut b_cap_idx = None;
        if let Some(cap) = mv.captured_piece {
            if cap != BoardPiece::NONE {
                w_cap_idx = Some(get_feature_index(w_king_sq, cap, captured_sq, false));
                b_cap_idx = Some(get_feature_index(b_king_sq, cap, captured_sq, true));
            }
        }

        // --- 4. Reverse Hardware Update Block (SIMD Auto-Vectorized) ---
        for i in 0..256 {
            // Rollback White Accumulator
            self.white.vals[i] += nn.l1_weights[w_add_origin][i];
            if let Some(w_cap) = w_cap_idx {
                self.white.vals[i] += nn.l1_weights[w_cap][i];
            }
            self.white.vals[i] -= nn.l1_weights[w_remove_dest][i];

            // Rollback Black Accumulator
            self.black.vals[i] += nn.l1_weights[b_add_origin][i];
            if let Some(b_cap) = b_cap_idx {
                self.black.vals[i] += nn.l1_weights[b_cap][i];
            }
            self.black.vals[i] -= nn.l1_weights[b_remove_dest][i];
        }
    }
}