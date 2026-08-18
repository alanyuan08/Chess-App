use crate::nnue_network::*;
use crate::move_command::*;

// Retain White / Black Accumulator values across Positions
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct Accumulator {
    pub vals: [i32; 256],
}

#[derive(Debug, Clone, Copy)]
pub struct BoardAccumulators {
    pub white: Accumulator,
    pub black: Accumulator,
}


impl Default for BoardAccumulators {
    fn default() -> Self {
        Self {
            white: Accumulator { vals: [0i32; 256] },
            black: Accumulator { vals: [0i32; 256] },
        }
    }
}

impl BoardAccumulators {
    /// Computes the forward evaluation pass using the current accumulator states.
    /// Returns the final centipawn assessment.
    pub fn evaluate(
        &self, 
        active_player: Side,
        nn: &'static NnueNetwork, 
        buffer: &mut NnueInferenceBuffer
    ) -> i32 {
        // --- PERSPECTIVE ROUTING ---
        // Side to move (US) always fills the first 256 inputs.
        // Opponent (THEM) always fills the second 256 inputs.
        let (active_acc, opp_acc) = match active_player {
            Side::WHITE => (&self.white, &self.black),
            Side::BLACK => (&self.black, &self.white),
        };

        // --- STEP 0: ACCUMLATOR (Input -> Accumlator) ---
        // The accumlator is maintained by the init / move functions

        // --- STEP 1: CONCATENATION & ACTIVATION (L1 -> L2) ---
        for (i, &val) in active_acc.vals.iter().enumerate().take(256) {
            buffer.l2_inputs[i] = val.clamp(0, 128) as i16;
        }

        for (i, &val) in opp_acc.vals.iter().enumerate().take(256) {
            buffer.l2_inputs[i + 256] = val.clamp(0, 128) as i16;
        }

        // --- STEP 2: HIDDEN LAYER 2 (512 -> 64) ---
        // Input Scale (128) * Weight Scale (32) = Sum Scale (4096).
        // Shift Down by >> 7 to Scale (32)
        // Clamp at 32 to match Python's ReLU1 (1.0).
        let l2_layer = nn.l2_weights.iter().zip(nn.l2_biases.iter());
        for (neuron, (row, &bias)) in l2_layer.enumerate().take(64) {
            let mut sum: i32 = bias;

            // Process chunks of 16 elements to enable aggressive SIMD auto-vectorization
            let inputs = &buffer.l2_inputs[..512];
            for (chunk_weights, chunk_inputs) in row.chunks(16).zip(inputs.chunks(16)) {
                for (&w, &inp) in chunk_weights.iter().zip(chunk_inputs.iter()) {
                    sum += (inp as i32) * (w as i32);
                }
            }

            let activated = sum >> 7;
            buffer.l3_inputs[neuron] = activated.clamp(0, 32) as i16;
        }

        // --- STEP 3: HIDDEN LAYER 3 (64 -> 32) ---
        // Input Scale (32) * Weight Scale (32) = Sum Scale (1024).
        // Shift Down by 5 to Scale (32)
        // Clamp at 32 to match Python's ReLU1 (1.0).
        let l3_layer = nn.l3_weights.iter().zip(nn.l3_biases.iter());
        for (neuron, (row, &bias)) in l3_layer.enumerate().take(32) {
            let mut sum: i32 = bias;

            let inputs = &buffer.l3_inputs[..64];
            for (chunk_weights, chunk_inputs) in row.chunks(16).zip(inputs.chunks(16)) {
                for (&w, &inp) in chunk_weights.iter().zip(chunk_inputs.iter()) {
                    sum += (inp as i32) * (w as i32);
                }
            }

            let activated = sum >> 5;
            buffer.l4_inputs[neuron] = activated.clamp(0, 32) as i16;
        }


        // --- STEP 4: OUTPUT LAYER (32 -> 1) ---
        // Input Scale (32) * Weight Scale (128) = Sum Scale (4096).
        // Shift Down by 5 to Scale (128)
        let mut final_sum: i32 = nn.output_bias[0];
        let row = &nn.output_weights[0];

        let inputs = &buffer.l4_inputs[..32];
        for (chunk_weights, chunk_inputs) in row.chunks(16).zip(inputs.chunks(16)) {
            for (&w, &inp) in chunk_weights.iter().zip(chunk_inputs.iter()) {
                final_sum += (inp as i32) * (w as i32);
            }
        }
        let internal_pawns_scaled = final_sum >> 5;

        // Shift by >> 7 remove remaining scale
        (internal_pawns_scaled * 100) / 128
    }

    /// Re-reads the entire board layout from scratch to perform a full baseline refresh
    pub fn refresh_from_scratch(
        &mut self, 
        nn: &'static NnueNetwork, 
        mailbox: &[BoardPiece; 64], 
        w_king_sq: usize, 
        b_king_sq: usize
    ) {
        // 1. Reset both accumulators to the initial layer 1 baseline biases (Casting i16 to i32)
        // Slicing to [..256] removes the compiler's bounds-checking overhead
        let target_white = &mut self.white.vals[..256];
        let target_black = &mut self.black.vals[..256];
        let biases = &nn.l1_biases[..256];

        for i in 0..256 {
            target_white[i] = biases[i] as i32;
            target_black[i] = biases[i] as i32;
        }

        // 2. Loop through every square on the board and add active pieces
        for (sq, &piece) in mailbox.iter().enumerate().take(64) {
            if piece != BoardPiece::NONE {
                // Get the unique HalfKA indices for both king perspectives
                let w_idx = get_feature_index(w_king_sq, piece, sq, false);
                let b_idx = get_feature_index(b_king_sq, piece, sq, true);
                
                // Grab direct references to the row weights and explicitly slice them to 256
                let w_row = &nn.l1_weights[w_idx][..256];
                let b_row = &nn.l1_weights[b_idx][..256];

                // 3. Unroll the nested zip into a clean, contiguous loop.
                // The exact bounds match allows the compiler to confidently auto-vectorize this loop.
                for i in 0..256 {
                    target_white[i] += w_row[i] as i32; 
                    target_black[i] += b_row[i] as i32; 
                }
            }
        }
    }

    /// Progresses the network forward incrementally during move making
    #[inline(always)]
    pub fn make_move(
        &mut self, 
        nn: &'static NnueNetwork, 
        mv: ForwardMove,
        mailbox: &[BoardPiece; 64], 
        w_king_sq: usize, 
        b_king_sq: usize
    ) {
        let move_piece: BoardPiece = mailbox[mv.start_sq];

        // --- 1. Identify Target Added Piece (Handles Promotions) ---
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
                mv.end_sq - 8 
            } else {
                mv.end_sq + 8 
            };
            (sq, mailbox[sq])
        } else {
            (mv.end_sq, mailbox[mv.end_sq])
        };

        // --- 3. Compute Sparse Feature Indices ---
        let w_remove = get_feature_index(w_king_sq, move_piece, mv.start_sq, false);
        let b_remove = get_feature_index(b_king_sq, move_piece, mv.start_sq, true);

        let w_add = get_feature_index(w_king_sq, added_piece, mv.end_sq, false);
        let b_add = get_feature_index(b_king_sq, added_piece, mv.end_sq, true);

        // Get basic rows
        let w_rem_row = &nn.l1_weights[w_remove][..256];
        let b_rem_row = &nn.l1_weights[b_remove][..256];
        let w_add_row = &nn.l1_weights[w_add][..256];
        let b_add_row = &nn.l1_weights[b_add][..256];

        // --- 4. High-Density Auto-Vectorized Parallel Loop Block ---
        if captured_piece != BoardPiece::NONE {
            let w_cap = get_feature_index(w_king_sq, captured_piece, captured_sq, false);
            let b_cap = get_feature_index(b_king_sq, captured_piece, captured_sq, true);
            
            let w_cap_row = &nn.l1_weights[w_cap][..256];
            let b_cap_row = &nn.l1_weights[b_cap][..256];

            // Clean, direct loops easily targeted by SIMD auto-vectorization.
            // i16 values are cast to i32 to maintain perfect accumulator precision.
            for i in 0..256 {
                self.white.vals[i] += (w_add_row[i] as i32) - (w_rem_row[i] as i32) - (w_cap_row[i] as i32);
                self.black.vals[i] += (b_add_row[i] as i32) - (b_rem_row[i] as i32) - (b_cap_row[i] as i32);
            }
        } else {
            for i in 0..256 {
                self.white.vals[i] += (w_add_row[i] as i32) - (w_rem_row[i] as i32);
                self.black.vals[i] += (b_add_row[i] as i32) - (b_rem_row[i] as i32);
            }
        }
    }

    #[inline(always)]
    pub fn unmake_move(
        &mut self, 
        nn: &'static NnueNetwork, 
        mv: UndoMove,
        mailbox: &[BoardPiece; 64], 
        w_king_sq: usize, 
        b_king_sq: usize
    ) {
        // Since bitboards haven't rolled back yet, the piece at the end square is the added piece
        let remove_piece: BoardPiece = mailbox[mv.end_sq];

        // --- 1. Identify Original Moving Piece (Reverse Promotion Logic) ---
        let add_piece = match mv.move_type {
            MoveFlag::PROMOTIONQUEEN | MoveFlag::PROMOTIONROOK | MoveFlag::PROMOTIONBISHOP | MoveFlag::PROMOTIONKNIGHT => {
                if remove_piece == BoardPiece::WQUEEN || remove_piece == BoardPiece::WROOK || remove_piece == BoardPiece::WBISHOP || remove_piece == BoardPiece::WKNIGHT {
                    BoardPiece::WPAWN
                } else {
                    BoardPiece::BPAWN
                }
            }
            _ => remove_piece,
        };

        // --- 2. Identify Captured Piece Coordinate (Handles En Passant) ---
        let captured_sq = if mv.move_type == MoveFlag::ENPASSANT {
            if add_piece == BoardPiece::WPAWN {
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
        let w_add_origin = get_feature_index(w_king_sq, add_piece, mv.start_sq, false);
        let b_add_origin = get_feature_index(b_king_sq, add_piece, mv.start_sq, true);

        let w_remove_dest = get_feature_index(w_king_sq, remove_piece, mv.end_sq, false);
        let b_remove_dest = get_feature_index(b_king_sq, remove_piece, mv.end_sq, true);

        let w_add_row = &nn.l1_weights[w_add_origin][..256];
        let b_add_row = &nn.l1_weights[b_add_origin][..256];
        let w_rem_row = &nn.l1_weights[w_remove_dest][..256];
        let b_rem_row = &nn.l1_weights[b_remove_dest][..256];

        // --- 4. High-Density Auto-Vectorized Parallel Loop Block ---
        let cap_piece = mv.captured_piece.unwrap_or(BoardPiece::NONE);
        
        if cap_piece != BoardPiece::NONE {
            let w_cap = get_feature_index(w_king_sq, cap_piece, captured_sq, false);
            let b_cap = get_feature_index(b_king_sq, cap_piece, captured_sq, true);

            let w_cap_row = &nn.l1_weights[w_cap][..256];
            let b_cap_row = &nn.l1_weights[b_cap][..256];

            // Flawless capture undo loop targeting compiler auto-vectorization registers
            for i in 0..256 {
                self.white.vals[i] += (w_add_row[i] as i32) + (w_cap_row[i] as i32) - (w_rem_row[i] as i32);
                self.black.vals[i] += (b_add_row[i] as i32) + (b_cap_row[i] as i32) - (b_rem_row[i] as i32);
            }
        } else {
            // Flat quiet move undo loop
            for i in 0..256 {
                self.white.vals[i] += (w_add_row[i] as i32) - (w_rem_row[i] as i32);
                self.black.vals[i] += (b_add_row[i] as i32) - (b_rem_row[i] as i32);
            }
        }
    }
}