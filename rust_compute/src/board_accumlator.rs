use crate::nnue_network::*;
use crate::move_command::*;

// Retain White / Black Accumulator values across Positions
#[derive(Clone, Copy)]
pub struct Accumulator {
    pub vals: [i16; 256],
}

pub struct BoardAccumulators {
    pub white: Accumulator,
    pub black: Accumulator,
}

impl BoardAccumulators {
    /// Re-reads the entire board layout from scratch to perform a full baseline refresh
    pub fn refresh_from_scratch(&mut self, nn: &NnueNetwork, 
        pieces: &[BoardPiece; 64], w_king_sq: usize, b_king_sq: usize) {
        // Reset both accumulators to the initial layer 1 baseline biases
        self.white.vals.copy_from_slice(&nn.l1_biases);
        self.black.vals.copy_from_slice(&nn.l1_biases);

        // Loop through every square on the board and add active pieces
        for sq in 0..64 {
            let piece = pieces[sq];
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
    pub fn make_move(&mut self, nn: &NnueNetwork, 
        mv: ForwardMove, w_king_sq: usize, b_king_sq: usize) {
        // If either king moves, trigger a clean refresh to redefine perspective baselines
        if mv.piece == BoardPiece::WKING || mv.piece == BoardPiece::BKING {
            // Assume you update the king squares in your board state *before* or *during* this branch
            // self.refresh_from_scratch(nn, current_pieces, new_w_king, new_b_king);
            return;
        }

        // --- 1. Remove the moving piece from its origin square ---
        let w_remove = get_feature_index(w_king_sq, mv.piece, mv.from, false);
        let b_remove = get_feature_index(b_king_sq, mv.piece, mv.from, true);

        // --- 2. Remove any captured enemy piece from the destination square ---
        let mut w_cap_idx = None;
        let mut b_cap_idx = None;
        if let Some(cap) = mv.captured_piece {
            w_cap_idx = Some(get_feature_index(w_king_sq, cap, mv.to, false));
            b_cap_idx = Some(get_feature_index(b_king_sq, cap, mv.to, true));
        }

        // --- 3. Add the moving piece to its destination square ---
        let w_add = get_feature_index(w_king_sq, mv.piece, mv.to, false);
        let b_add = get_feature_index(b_king_sq, mv.piece, mv.to, true);

        // --- 4. Parallel Hardware Update Block ---
        // Loops are structured cleanly for compiler unrolling and auto-vectorization
        for i in 0..256 {
            // White Update Pipeline
            self.white.vals[i] -= nn.l1_weights[w_remove][i];
            if let Some(w_cap) = w_cap_idx {
                self.white.vals[i] -= nn.l1_weights[w_cap][i];
            }
            self.white.vals[i] += nn.l1_weights[w_add][i];

            // Black Update Pipeline
            self.black.vals[i] -= nn.l1_weights[b_remove][i];
            if let Some(b_cap) = b_cap_idx {
                self.black.vals[i] -= nn.l1_weights[b_cap][i];
            }
            self.black.vals[i] += nn.l1_weights[b_add][i];
        }
    }

    /// Backtracks the network state perfectly when unmaking an evaluated move
    #[inline(always)]
    pub fn unmake_move(&mut self, nn: &NnueNetwork, mv: UndoMove, w_king_sq: usize, b_king_sq: usize) {
        if mv.piece == BoardPiece::WKING || mv.piece == BoardPiece::BKING {
            // If the king moved, roll back using a clean board state reload
            return;
        }

        // To undo, reverse every sign from the make_move function:
        // Add back old origin, add back the captured piece, remove the destination item
        let w_add_origin = get_feature_index(w_king_sq, mv.piece, mv.from, false);
        let b_add_origin = get_feature_index(b_king_sq, mv.piece, mv.from, true);

        let mut w_cap_idx = None;
        let mut b_cap_idx = None;
        if let Some(cap) = mv.captured_piece {
            w_cap_idx = Some(get_feature_index(w_king_sq, cap, mv.to, false));
            b_cap_idx = Some(get_feature_index(b_king_sq, cap, mv.to, true));
        }

        let w_remove_dest = get_feature_index(w_king_sq, mv.piece, mv.to, false);
        let b_remove_dest = get_feature_index(b_king_sq, mv.piece, mv.to, true);

        for i in 0..256 {
            // Rollback White
            self.white.vals[i] += nn.l1_weights[w_add_origin][i];
            if let Some(w_cap) = w_cap_idx {
                self.white.vals[i] += nn.l1_weights[w_cap][i];
            }
            self.white.vals[i] -= nn.l1_weights[w_remove_dest][i];

            // Rollback Black
            self.black.vals[i] += nn.l1_weights[b_add_origin][i];
            if let Some(b_cap) = b_cap_idx {
                self.black.vals[i] += nn.l1_weights[b_cap][i];
            }
            self.black.vals[i] -= nn.l1_weights[b_remove_dest][i];
        }
    }
}