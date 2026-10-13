// Retain White / Black Accumulator values across Positions
#[cfg_attr(target_arch = "aarch64", repr(C, align(128)))]
#[cfg_attr(not(target_arch = "aarch64"), repr(C, align(64)))]
#[derive(Debug, Clone, Copy)]
pub struct BoardAccumulators {
    /// 256 * 2 bytes = 512 bytes (Exactly 4 cache lines)
    pub white: [i16; 256],
    /// 256 * 2 bytes = 512 bytes (Exactly 4 cache lines)
    pub black: [i16; 256],
}

impl Default for BoardAccumulators {
    #[inline(always)]
    fn default() -> Self {
        Self {
            white: [0i16; 256],
            black: [0i16; 256],
        }
    }
}

impl BoardAccumulators {    
    /// Explicitly zeroes out all internal scratchpad arrays in-place.
    /// This prevents historical calculations from bleeding into a new evaluation.
    #[inline(always)]
    pub fn clear(&mut self) {
        self.white.fill(0);
        self.black.fill(0);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AccumulatorHistory {
    pub moves: [BoardAccumulators; MAX_GAME_MOVES],
}

impl Default for AccumulatorHistory {
    fn default() -> Self {
        Self {
            moves: [BoardAccumulators::default(); MAX_GAME_MOVES],
        }
    }
}