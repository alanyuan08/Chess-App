pub const MAX_MOVES_ALLOCATION: usize = 256;

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

#[derive(Debug, Clone)]
pub struct AccumulatorHistory {
    pub moves: Box<[BoardAccumulators; MAX_MOVES_ALLOCATION]>,
}

impl Default for AccumulatorHistory {
    fn default() -> Self {
        Self { 
            moves: Box::new([BoardAccumulators::default(); MAX_MOVES_ALLOCATION]) 
        }
    }
}