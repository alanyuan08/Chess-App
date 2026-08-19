// Retain White / Black Accumulator values across Positions
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct Accumulator {
    pub vals: [i16; 256],
}

#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct BoardAccumulators {
    pub white: Accumulator,
    pub black: Accumulator,
}

impl Default for BoardAccumulators {
    fn default() -> Self {
        Self {
            white: Accumulator { vals: [0i16; 256] },
            black: Accumulator { vals: [0i16; 256] },
        }
    }
}
