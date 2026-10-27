pub fn calculate_lmr_base(depth: usize, moves: usize) -> i8 {
    if depth < 3 || moves < 3 {
        return 0;
    }
    // Elite Tuning (1.42 Divisor): Aggressive narrowing optimized for highly stable NNUE horizons
    let r = (depth as f64).ln() * (moves as f64).ln() / 1.42;
    r as i8
}

pub static LMR_TABLE: std::sync::LazyLock<[[i8; 64]; 64]> = std::sync::LazyLock::new(|| {
    let mut table = [[0; 64]; 64];
    
    for (depth, row) in table.iter_mut().enumerate() {
        for (moves, cell) in row.iter_mut().enumerate() {
            *cell = calculate_lmr_base(depth, moves);
        }
    }
    
    table
});