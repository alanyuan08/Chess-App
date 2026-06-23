use std::sync::LazyLock;

// Compute the Late Move Redunction Table
pub static LMR_TABLE: LazyLock<[[i32; 64]; 64]> = LazyLock::new(|| {
    let mut table = [[0; 64]; 64];
    for depth in 1..64 {
        for moves in 1..64 {
            // Classic logarithmic reduction scaling formula
            let reduction = 0.5 + (depth as f64).ln() * (moves as f64).ln() / 2.25;
            table[depth][moves] = reduction.floor() as i32;
        }
    }
    table
});