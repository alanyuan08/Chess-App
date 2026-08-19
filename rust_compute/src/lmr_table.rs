use std::sync::LazyLock;

// Compute the Late Move Reduction Table
pub static LMR_TABLE: LazyLock<[[i32; 64]; 64]> = LazyLock::new(|| {
    let mut table = [[0; 64]; 64];
    
    for (depth, row) in table.iter_mut().enumerate() {
        for (moves, cell) in row.iter_mut().enumerate() {
            // Keep your original check to skip low numbers
            if depth < 3 || moves < 3 {
                *cell = 0;
                continue;
            }
                         
            let r = 0.5 + ((depth as f64).ln() * (moves as f64).ln() / 2.25);
            *cell = r.floor() as i32;
        }
    }
         
    table
});