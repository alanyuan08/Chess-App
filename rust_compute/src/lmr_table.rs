use std::sync::LazyLock;

// Compute the Late Move Reduction Table
pub static LMR_TABLE: LazyLock<[[i32; 64]; 64]> = LazyLock::new(|| {
    let mut table = [[0; 64]; 64];
    
    for (depth, <item>) in table.iter_mut().enumerate() {
        for (moves, <item>) in table.iter_mut().enumerate() {
            if depth < 3 || moves < 3 {
                table[depth][moves] = 0;
                continue;
            }
            
            // Adding 1.0 ensures we never take the log of 0 or 1, smoothing out the curve
            let r = 0.5 + ((depth as f64).ln() * (moves as f64).ln() / 2.25);
            
            table[depth][moves] = r.floor() as i32;
        }
    }
    
    table
});