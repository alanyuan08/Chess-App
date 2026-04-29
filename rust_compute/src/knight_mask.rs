use std::sync::LazyLock;

// Compute Knight Attack on Compile
pub const KNIGHT_ATTACKS: [u64; 64] = {
    let mut knight_attack = [0u64; 64];

    for i in 0..64 {
        let r = (i / 8) as i8;
        let f = (i % 8) as i8;

        // The 8 directions for the knight
        let directions [(i8, i8); 8] = 
            [(-2, -1), (-1, -2), (1, 2), (2, 1), (1, -2), (2, -1), (-1, 2), (-2, 1)]

        let mut d = 0;
        while d < 8 {
            let (dr, df) = directions[d];
            let (mut cur_r, mut cur_f) = (r + dr, f + df);
            
            if (cur_r >= 0 && cur_r <= 7 && cur_f >= 0 && cur_f <= 7) {
                knight_attack[i] |= 1 << (cur_r * 8 + cur_f);
            }
            d += 1;
        }
    }

    knight_attack
});
