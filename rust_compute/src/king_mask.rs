// Compute King Attack on Compile
pub const KING_ATTACKS: [u64; 64] = {
    let mut king_attack = [0u64; 64];

    let mut i = 0;
    while i < 64 {
        let r = (i / 8) as i8;
        let f = (i % 8) as i8;

        // The 8 directions for the knight
        let directions: [(i8, i8); 8] = 
        [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)];

        let mut d = 0;
        while d < 8 {
            let (dr, df) = directions[d];
            let (cur_r, cur_f) = (r + dr, f + df);
            
            if cur_r >= 0 && cur_r <= 7 && cur_f >= 0 && cur_f <= 7 {
                king_attack[i] |= 1 << (cur_r * 8 + cur_f);
            }
            d += 1;
        }
        i += 1;
    }

    king_attack
};
