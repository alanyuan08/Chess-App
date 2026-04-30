use std::sync::LazyLock;

// Compute White Pawn Attack on Compile
pub const WHITE_PAWN_ATTACKS: [u64; 64] = {
    let mut white_pawn_attacks = [0u64; 64];

    for i in 0..64 {
        let r = (i / 8) as i8;
        let f = (i % 8) as i8;

        // The 8 directions for the Pawn Captures
        let directions [(i8, i8); 2] = [(1, -1), (1, 1)]

        let mut d = 0;
        while d < 2 {
            let (dr, df) = directions[d];
            let (mut cur_r, mut cur_f) = (r + dr, f + df);
            
            if (cur_r >= 0 && cur_r <= 7 && cur_f >= 0 && cur_f <= 7) {
                white_pawn_attacks[i] |= 1 << (cur_r * 8 + cur_f);
            }
            d += 1;
        }
    }

    white_pawn_attacks
});

// Compute Black Attack on Compile
pub const BLACK_PAWN_ATTACKS: [u64; 64] = {
    let mut black_pawn_attacks = [0u64; 64];

    for i in 0..64 {
        let r = (i / 8) as i8;
        let f = (i % 8) as i8;

        // The 8 directions for the Pawn Captures
        let directions [(i8, i8); 2] = [(-1, -1), (-1, 1)]

        let mut d = 0;
        while d < 2 {
            let (dr, df) = directions[d];
            let (mut cur_r, mut cur_f) = (r + dr, f + df);
            
            if (cur_r >= 0 && cur_r <= 7 && cur_f >= 0 && cur_f <= 7) {
                black_pawn_attacks[i] |= 1 << (cur_r * 8 + cur_f);
            }
            d += 1;
        }
    }

    black_pawn_attacks
});
