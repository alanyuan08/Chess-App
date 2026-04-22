pub const BISHOP_MASKS: [u64; 64] = {
    let mut masks = [0u64; 64];
    let mut i = 0;
    while i < 64 {
        masks[i] = mask(i as u8);
        i += 1;
    }
    masks
};

pub const fn mask(sq: u8) -> u64 {
    let mut mask: u64 = 0;
    let r = (sq / 8) as i8;
    let f = (sq % 8) as i8;

    // The 4 diagonal directions
    let directions: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

    let mut d = 0;
    while d < 4 {
        let (dr, df) = directions[d];
        let (mut cur_r, mut cur_f) = (r + dr, f + df);
        
        // Typical bishop mask excludes the very edge of the board 
        // to minimize magic bitboard table sizes. 
        // If you need the full ray, keep these as >= 0 and <= 7.
        while cur_r > 0 && cur_r < 7 && cur_f > 0 && cur_f < 7 {
            mask |= 1 << (cur_r * 8 + cur_f);
            cur_r += dr;
            cur_f += df;
        }
        d += 1;
    }

    mask
}
