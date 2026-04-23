pub const BISHOP_MASKS: [u64; 64] = {
    let mut masks = [0u64; 64];
    let mut i = 0;
    while i < 64 {
        masks[i] = mask(i as u8);
        i += 1;
    }
    masks
};

// Retrieval 
// 1 - Apply Board + Mask only diagonal values
// 2 - ((occupancy & mask) * magic_number) >> (64 - shift)

// Compile - Build
// 1 - Generate every possiblity -> Store inside a set
// 2 - Test magic number for all 2096 possbilities to see if there are collisions
// 3 - Build the Hash Table

// Mask the diagonal route
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
        
        // Bitboard Excludes the Edges of the Board
        while cur_r > 0 && cur_r < 7 && cur_f > 0 && cur_f < 7 {
            mask |= 1 << (cur_r * 8 + cur_f);
            cur_r += dr;
            cur_f += df;
        }
        d += 1;
    }

    mask
}

// Mask the capture targets
pub const fn compute_path_mask(sq: u8, input_board: u64) -> u64 {
    let mut mask: u64 = 0;
    let r = (sq / 8) as i8;
    let f = (sq % 8) as i8;

    // The 4 diagonal directions
    let directions: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

    let mut d = 0;
    while d < 4 {
        let (dr, df) = directions[d];
        let (mut cur_r, mut cur_f) = (r + dr, f + df);
        
        // Bitboard Excludes the Edges of the Board
        while cur_r > 0 && cur_r < 7 && cur_f > 0 && cur_f < 7 {

            if (input_board >> (cur_r * 8 + cur_f)) & 1 == 0 {
                mask |= 1 << (cur_r * 8 + cur_f);
                cur_r += dr;
                cur_f += df;
            }
            break;
        }
        d += 1;
    }

    mask
}
