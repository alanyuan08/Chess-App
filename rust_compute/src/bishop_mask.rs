// Mask the Irrelevant Bits no in the Diagonal Path
pub const BISHOP_MASKS: [u64; 64] = {
    let mut masks = [0u64; 64];
    let mut i = 0;
    while i < 64 {
        masks[i] = mask(i as u8);
        i += 1;
    }
    masks
};

// (Board & Mask) * Magic Number >> (64 - n)

// Compute the number of significant bits to shift 
pub const BISHOP_SHIFT: [u64; 64] = {
    let mut shifts = [0u64; 64];
    let mut i = 0;
    while i < 64 {
        shifts[i] = compute_shift(i as u8);
        i += 1;
    }
    shifts
};

pub const BISHOP_OFFSETS: [u64; 64] = {
    let mut magic_number = [0u64; 64];

    magic_number
};

// Compute 
pub const BISHOP_MAGIC: [u64; 64] = {
    let mut magic_number = [0u64; 64];

    magic_number
};

pub const BISHOP_ATTACKS: [u64; 64] = {
    let mut magic_number = [0u64; 64];

    magic_number
};

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

// Compute shift 
pub const fn compute_shift(sq: u8) -> u64 {
    let r = (sq / 8) as i8;
    let f = (sq % 8) as i8;

    let col_left = if f > 1 { f - 1 } else { 0 };
    let col_right = if f < 6 { 6 - f } else { 0 };
    let row_bot = if r > 1 { r - 1 } else { 0 };
    let row_top = if r < 6 { 6 - r } else { 0 };

    let mut shift = 0;
    shift += if col_left < row_bot { col_left } else { row_bot };
    shift += if col_right < row_bot { col_right } else { row_bot };
    shift += if col_left < row_top { col_left } else { row_top };
    shift += if col_right < row_top { col_right } else { row_top };

    1 << shift
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
