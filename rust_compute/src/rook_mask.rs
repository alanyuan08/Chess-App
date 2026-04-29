use std::sync::LazyLock;

// Mask the Irrelevant Bits no in the Diagonal Path
pub const ROOK_MASKS: [u64; 64] = {
    let mut masks = [0u64; 64];
    let mut i = 0;
    while i < 64 {
        masks[i] = mask(i as u8);
        i += 1;
    }
    masks
};

// Compute the number of significant bits to shift 
pub const ROOK_SHIFT: [u64; 64] = {
    let mut shifts = [0u64; 64];
    let mut i = 0;
    while i < 64 {
        shifts[i] = compute_shift(i as u8);
        i += 1;
    }
    shifts
};

// Compute the offset for the calculation 
pub const ROOK_OFFSETS: [u64; 64] = {
    let mut offset = [0u64; 64];

    let mut i = 0;
    let mut agg = 0;
    while i < 64 {
        offset[i] = agg;
        agg += 1 << ROOK_SHIFT[i];
        i += 1;
    }

    offset
};

// Compute at Rook Magic at Runtime.
pub static ROOK_MAGIC: LazyLock<[u64; 64]> = LazyLock::new(|| {
    let mut magic_number = [0u64; 64];

    let mut i = 0;
    while i < 64 {
        magic_number[i] = compute_rook_magic(i);
        i += 1;
    }

    magic_number
});

// Compute Rook Attack after Rook Magic is computed

// Number of Rook Combinations is 5248
pub static ROOK_ATTACKS: LazyLock<[u64; 5248]> = LazyLock::new(|| {
    let mut bishop_attack = [0u64; 5248];

    for i in 0..64 {
        let magic_number = ROOK_MAGIC[i];
        let shift = ROOK_SHIFT[i];

        let mask = ROOK_MASKS[i];
        let offset = ROOK_OFFSETS[i];

        let mut board = mask;
        loop {
            let index = ((board & mask) * magic_number) >> (64 - shift);
            let attack = compute_rook_attacks(i, board);

            bishop_attack[(offset + index) as usize] = attack;

            board = (board - 1) & mask;
            if board == 0 { break; }
        }
    }

    bishop_attack
});

// Retrieve the Bishop Attack Paths for board / position
pub fn bishop_move_paths(sq: usize, board: u64) -> u64{
    let magic_number = BISHOP_MAGIC[sq];
    let shift = BISHOP_SHIFT[sq];

    let mask = BISHOP_MASKS[sq];
    let offset = BISHOP_OFFSETS[sq];

    let index = ((board & mask) * magic_number) >> (64 - shift);
    BISHOP_ATTACKS[(offset + index) as usize]
}

// Mask the Cross route
pub const fn mask(sq: u8) -> u64 {
    let mut mask: u64 = 0;
    let r = (sq / 8) as i8;
    let f = (sq % 8) as i8;

    // The cross directions
    let directions: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

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

    shift as u64
}

// Compute Rook Attack Targets - Including First Obstacle
pub const fn compute_rook_attacks(sq: usize, occupancy: u64) -> u64 {
    let mut attacks = 0u64;    
    let r = (sq / 8) as i8;
    let f = (sq % 8) as i8;

    let directions: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

    let mut d = 0;
    while d < 4 {
        let (dr, df) = directions[d];
        let (mut cur_r, mut cur_f) = (r + dr, f + df);

        while cur_r >= 0 && cur_r <= 7 && cur_f >= 0 && cur_f <= 7 {
            let target_bit = 1u64 << (cur_r * 8 + cur_f);

            attacks |= target_bit;
            if (occupancy & target_bit) != 0 {
                break;
            }
            
            cur_r += dr;
            cur_f += df;
        }
        d += 1;
    }
    attacks
}

// Compute Rook Magic -> RunTime
pub fn compute_rook_magic(sq: usize) -> u64 {
    let shift = ROOK_SHIFT[sq];
    let mask = ROOK_MASKS[sq];
    let size = 1 << shift;

    let mut seed: u64 = 0x12345678;

    loop {
        // Rust RNG is generated at compile time and not run run
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        
        let magic_candidate = seed & (seed.rotate_left(15)) & (seed.rotate_left(31));

        let mut results = vec![0u64; size];
        let mut found = true;

        // Ripply-Carry - Iterative until 0 
        // s = (s - 1) & m
        // Find Magic Number such that this is unique hash
        // (Board & Mask) * Magic Number >> (64 - n)
        let mut board = mask;
        loop {
            let index = ((board & mask) * magic_candidate) >> (64 - shift);
            let attack = compute_rook_attacks(sq, board);

            if results[index as usize] == 0 || results[index as usize] == attack {
                results[index as usize] = attack;
            } else {
                found = false;
                break;
            }
            board = (board - 1) & mask;
            if board == 0 { break; } 
        }

        if found {
            return magic_candidate;
        }
    }
}
