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
        shifts[i] = mask(i as u8).count_ones() as u64;
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

/// Computes the total size of the rook attack table
const fn rook_table_size() -> usize {
    let mut total = 0;
    let mut i = 0;
    while i < 64 {
        total += 1 << ROOK_SHIFT[i];
        i += 1;
    }
    total
}

// Compute Rook Attack after rook Magic is computed
pub const ROOK_ATTACK_SIZE: usize = rook_table_size();

pub static ROOK_ATTACKS: LazyLock<Box<[u64; ROOK_ATTACK_SIZE]>> = LazyLock::new(|| {
    let mut rook_attack = Box::new([0u64; ROOK_ATTACK_SIZE]);

    for i in 0..64 {
        let magic_number = ROOK_MAGIC[i];
        let shift = ROOK_SHIFT[i];

        let mask = ROOK_MASKS[i];
        let offset = ROOK_OFFSETS[i];

        let mut board = 0u64;
        loop {
            let index = (board.wrapping_mul(magic_number)) >> (64 - shift);
            let attack = compute_rook_attacks(i, board);
            rook_attack[(offset + index) as usize] = attack;

            board = (board.wrapping_sub(mask)) & mask;
            if board == 0 { break; }
        }
    }

    rook_attack
});

// Retrieve the Rook Attack Paths for board / position
pub fn rook_attack_paths(sq: usize, board: u64) -> u64{
    let magic_number = ROOK_MAGIC[sq];
    let shift = ROOK_SHIFT[sq];

    let mask = ROOK_MASKS[sq];
    let offset = ROOK_OFFSETS[sq];

    let index = ((board & mask).wrapping_mul(magic_number)) >> (64 - shift);
    ROOK_ATTACKS[(offset + index) as usize]
}

pub const fn mask(sq: u8) -> u64 {
    let mut mask: u64 = 0;
    let r = (sq / 8) as i8;
    let f = (sq % 8) as i8;

    // Horizontal Searches 
    let horizontal_directions: [i8; 2] = [1, -1];
    let mut d = 0;
    while d < 2 {
        let dr = horizontal_directions[d];
        let mut cur_r = r + dr;
        while cur_r > 0 && cur_r < 7 {
            mask |= 1u64 << (cur_r * 8 + f);
            cur_r += dr;
        }
        d += 1;
    }

    // Vertical Searches 
    let vertical_directions: [i8; 2] = [1, -1];
    let mut d = 0;
    while d < 2 {
        let df = vertical_directions[d];
        let mut cur_f = f + df;
        while cur_f > 0 && cur_f < 7 {
            mask |= 1u64 << (r * 8 + cur_f);
            cur_f += df;
        }
        d += 1;
    }
    mask
}

// Compute Rook Attack Targets - Including First Obstacle
pub const fn compute_rook_attacks(sq: usize, occupancy: u64) -> u64 {
    let mut attacks = 0u64;    
    let r = (sq / 8) as i8;
    let f = (sq % 8) as i8;

    // Horizontal Searches
    let horizontal_attacks: [i8; 2] = [1, -1];
    let mut d = 0;
    while d < 2 {
        let dr = horizontal_attacks[d];
        let mut cur_r = r + dr;
        // Only skip the LAST square in each direction
        while cur_r >= 0 && cur_r <= 7 {
            let target_bit = 1u64 << (cur_r * 8 + f);

            attacks |= target_bit;
            if (occupancy & target_bit) != 0 {
                break;
            }
            cur_r += dr;
        }
        d += 1;
    }

    // Vertical Searches 
    let vertical_directions: [i8; 2] = [1, -1];
    let mut d = 0;
    while d < 2 {
        let df = vertical_directions[d];
        let mut cur_f = f + df;
        // Only skip the LAST square in each direction
         while cur_f >= 0 && cur_f <= 7 {
            let target_bit = 1u64 << (r * 8 + cur_f);

            attacks |= target_bit;
            if (occupancy & target_bit) != 0 {
                break;
            }
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
            let index = ((board & mask).wrapping_mul(magic_candidate)) >> (64 - shift);
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
