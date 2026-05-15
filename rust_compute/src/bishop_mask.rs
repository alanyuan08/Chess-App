use std::sync::LazyLock;
use crate::move_command::*;

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

// Compute the number of significant bits to shift 
pub const BISHOP_SHIFT: [u64; 64] = {
    let mut shifts = [0u64; 64];
    let mut i = 0;
    while i < 64 {
        shifts[i] = mask(i as u8).count_ones() as u64;
        i += 1;
    }
    shifts
};

// Compute the offset for the calculation 
pub const BISHOP_OFFSETS: [u64; 64] = {
    let mut offset = [0u64; 64];

    let mut i = 0;
    let mut agg = 0;
    while i < 64 {
        offset[i] = agg;
        agg += 1 << BISHOP_SHIFT[i];
        i += 1;
    }

    offset
};

// Compute at Bishop Magic at Runtime.
pub static BISHOP_MAGIC: LazyLock<[u64; 64]> = LazyLock::new(|| {
    let mut magic_number = [0u64; 64];

    let mut i = 0;
    while i < 64 {
        magic_number[i] = compute_bishop_magic(i);
        i += 1;
    }

    magic_number
});

/// Computes the total size of the bishop attack table
const fn bishop_table_size() -> usize {
    let mut total = 0;
    let mut i = 0;
    while i < 64 {
        total += 1 << BISHOP_SHIFT[i];
        i += 1;
    }
    total
}

// Compute Bishop Attack after Bishop Magic is computed
pub const BISHOP_ATTACK_SIZE: usize = bishop_table_size();

pub static BISHOP_ATTACKS: LazyLock<Box<[u64; BISHOP_ATTACK_SIZE]>> = LazyLock::new(|| {
    let mut bishop_attack = Box::new([0u64; BISHOP_ATTACK_SIZE]);

    for i in 0..64 {
        let magic_number = BISHOP_MAGIC[i];
        let shift = BISHOP_SHIFT[i];

        let mask = BISHOP_MASKS[i];
        let offset = BISHOP_OFFSETS[i];

        let mut board = mask;
        loop {
            let index = (board.wrapping_mul(magic_number)) >> (64 - shift);
            let attack = compute_bishop_attacks(i, board);
            bishop_attack[(offset + index) as usize] = attack;

            if board == 0 { break; }
            board = board.wrapping_sub(1) & mask;
        }
    }

    bishop_attack
});

// Retrieve the Bishop Attack Paths for board / position
pub fn bishop_attack_paths(sq: usize, board: u64) -> u64{
    let magic_number = BISHOP_MAGIC[sq];
    let shift = BISHOP_SHIFT[sq];

    let mask = BISHOP_MASKS[sq];
    let offset = BISHOP_OFFSETS[sq];

    let index = ((board & mask).wrapping_mul(magic_number)) >> (64 - shift);
    BISHOP_ATTACKS[(offset + index) as usize]
}

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
            mask |= 1u64 << (cur_r * 8 + cur_f);
            cur_r += dr;
            cur_f += df;
        }
        d += 1;
    }

    mask
}

// Compute Bishop Attack Targets - Including First Obstacle
pub const fn compute_bishop_attacks(sq: usize, occupancy: u64) -> u64 {
    let mut attacks = 0u64;    
    let r = (sq / 8) as i8;
    let f = (sq % 8) as i8;

    let directions: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

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

// Compute Bishop Magic -> RunTime
pub fn compute_bishop_magic(sq: usize) -> u64 {
    let shift = BISHOP_SHIFT[sq];
    let mask = BISHOP_MASKS[sq];
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
            let index = (board.wrapping_mul(magic_candidate)) >> (64 - shift);
            let attack = compute_bishop_attacks(sq, board);

            if results[index as usize] == 0 {
                results[index as usize] = attack;
            } else if results[index as usize] != attack {
                found = false;
                break;
            }

            if board == 0 { break; } 
            board = board.wrapping_sub(1) & mask;
        }

        if found {
            return magic_candidate;
        }
    }
}

pub fn bishop_moves(mut bishop_bitboard: u64, occupancy: u64, 
    opponent_pieces: u64, moves: &mut Vec<Move>)  {

    while bishop_bitboard != 0 {
        let bishop = bishop_bitboard.trailing_zeros() as usize;

        let bishop_attack_paths = bishop_attack_paths(bishop, occupancy);

        let mut bishop_moves = bishop_attack_paths & !occupancy;
        let mut bishop_captures = bishop_attack_paths & opponent_pieces;

        while bishop_moves != 0 {
            let target = bishop_moves.trailing_zeros() as usize;
            moves.push(Move { startSq: bishop, endSq: target, moveType: MoveFlag::MOVE });
            bishop_moves &= bishop_moves - 1;
        }

        while bishop_captures != 0 {
            let target = bishop_captures.trailing_zeros() as usize;
            moves.push(Move { startSq: bishop, endSq: target, moveType: MoveFlag::CAPTURE });
            bishop_captures &= bishop_captures - 1;
        }

        bishop_bitboard &= bishop_bitboard - 1;
    }
}