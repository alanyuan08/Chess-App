use std::sync::OnceLock;

pub fn mask(sq: u8) -> u64 {
    let mut mask: u64 = 0;
    let r = (sq / 8) as i8;
    let f = (sq % 8) as i8;

    // The 4 diagonal directions
    let directions: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

    for &(dr, df) in &directions {
        let (mut cur_r, mut cur_f) = (r + dr, f + df);
        
        // BOUNDARY CHANGE: Allow 0 and 7 to include the board edges.
        while cur_r >= 0 && cur_r <= 7 && cur_f >= 0 && cur_f <= 7 {
            mask |= 1 << (cur_r * 8 + cur_f);
            cur_r += dr;
            cur_f += df;
        }
    }
    mask
}

pub static BISHOP_MASKS: OnceLock<[u64; 64]> = OnceLock::new();

pub fn precompute_all() -> [u64; 64] {
    let mut masks = [0u64; 64];
    for i in 0..64 {
        masks[i] = mask(i as u8);
    }
    masks
}