use crate::move_command::*;
use crate::chess_game::*;

use std::sync::atomic::{AtomicU64, Ordering};

// TT Entry Flag definitions
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum HashFlag {
    EXACT = 0,
    LOWERBOUND = 1,
    UPPERBOUND = 2,
}

// TTEntry Structure
#[derive(Clone, Copy, Debug)]
pub struct TTEntry {
    pub key: u64,
    pub move_id: u16,
    pub score: i16,
    pub depth: i8,
    pub flag: HashFlag,
}

// Simple Transposition Table mapping 1 hash key to 1 slot
pub struct TranspositionTable {
    entries: Vec<AtomicU64>,
    mask: usize,
}

impl TranspositionTable {
    /// Creates a flat table matching the nearest power-of-two megabytes
    pub fn new(mb: usize) -> Self {
        let size_bytes = mb * 1024 * 1024;
        let count = size_bytes / std::mem::size_of::<AtomicU64>();
        
        // Round down to power of two for fast bitwise indexing
        let power_of_two_count = count.next_power_of_two() >> 1;
        let final_count = std::cmp::max(1, power_of_two_count);

        let entries = (0..final_count).map(|_| AtomicU64::new(0)).collect();
        Self {
            entries,
            mask: final_count - 1,
        }
    }

    #[inline(always)]
    pub fn probe(&self, key: u64, ply: i32) -> Option<TTEntry> {
        let index = (key as usize) & self.mask;
        
        // Single CPU cycle fetch (1 instruction if cached)
        let packed = self.entries[index].load(Ordering::Relaxed);
        if packed == 0 {
            return None;
        }

        // Extract 22-bit tag precisely (Bits 40..61)
        let stored_tag = ((packed >> 40) & 0x3F_FFFF) as u32;
        let current_tag = ((key >> 42) & 0x3F_FFFF) as u32; 

        // If the partial keys match, we accept the entry
        if stored_tag == current_tag {
            let move_id = packed as u16;
            let score = (packed >> 16) as i16;
            let depth = ((packed >> 32) & 0xFF) as i8;
            
            let flag_val = ((packed >> 62) & 0b11) as u8;
            let flag = match flag_val {
                0 => HashFlag::EXACT,
                1 => HashFlag::LOWERBOUND,
                _ => HashFlag::UPPERBOUND,
            };

            let mut entry = TTEntry {
                key, 
                move_id,
                score,
                depth,
                flag,
            };

            if entry.score > (MATE_VALUE - MAX_DEPTH) as i16 { 
                entry.score -= ply as i16; 
            } else if entry.score < (-MATE_VALUE + MAX_DEPTH) as i16 { 
                entry.score += ply as i16; 
            }
            return Some(entry);
        }
        
        None
    }

    #[inline(always)]
    pub fn store(&self, key: u64, score: i32, ply: i32,
        forward_move: Option<ForwardMove>, depth: i32, flag: HashFlag) 
    {
        let index = (key as usize) & self.mask;
        let move_id = forward_move.map_or(0, |mv| mv.pack()); 

        let mut store_score = score as i16;
        if store_score > (MATE_VALUE - MAX_DEPTH) as i16 { 
            store_score += ply as i16; 
        } else if store_score < (-MATE_VALUE + MAX_DEPTH) as i16 { 
            store_score -= ply as i16; 
        }

        // Extract exact 22 bits from the key for collision checks
        let tag_22 = (key >> 42) & 0x3F_FFFF; 

        // Pre-pack our incoming payload data
        let mut final_packed = 0u64;
        final_packed |= (move_id as u64) & 0xFFFF;             // Enforce 16-bit bounds
        final_packed |= ((store_score as u16) as u64) << 16;
        final_packed |= ((depth as u8) as u64) << 32;
        final_packed |= tag_22 << 40;
        final_packed |= (flag as u8 as u64) << 62;

        // --- LOCK-FREE COMPARE-AND-EXCHANGE LOOP ---
        // Load the initial state of the slot
        let mut current_packed = self.entries[index].load(Ordering::Relaxed);

        loop {
            let existing_depth = ((current_packed >> 32) & 0xFF) as i8 as i32;

            if depth < existing_depth {
                return;
            }

            match self.entries[index].compare_exchange_weak(
                current_packed,
                final_packed,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => {
                    current_packed = actual;
                }
            }
        }
    }
}