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

// Condon-Thompson Bucket using 100% stable AtomicU64 primitives.
#[derive(Debug)]
#[repr(C)] // Guarantees fields remain in this exact order in memory
pub struct TT_Bucket {
    pub depth_preferred: AtomicU64, // Slot 1 (8 bytes)
    pub always_replace: AtomicU64,  // Slot 2 (8 bytes)
}

// Condon-Thompson transposition table using packed 128-bit buckets
pub struct TranspositionTable {
    buckets: Vec<TT_Bucket>,
    mask: usize,
}

impl TranspositionTable {
    /// Creates a flat table matching the nearest power-of-two megabytes
    pub fn new(mb: usize) -> Self {
        let size_bytes = mb * 1024 * 1024;
        let count = size_bytes / std::mem::size_of::<TT_Bucket>();
        
        // Round down to power of two for fast bitwise indexing
        let power_of_two_count = count.next_power_of_two() >> 1;
        let final_count = std::cmp::max(1, power_of_two_count);

        let buckets = (0..final_count)
            .map(|_| TT_Bucket {
                depth_preferred: AtomicU64::new(0),
                always_replace: AtomicU64::new(0),
            })
            .collect();
        Self {
            buckets,
            mask: final_count - 1,
        }
    }

    /// Packs raw components into a 64-bit word
    #[inline(always)]
    fn pack_entry(move_id: u16, score: i16, depth: i32, flag: HashFlag, key: u64, ply: i32) -> u64 {
        
        let mut store_score = score;
        if store_score > (MATE_VALUE - MAX_DEPTH) as i16 { 
            store_score += ply as i16; 
        } else if store_score < (-MATE_VALUE + MAX_DEPTH) as i16 { 
            store_score -= ply as i16; 
        }
        
        let tag_22 = (key >> 42) & 0x3F_FFFF; 
        let mut packed = 0u64;
        packed |= (move_id as u64) & 0xFFFF;
        packed |= (store_score as u64) << 16;
        packed |= ((depth as u8) as u64) << 32;
        packed |= tag_22 << 40;
        packed |= (flag as u8 as u64) << 62;
        packed
    }

    /// Unpacks a 64-bit word into an operational TTEntry if the tag matches
    #[inline(always)]
    fn unpack_entry(packed: u64, key: u64, ply: i32) -> Option<TTEntry> {
        if packed == 0 {
            return None;
        }

        let stored_tag = ((packed >> 40) & 0x3F_FFFF) as u32;
        let current_tag = ((key >> 42) & 0x3F_FFFF) as u32; 

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
    pub fn probe(&self, key: u64, ply: i32) -> Option<TTEntry> {
        let index = (key as usize) & self.mask;
        let bucket = &self.buckets[index];
        
        // Fetching depth_preferred pulls the entire TT_Bucket cache line into L1.
        let dp_packed = bucket.depth_preferred.load(Ordering::Relaxed);
        if let Some(entry) = Self::unpack_entry(dp_packed, key, ply) {
            return Some(entry);
        }

        // Fetching always_replace is an immediate L1 hit (0 penalty)
        let ar_packed = bucket.always_replace.load(Ordering::Relaxed);
        if let Some(entry) = Self::unpack_entry(ar_packed, key, ply) {
            return Some(entry);
        }
        
        None
    }

    #[inline(always)]
    pub fn store(&self, key: u64, score: i32, ply: i32, 
        forward_move: Option<ForwardMove>, depth: i32, flag: HashFlag) 
    {
        let index = (key as usize) & self.mask;
        let bucket = &self.buckets[index];

        let move_id = forward_move.map_or(0, |mv| mv.pack()); 
        let new_packed = Self::pack_entry(move_id, score as i16, depth, flag, key, ply);

        // --- SLOT 1: DEPTH PREFERRED ---
        let mut current_dp = bucket.depth_preferred.load(Ordering::Relaxed);
        loop {
            let existing_dp_depth = ((current_dp >> 32) & 0xFF) as i8 as i32;

            if depth >= existing_dp_depth {
                // If the new entry is deeper, overwrite depth_preferred.
                // Demote the displaced deep entry down into the always_replace slot.
                match bucket.depth_preferred.compare_exchange_weak(
                    current_dp,
                    new_packed,
                    Ordering::Release,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        if current_dp != 0 {
                            bucket.always_replace.store(current_dp, Ordering::Release);
                        }
                        return;
                    }
                    Err(actual) => current_dp = actual,
                }
            } else {
                // --- SLOT 2: FALLBACK TO ALWAYS REPLACE ---
                // If it's too shallow for the depth slot, directly write it here.
                bucket.always_replace.store(new_packed, Ordering::Release);
                return;
            }
        }
    }
}