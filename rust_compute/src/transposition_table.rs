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
    EMPTY = 3,
}

// TTEntry Structure
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))] 
pub struct TTEntry {
    pub key: u64,
    pub move_id: u16,
    pub score: i16,
    pub depth: i8,
    pub flag: HashFlag,
}

// Condon-Thompson Bucket using 100% stable AtomicU64 primitives.
// current implement
pub struct TtBucket {
    pub depth_preferred: AtomicU64, // Slot 1 (8 bytes)
    pub always_replace: AtomicU64,  // Slot 2 (8 bytes)
}

// Condon-Thompson transposition table using packed 128-bit buckets
// The current implementation uses a 32 MB size cache to retain the memory within
// Apple M4 L3 Cache - This is implementation is currently unneccessary. 
pub struct TranspositionTable {
    buckets: Vec<TtBucket>,
    mask: usize,
    age: u8
}

impl TranspositionTable {
    /// Creates a flat table matching the nearest power-of-two megabytes
    pub fn new(mb: usize) -> Self {
        let size_bytes = mb * 1024 * 1024;
        let count = size_bytes / std::mem::size_of::<TtBucket>();
        
        // Round down to power of two for fast bitwise indexing
        let power_of_two_count = count.next_power_of_two() >> 1;
        let final_count = std::cmp::max(1, power_of_two_count);

        let buckets = (0..final_count)
            .map(|_| TtBucket {
                depth_preferred: AtomicU64::new(0),
                always_replace: AtomicU64::new(0),
            })
            .collect();
        Self {
            buckets,
            mask: final_count - 1,
            age: 0,
        }
    }

    pub fn increment_age(&mut self) {
        self.age = self.age.wrapping_add(1);
    }

    /// Packs raw components into a 64-bit word
    #[inline(always)]
    fn pack_entry(&self, move_id: u16, score: i16, depth: i32, 
        flag: HashFlag, key: u64) -> u64 {
        
        let tag_22 = (key >> 42) & 0x3F_FFFF; 
        let mut packed = 0u64;

        packed |= (move_id as u64) & 0xFFFF;
        packed |= ((score as u16) as u64) << 16;
        packed |= ((depth as u8) as u64 & 0xFF) << 32;
        packed |= (tag_22 & 0x3F_FFFF) << 40;
        packed |= (flag as u8 as u64) << 62;
        packed
    }

    /// Unpacks a 64-bit word into an operational TTEntry if the tag matches
    #[inline(always)]
    fn unpack_entry(&self, packed: u64, key: u64, ply: i32) -> TTEntry {
        let empty_entry = TTEntry {
            key: 0,
            move_id: 0,
            score: 0,
            depth: 0,
            flag: HashFlag::EMPTY,
        };

        if packed == 0 {
            return empty_entry;
        }

        let stored_tag = ((packed >> 40) & 0x3F_FFFF) as u32;
        let current_tag = ((key >> 42) & 0x3F_FFFF) as u32; 

        if stored_tag != current_tag {
            return empty_entry;
        }
        
        let move_id = packed as u16;
        let score = ((packed >> 16) & 0xFFFF) as i16;
        let depth = ((packed >> 32) & 0xFF) as u8 as i8;
        let flag_val = ((packed >> 62) & 0b11) as u8;

        let flag = match flag_val {
            0 => HashFlag::EXACT,
            1 => HashFlag::LOWERBOUND,
            2 => HashFlag::UPPERBOUND,
            _ => HashFlag::EMPTY,
        };

        let mut entry = TTEntry {
            key, 
            move_id,
            score,
            depth,
            flag,
        };

        if entry.score > MATE_THRESHOLD as i16 { 
            entry.score -= ply as i16; 
        } else if entry.score < -MATE_THRESHOLD as i16 { 
            entry.score += ply as i16; 
        }
        entry
    }

    #[inline(always)]
    pub fn probe(&self, key: u64, ply: i32) -> TTEntry {
        let index = (key as usize) & self.mask;
        let bucket = &self.buckets[index];
        
        // Fetching depth_preferred pulls the entire TT_Bucket cache line into L1.
        let dp_packed = bucket.depth_preferred.load(Ordering::Relaxed);
        let entry = self.unpack_entry(dp_packed, key, ply);
        if entry.flag != HashFlag::EMPTY {
            return entry;
        }

        // Fetching always_replace is an immediate L1 hit (0 penalty)
        let ar_packed = bucket.always_replace.load(Ordering::Relaxed);
        let entry = self.unpack_entry(ar_packed, key, ply);
        if entry.flag != HashFlag::EMPTY {
            return entry;
        }
        
        TTEntry {
            key: 0,
            move_id: 0,
            score: 0,
            depth: 0,
            flag: HashFlag::EMPTY,
        }
    }

    #[inline(always)]
    pub fn store(&self, key: u64, score: i32, ply: i32, 
        forward_move: ForwardMove, depth: i32, flag: HashFlag) 
    {
        let index = (key as usize) & self.mask;
        let bucket = &self.buckets[index];

        let move_id = if forward_move.move_type == MoveFlag::NULL {
            0
        } else {
            forward_move.pack()
        };

        let mut storage_score = score;
        if storage_score >= MATE_THRESHOLD {
            storage_score += ply;
        } else if storage_score <= -MATE_THRESHOLD {
            storage_score -= ply;
        }

        let packed_new = self.pack_entry(move_id, storage_score as i16, depth, flag, key);

        let packed_depth_slot = bucket.depth_preferred.load(Ordering::Relaxed);
        let current_depth_entry = self.unpack_entry(packed_depth_slot, key, ply);

        // Replacement Strategy Logic:
        // Overwrite depth preferred if the slot is empty, if the old entry belongs to an 
        // older engine search iteration, or if the new search depth is deeper.
        if current_depth_entry.flag == HashFlag::EMPTY || (depth as i8) >= current_depth_entry.depth {
            bucket.depth_preferred.store(packed_new, Ordering::Relaxed);
        } else {
            // If the depth slot is too high quality to overwrite, put this shallower entry
            // (like a quiescence search result) into the secondary replacement tier.
            bucket.always_replace.store(packed_new, Ordering::Relaxed);
        }
    }
}