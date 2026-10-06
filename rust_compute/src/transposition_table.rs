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
    pub key: u64,      // 8 bytes
    pub move_id: u16,  // 2 bytes
    pub score: i16,    // 2 bytes
    pub depth: i8,     // 1 byte
    pub flag: HashFlag,// 1 byte (u8)
    pub age: u8,       // 1 byte -> (Uses 1 byte of the old padding)
}

// Condon-Thompson Bucket using 100% stable AtomicU64 primitives.
// current implement
#[repr(align(128))]
pub struct TtBucket {
    // Slots 0 & 4: Depth Preferred
    // Slots 5 through 7: Always Replace
    pub slots: [TtSlot; 8],

}

// A stable 16-byte slot using two AtomicU64 primitives
pub struct TtSlot {
    pub key: AtomicU64,  // 8 bytes
    pub data: AtomicU64, // 8 bytes (Packed fields)
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
        // 1 MB = 1,048,576 bytes (1024 * 1024)
        let size_bytes = mb * 1024 * 1024;
        let count = size_bytes / std::mem::size_of::<TtBucket>();
        
        // Round down to the nearest power of two to guarantee the table fits safely 
        // within the user's requested megabyte limit while preserving bitwise masking.
        let mut final_count = count.next_power_of_two();
        if final_count > count && final_count > 1 {
            final_count >>= 1;
        }
        final_count = std::cmp::max(1, final_count);

        // Pre-allocate the vector with the exact capacity needed to avoid resizing overhead
        let mut buckets = Vec::with_capacity(final_count);
        
        for _ in 0..final_count {
            // Initialize each of the 8 slots in the bucket array with empty values (0)
            // By initializing the flag value to 0, it maps to HashFlag::EXACT, but since 
            // the data is 0, the depth is 0, and the key is 0, it effectively reads as empty.
            // However, to ensure absolute clarity, 0 indicates an unwritten slot.
            buckets.push(TtBucket {
                slots: [
                    TtSlot { key: AtomicU64::new(0), data: AtomicU64::new(0) },
                    TtSlot { key: AtomicU64::new(0), data: AtomicU64::new(0) },
                    TtSlot { key: AtomicU64::new(0), data: AtomicU64::new(0) },
                    TtSlot { key: AtomicU64::new(0), data: AtomicU64::new(0) },
                    TtSlot { key: AtomicU64::new(0), data: AtomicU64::new(0) },
                    TtSlot { key: AtomicU64::new(0), data: AtomicU64::new(0) },
                    TtSlot { key: AtomicU64::new(0), data: AtomicU64::new(0) },
                    TtSlot { key: AtomicU64::new(0), data: AtomicU64::new(0) },
                ],
            });
        }

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
    fn pack_entry(&self, move_id: u16, score: i16, depth: i32, flag: HashFlag, age: u8) -> u64 {
        let mut packed = 0u64;

        // Bitwise Layout: 
        // [16 bits Move] [16 bits Score] [8 bits Depth] [8 bits Flag] [8 bits Age] [12 bits Unused Padding]
        packed |= (move_id as u64) & 0xFFFF;
        packed |= ((score as u16) as u64) << 16;
        packed |= ((depth as u8) as u64 & 0xFF) << 32;
        packed |= (flag as u8 as u64 & 0xFF) << 40;
        packed |= (age as u64 & 0xFF) << 48; // Age utilizes the first 8 bits of padding
        
        packed
    }

    /// Unpacks a 64-bit word into an operational TTEntry if the tag matches
    #[inline(always)]
    fn unpack_entry(&self, packed: u64, key: u64, ply: i32) -> TTEntry {
        if packed == 0 {
            return TTEntry {
                key: 0,
                move_id: 0,
                score: 0,
                depth: 0,
                flag: HashFlag::EMPTY,
                age: 0, // Added default age value
            };
        }

        let move_id = packed as u16;
        let score = ((packed >> 16) & 0xFFFF) as i16;
        let depth = ((packed >> 32) & 0xFF) as u8 as i8;
        let flag_val = ((packed >> 40) & 0xFF) as u8;
        let age = ((packed >> 48) & 0xFF) as u8; // Extracted age from the 48th bit shift

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
            age, // Assigned extracted age
        };

        // --- Retained Mate Score Scaling Logic ---
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
        
        // Scan all 8 slots inside the 128-byte bucket cache line.
        // Fetching the key of the first slot pulls the entire line into L1 cache,
        // making subsequent iterations lightning-fast L1 register hits.
        for i in 0..8 {
            let slot = &bucket.slots[i];
            let stored_key = slot.key.load(Ordering::Relaxed);
            
            // Check for an exact, un-truncated 64-bit Zobrist key match
            if stored_key == key {
                let packed_data = slot.data.load(Ordering::Relaxed);
                
                // Directly check the flag bits from the raw data to quickly avoid empty slots 
                // without immediately executing the full unpack_entry logic.
                let flag_val = ((packed_data >> 40) & 0xFF) as u8;
                
                if flag_val != HashFlag::EMPTY as u8 {
                    // Exact populated match found. Unpack and handle mate calculations.
                    return self.unpack_entry(packed_data, key, ply);
                }
            }
        }
        
        // Return a clean empty entry if there's a complete bucket miss
        TTEntry {
            age: 0,
            key: 0,
            move_id: 0,
            score: 0,
            depth: 0,
            flag: HashFlag::EMPTY,
        }
    }

    #[inline(always)]
    pub fn store(&self, key: u64, score: i32, ply: i32, forward_move: ForwardMove, depth: i32, flag: HashFlag) {
        let index = (key as usize) & self.mask;
        let bucket = &self.buckets[index];
        
        // Read the current search age from self
        let age = self.age;

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

        let incoming_depth = depth as i8;
        // Pack the entry using the age retrieved from self
        let packed_data = self.pack_entry(move_id, storage_score as i16, depth, flag, age);

        // =========================================================================
        // PASS 1: Exact Key Match Check (Scan all 8 slots)
        // =========================================================================
        // Fetching the first slot pulls the whole 128-byte block into the M4 Pro L1.
        for i in 0..8 {
            let slot = &bucket.slots[i];
            let stored_key = slot.key.load(Ordering::Relaxed);

            if stored_key == key {
                let current_data = slot.data.load(Ordering::Relaxed);
                let current_depth = (current_data >> 32) as i8;
                let flag_val = ((current_data >> 40) & 0xFF) as u8;
                let current_age = ((current_data >> 48) & 0xFF) as u8;

                // Overwrite if empty, if the entry is stale (old age), or if the new data is deeper.
                if flag_val == HashFlag::EMPTY as u8 
                    || age != current_age 
                    || incoming_depth >= current_depth 
                {
                    slot.data.store(packed_data, Ordering::Relaxed);
                }
                return; // Match handled. Exit immediately.
            }
        }

        // =========================================================================
        // PASS 2: Condon-Thompson Replacement Logic (Brand New Position)
        // =========================================================================
                
        // --- Tier 1: Evaluate the Two Depth-Preferred Slots (Indices 0 and 1) ---
        for i in 0..2 {
            let slot = &bucket.slots[i];
            let current_data = slot.data.load(Ordering::Relaxed);
            let current_depth = (current_data >> 32) as i8;
            let flag_val = ((current_data >> 40) & 0xFF) as u8;
            let current_age = ((current_data >> 48) & 0xFF) as u8;

            // Claim if completely empty, if the entry belongs to an old search age,
            // or overwrite if incoming search depth is superior.
            if flag_val == HashFlag::EMPTY as u8 
                || age != current_age 
                || incoming_depth >= current_depth 
            {
                slot.key.store(key, Ordering::Relaxed);
                slot.data.store(packed_data, Ordering::Relaxed);
                return;
            }
        }

        // --- Tier 2: Evaluate the Six Always-Replace Slots (Indices 2 through 7) ---
        let mut target_index = 2;
        let mut min_depth = i8::MAX;
        let mut max_age_diff = 0i16; 

        for i in 2..8 {
            let slot = &bucket.slots[i];
            let current_data = slot.data.load(Ordering::Relaxed);
            let flag_val = ((current_data >> 40) & 0xFF) as u8;
            let current_depth = (current_data >> 32) as i8;
            let current_age = ((current_data >> 48) & 0xFF) as u8;

            // 1. Instantly claim any slot in the replacement pool that is completely empty
            if flag_val == HashFlag::EMPTY as u8 {
                slot.key.store(key, Ordering::Relaxed);
                slot.data.store(packed_data, Ordering::Relaxed);
                return;
            }

            // 2. Track the oldest entry (or the shallowest if ages match) for eviction
            let age_diff = (age as i16 - current_age as i16).abs();

            if age_diff > max_age_diff || (age_diff == max_age_diff && current_depth < min_depth) {
                max_age_diff = age_diff;
                min_depth = current_depth;
                target_index = i;
            }
        }

        // Overwrite the weakest/oldest available slot if our incoming entry holds superior depth,
        // or if the existing slot is completely stale (from a previous search age).
        let target_slot = &bucket.slots[target_index];
        let target_data = target_slot.data.load(Ordering::Relaxed);
        let target_age = ((target_data >> 48) & 0xFF) as u8;

        if age != target_age || incoming_depth >= min_depth {
            target_slot.key.store(key, Ordering::Relaxed);
            target_slot.data.store(packed_data, Ordering::Relaxed);
        }
    }

}