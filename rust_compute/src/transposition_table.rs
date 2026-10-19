use portable_atomic::{AtomicU128, Ordering};

use crate::move_command::*;
use crate::chess_game::*;

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
    pub key: u64,                       // 8 bytes

     // --- Flattened Move Data ---
    pub start_sq: u8,                   // 1 Byte
    pub end_sq: u8,                     // 1 Byte
    pub move_type: MoveFlag,            // 1 Byte

    pub score: i16,                     // 2 bytes
    pub depth: i8,                      // 1 byte
    pub flag: HashFlag,                 // 1 byte
    pub age: i8                         // 1 byte
}

#[derive(Debug)]
pub struct TTResult {
    pub score: i16,
    pub depth: i8,
    pub flag: HashFlag,
    pub forward_move: ForwardMove,
}

impl TTResult {
    /// Creates a default structure representing a total transposition cache miss.
    pub fn empty() -> Self {
        Self {
            score: 0,
            depth: 0,
            flag: HashFlag::EMPTY,
            forward_move: ForwardMove::NULL_MOVE,
        }
    }
}

// The TTEntry needs to be restructued as a AtomicU128 such that it can be used 
// inside load(Ordering::Relaxed) / store(packed_new, Ordering::Relaxed) without Locks 
impl TTEntry {
    pub fn empty() -> Self {
        Self {
            key: 0,
            start_sq: 0,
            end_sq: 0,
            move_type: MoveFlag::NULL,
            score: 0,
            depth: 0,
            flag: HashFlag::EXACT,
            age: 0,
        }
    }

    pub fn pack(&self) -> u128 {
        let key_bits = self.key as u128;
        
        let start_bits = (self.start_sq as u128) << 64;
        let end_bits   = (self.end_sq as u128) << 72;
        let type_bits  = (self.move_type as u128) << 80;
        let score_bits = (self.score as u16 as u128) << 88;
        let depth_bits = (self.depth as u8 as u128) << 104;
        let flag_bits  = (self.flag as u8 as u128) << 112;
        let age_bits   = (self.age as u8 as u128) << 120;
        
        key_bits | start_bits | end_bits | type_bits | score_bits | depth_bits | flag_bits | age_bits
    }

    pub fn unpack(bits: u128) -> Self {
        let key = bits as u64;
        let start_sq  = (bits >> 64) as u8;
        let end_sq    = (bits >> 72) as u8;
        
        let raw_type = (bits >> 80) as u8;
        let move_type: MoveFlag = unsafe { 
            if raw_type <= 10 {
                std::mem::transmute(raw_type)
            } else {
                MoveFlag::NULL
            }
        };
        
        let score = (bits >> 88) as i16;
        let depth = (bits >> 104) as i8;
        
        // Apply the same optimization to HashFlag (0..3)
        let raw_flag = (bits >> 112) as u8;
        let flag: HashFlag = unsafe {
            if raw_flag <= 3 {
                std::mem::transmute(raw_flag)
            } else {
                HashFlag::EMPTY
            }
        };
        
        let age = (bits >> 120) as i8;

        Self { key, start_sq, end_sq, move_type, score, depth, flag, age }
    }
}

// Condon-Thompson multi-tiered bucket layout.
#[cfg(target_arch = "aarch64")]
pub const SLOTS_PER_BUCKET: usize = 8;

#[cfg(not(target_arch = "aarch64"))]
pub const SLOTS_PER_BUCKET: usize = 4;

// Dynamically matches CPU architectures to guarantee optimal L1/L2 cache properties.
#[cfg_attr(target_arch = "aarch64", repr(C, align(128)))] // Apple Silicon (M1-M4) / ARM64
#[cfg_attr(not(target_arch = "aarch64"), repr(C, align(64)))] // Standard Intel/AMD x86_64
pub struct TtBucket {
    pub slots: [AtomicU128; 8],
}

// Condon-Thompson transposition table using packed 128-bit buckets
pub struct TranspositionTable {
    buckets: Vec<TtBucket>,
    mask: usize,
}

impl TranspositionTable {
    /// Creates a flat table matching the nearest power-of-two megabytes
    pub fn new(mb: usize) -> Self {
        let size_bytes = mb * 1024 * 1024;
        let count = size_bytes / std::mem::size_of::<TtBucket>();
        
        let mut final_count = count.next_power_of_two();
        if final_count > count && final_count > 1 {
            final_count >>= 1;
        }
        final_count = std::cmp::max(1, final_count);

        let mut buckets = Vec::with_capacity(final_count);
        for _ in 0..final_count {
            // Fill array with 0 atomic integers
            let slots = unsafe { std::mem::zeroed() };
            buckets.push(TtBucket { slots });
        }

        Self {
            buckets,
            mask: final_count - 1,
        }
    }

    // ========================================================================
    // THE DISTANCE-TO-MATE CONVERSION
    // ========================================================================
    // Our search loop evaluates checkmates relative to the tree root, returning:
    //   Winning Mate:  MATE_VALUE - total_search_ply
    //   Losing Mate:  -MATE_VALUE + total_search_ply
    //
    // Transposition Table should store distance to mate, not total Ply. Suppose at ply_10
    // 
    //   Winning: (MATE_VALUE - 14) + ply_10  => MATE_VALUE - 4  (Mate is 4 steps)
    //   Losing:  (-MATE_VALUE + 11) - ply_10 => -MATE_VALUE + 1 (Mate is 1 step)
    //
    // Probe -> Current Ply should be removed from the final score
    // Store -> Current Ply should be added to the final score to find the absolute distance
    // ========================================================================

    // The engine may return a combined ply that is beyond search horizon, this may be risky for an engine to execute
    // as Stockfish RL uses 80% Deep Search + 20% WLD. This will be kept for now, however we may need to revisit

    // Transposition Table Entries are unique for every key 
    // The 1/4 of the buckets is depth-preferred - 3/4 is always-replace 
    #[inline(always)]
     pub fn probe(&self, key: u64, ply: i16) -> TTResult {
        let bucket_idx = (key as usize) & self.mask;
        let bucket = &self.buckets[bucket_idx];

        for i in 0..SLOTS_PER_BUCKET {
            let bits = bucket.slots[i].load(Ordering::Relaxed);
            if bits == 0 {
                continue;
            }

            let entry = TTEntry::unpack(bits);           
            if entry.key == key {
                let mut local_score = entry.score;

                let is_pos_mate = local_score >= MATE_THRESHOLD;
                let is_neg_mate = local_score <= -MATE_THRESHOLD;
                local_score = local_score - (is_pos_mate as i16 * ply) + (is_neg_mate as i16 * ply);
                
                return TTResult {
                    score: local_score,
                    depth: entry.depth,
                    flag: entry.flag,
                    forward_move: ForwardMove {
                        start_sq: entry.start_sq,
                        end_sq: entry.end_sq,
                        capture_sq: 0,
                        move_type: entry.move_type,
                        pv_score: 0,
                    },
                };
            }
        }
        
        TTResult::empty()
    }

    // Key should be unique and is incremented once per move 
    // Always replace older entries -> Starting position with 30+ positions will not be encountered again
    // Depth-Preferred -> Replace if it defeats the deepest position
    // Always-Replace -> Randomly replace an entry 
    #[inline(always)]
    pub fn store(&self, 
        hash: u64, score: i16, 
        depth: i8, ply: i16, age: i8,
        mv: ForwardMove, flag: HashFlag, 
    ) {
        let bucket_idx = (hash as usize) & self.mask;
        let bucket = &self.buckets[bucket_idx];
        let midpoint = SLOTS_PER_BUCKET / 2;

        let is_pos_mate = score >= MATE_THRESHOLD;
        let is_neg_mate = score <= -MATE_THRESHOLD;
        let adjusted_score = score + (is_pos_mate as i16 * ply) - (is_neg_mate as i16 * ply);

       let new_entry = TTEntry { 
            key: hash, 
            start_sq: mv.start_sq,
            end_sq: mv.end_sq,
            move_type: mv.move_type,
            score: adjusted_score, 
            depth, 
            flag, 
            age 
        };
        let packed_new = new_entry.pack();

        // --- PHASE 1: IF HASH EXISTS, REPLACE THAT ITEM ---
        // Enforces the rule that all keys across the bucket must be completely unique.
        for i in 0..SLOTS_PER_BUCKET {
            let bits = bucket.slots[i].load(Ordering::Relaxed);
            if bits == 0 { continue; }

            let slot = TTEntry::unpack(bits);
            if slot.key == hash {
                let is_stale = slot.age != age;
                let is_depth_preferred = i < midpoint;
                let meets_depth = depth >= slot.depth;
                
                let should_replace = is_stale | !is_depth_preferred | (is_depth_preferred & meets_depth);
                let final_bits = if should_replace { packed_new } else { bits };
                bucket.slots[i].store(final_bits, Ordering::Relaxed);
        
                return;
            }
        }

        // --- PHASE 2: NEW HASH INSERTION ---
        
        // --- STEP A: Depth First Search ---
        // Replace Stale then Shallow-Depth
        let mut best_tier1_idx = 0;
        let mut lowest_priority = MAX_DEPTH;

        for i in 0..midpoint {
            let bits = bucket.slots[i].load(Ordering::Relaxed);
            if bits == 0 {
                bucket.slots[i].store(packed_new, Ordering::Relaxed);
                return;
            }

            let slot = TTEntry::unpack(bits);
            if slot.age != age {
                bucket.slots[i].store(packed_new, Ordering::Relaxed);
                return; // Claim stale slot instantly
            }

            if slot.depth < lowest_priority {
                lowest_priority = slot.depth;
                best_tier1_idx = i;
            }
        }

        if depth > lowest_priority {
            bucket.slots[best_tier1_idx].store(packed_new, Ordering::Relaxed);
            return;
        }

        // --- STEP B: Always Replace ---
        // Always Replace - Traffic is high and ovewrites are common.
        let random_offset = (hash >> 12) as usize % midpoint;
        let target_tier2_idx = midpoint + random_offset; 

        bucket.slots[target_tier2_idx].store(packed_new, Ordering::Relaxed);
    }
}