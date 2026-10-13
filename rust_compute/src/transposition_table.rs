#![feature(integer_atomics)]

use std::sync::atomic::{AtomicU128, Ordering};

use crate::move_command::*;
use crate::chess_game::*;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct TTForwardMove {
    pub start_sq: u8,        // 1 Byte
    pub end_sq: u8,          // 1 Byte
    pub move_type: MoveFlag, // 1 Byte
}

impl TTForwardMove {
    pub const NULL_MOVE: Self = TTForwardMove {
        start_sq: 0,
        end_sq: 0,
        move_type: MoveFlag::NULL,
    };
}

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
    pub key: u64,                       // 8 bytes
    pub tt_forward_move: TTForwardMove, // 3 bytes
    pub score: i16,                     // 2 bytes
    pub depth: i8,                      // 1 byte
    pub flag: HashFlag,                 // 1 byte
    pub age: i8                         // 1 byte
}

// Probe Result
#[derive(Debug)]
pub struct ProbeResult {
    pub score: i16,
    pub flag: HashFlag,
    pub depth: u8,
    pub alpha: i16,
    pub beta: i16,
    pub pv_move: TTForwardMove,
    pub is_cutoff: bool,
}

// The TTEntry needs to be restructued as a AtomicU128 such that it can be used 
// inside load(Ordering::Relaxed) / store(packed_new, Ordering::Relaxed) without Locks 
impl TTEntry {
    pub fn empty() -> Self {
        Self {
            key: 0,
            tt_forward_move: TTForwardMove::NULL_MOVE,
            score: 0,
            depth: 0,
            flag: HashFlag::EXACT,
            age: 0,
        }
    }

    pub fn pack(&self) -> u128 {
        let key_bits = self.key as u128;
        
        // Pack 3 bytes of move data
        let move_raw: [u8; 3] = unsafe { std::mem::transmute_copy(&self.tt_forward_move) };
        let move_bits = (move_raw[0] as u128) | ((move_raw[1] as u128) << 8) | ((move_raw[2] as u128) << 16);
        
        let score_bits = (self.score as u16 as u128) << 24;
        let depth_bits = (self.depth as u128) << 40;
        let flag_bits  = (self.flag as u8 as u128) << 48;
        let age_bits   = (self.age as u8 as u128) << 56;
        
        let data_bits = move_bits | score_bits | depth_bits | flag_bits | age_bits;
        key_bits | (data_bits << 64)
    }

    pub fn unpack(bits: u128) -> Self {
        let key = bits as u64;
        let data_bits = (bits >> 64) as u64;
        
        let move_raw = [
            data_bits as u8,
            (data_bits >> 8) as u8,
            (data_bits >> 16) as u8,
        ];
        let tt_forward_move: TTForwardMove = unsafe { std::mem::transmute_copy(&move_raw) };
        
        let score = (data_bits >> 24) as i16;
        let depth = (data_bits >> 40) as u8;
        
        let flag = match (data_bits >> 48) as u8 {
            1 => HashFlag::Alpha,
            2 => HashFlag::Beta,
            _ => HashFlag::Exact,
        };
        
        let age = (data_bits >> 56) as i8;

        Self { key, tt_forward_move, score, depth, flag, age }
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

    #[inline(always)]
    pub fn probe(&self, key: u64, ply: i16, depth: u8, orig_alpha: i16, orig_beta: i16) -> ProbeResult {
        let bucket_idx = (key as usize) & self.mask;
        let bucket = &self.buckets[bucket_idx];

        let mut current_alpha = orig_alpha;
        let mut current_beta = orig_beta;
        
        let mut deepest_fallback = TTEntry::empty();

        for i in 0..SLOTS_PER_BUCKET {
            let bits = bucket.slots[i].load(Ordering::Relaxed);
            if bits == 0 {
                continue;
            }

            let entry = TTEntry::unpack(bits);
            
            if entry.key == key {
                let mut returned_entry = entry;

                // --- Scale Mate Scores back to local search ply ---
                if returned_entry.score >= MATE_THRESHOLD { 
                    returned_entry.score -= ply; 
                } else if returned_entry.score <= -MATE_THRESHOLD { 
                    returned_entry.score += ply; 
                }
                
                // --- 1. Pure Exact Cutoff Check ---
                if returned_entry.depth >= depth && returned_entry.flag == HashFlag::Exact {
                    return ProbeResult {
                        score: returned_entry.score,
                        flag: HashFlag::Exact,
                        depth: returned_entry.depth,
                        alpha: current_alpha,
                        beta: current_beta,
                        pv_move: returned_entry.tt_forward_move,
                        is_cutoff: true,
                    };
                }

                // --- 2. Dynamic Window Squeezing ---
                if returned_entry.depth >= depth {
                    match returned_entry.flag {
                        HashFlag::Beta => {
                            // Lower bound found. Attempt to raise alpha.
                            if returned_entry.score > current_alpha {
                                current_alpha = returned_entry.score;
                            }
                        }
                        HashFlag::Alpha => {
                            // Upper bound found. Attempt to lower beta.
                            if returned_entry.score < current_beta {
                                current_beta = returned_entry.score;
                            }
                        }
                        _ => {}
                    }

                    // Alpha-Beta Cut-Off
                    if current_alpha >= current_beta {
                        return ProbeResult {
                            score: returned_entry.score,
                            flag: returned_entry.flag,
                            depth: returned_entry.depth,
                            alpha: current_alpha,
                            beta: current_beta,
                            pv_move: returned_entry.tt_forward_move,
                            is_cutoff: true,
                        };
                    }
                }

                // --- 3. Deepest Tracking Fallback ---
                if returned_entry.depth > deepest_fallback.depth {
                    deepest_fallback = returned_entry;
                }
            }
        }
        
        // --- 4. Open Window Pass-through ---
        ProbeResult {
            score: deepest_fallback.score,
            flag: deepest_fallback.flag,
            depth: deepest_fallback.depth,
            alpha: current_alpha,
            beta: current_beta,
            pv_move: deepest_fallback.tt_forward_move,
            is_cutoff: false,
        }
    }

    #[inline(always)]
    pub fn store(&self, hash: u64, depth: u8, flag: HashFlag, score: i16, mv: TTForwardMove, age: i8) {
        let bucket_idx = (hash as usize) & self.mask;
        let bucket = &self.buckets[bucket_idx];
        
        // Split exactly 50/50:
        let midpoint = SLOTS_PER_BUCKET / 2;

        let new_entry = TTEntry {
            key: hash,
            tt_forward_move: mv,
            score,
            depth,
            flag,
            age,
        };
        let packed_new = new_entry.pack();

        // --- PHASE 1: SCAN ALL 8 SLOTS FOR AN EXACT HASH MATCH FIRST ---
        // Guarantees key uniqueness across both tiers for fast probe early-exits.
        for i in 0..SLOTS_PER_BUCKET {
            let bits = bucket.slots[i].load(Ordering::Relaxed);
            if bits == 0 { continue; }

            let slot = TTEntry::unpack(bits);
            
            if slot.key == hash {
                // Determine entry strength criteria based on flag bounds and depth
                let incoming_has_stronger_flag = flag == HashFlag::Exact && slot.flag != HashFlag::Exact;
                let flags_are_equivalent = flag == slot.flag || (flag != HashFlag::Exact && slot.flag != HashFlag::Exact);
                
                let incoming_is_stronger = incoming_has_stronger_flag || (flags_are_equivalent && depth >= slot.depth);

                if i >= midpoint {
                    // --- Tier 2 (Always-Replace Section) Overwrite Rule ---
                    // Replace ONLY if the new cutoff representation is structurally stronger
                    if incoming_is_stronger {
                        bucket.slots[i].store(packed_new, Ordering::Relaxed);
                    }
                    return;
                }

                // --- Tier 1 (Depth-Preferred Section) Matching Path ---
                if incoming_is_stronger {
                    // Current entry is better or equal, overwrite in-place in Tier 1
                    bucket.slots[i].store(packed_new, Ordering::Relaxed);
                    return;
                }

                // Demotion Rule: The incoming entry is weaker than the existing Tier 1 entry. 
                // We update Tier 1 with the fresh entry and push the deep entry down to Tier 2.
                let mut worst_tier2_idx = midpoint;
                let mut oldest_age_diff = -1;
                let mut shallowest_t2_depth = u8::MAX;

                for j in midpoint..SLOTS_PER_BUCKET {
                    let t2_bits = bucket.slots[j].load(Ordering::Relaxed);
                    if t2_bits == 0 {
                        worst_tier2_idx = j;
                        break;
                    }
                    
                    let t2_slot = TTEntry::unpack(t2_bits);
                    let age_difference = (age as i16 - t2_slot.age as i16).abs();
                    
                    // Prioritize the oldest age difference. 
                    // If age is equal, evict the one that has the shallower search depth.
                    if age_difference > oldest_age_diff {
                        oldest_age_diff = age_difference;
                        shallowest_t2_depth = t2_slot.depth;
                        worst_tier2_idx = j;
                    } else if age_difference == oldest_age_diff && t2_slot.depth < shallowest_t2_depth {
                        shallowest_t2_depth = t2_slot.depth;
                        worst_tier2_idx = j;
                    }
                }

                // Atomically swap the nodes
                bucket.slots[worst_tier2_idx].store(bits, Ordering::Relaxed); // Demote deep entry down
                bucket.slots[i].store(packed_new, Ordering::Relaxed);        // Place fresh entry in Tier 1
                return;
            }
        }

        // --- PHASE 2: NO MATCHING KEY FOUND (NEW NODE INSERTION) ---
        
        // --- PART A: TIER 1 (DEPTH-PREFERRED / AGE FRESHNESS) ---
        let mut replace_idx = 0;
        let mut lowest_priority = i32::MAX;

        for i in 0..midpoint {
            let bits = bucket.slots[i].load(Ordering::Relaxed);
            if bits == 0 {
                // Found an empty slot, claim it instantly
                bucket.slots[i].store(packed_new, Ordering::Relaxed);
                return;
            }

            let slot = TTEntry::unpack(bits);
            
            // Age eviction priority rules: If the entry belongs to a dead search turn, 
            // set its score to 0 to make it the prime target for replacement.
            let is_stale = slot.age != age;
            let slot_priority = if is_stale { 0 } else { slot.depth as i32 };

            if slot_priority < lowest_priority {
                lowest_priority = slot_priority;
                replace_idx = i;
            }
        }

        // Claim the slot if our new node is deeper, or if the target slot is stale (priority 0)
        if (depth as i32) > lowest_priority || lowest_priority == 0 {
            bucket.slots[replace_idx].store(packed_new, Ordering::Relaxed);
            return;
        }

        // --- PART B: TIER 2 (ALWAYS-REPLACE WITH AGE EVICTION) ---
        let mut worst_tier2_idx = midpoint;
        let mut oldest_age_diff = -1;

        for i in midpoint..SLOTS_PER_BUCKET {
            let bits = bucket.slots[i].load(Ordering::Relaxed);
            if bits == 0 {
                bucket.slots[i].store(packed_new, Ordering::Relaxed);
                return;
            }

            let slot = TTEntry::unpack(bits);
            let age_difference = (age as i16 - slot.age as i16).abs();
            if age_difference > oldest_age_diff {
                oldest_age_diff = age_difference;
                worst_tier2_idx = i;
            }
        }

        // Replace the oldest, most decayed entry in Tier 2
        bucket.slots[worst_tier2_idx].store(packed_new, Ordering::Relaxed);
    }
}