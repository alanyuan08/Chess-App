use crate::move_command::*;
use crate::chess_game::*;

#[derive(Debug, Clone, Copy)]
#[repr(C)] // 3 Bytes
pub struct TTForwardMove {
    pub start_sq: u8, // 1 Byte
    pub end_sq: u8, // 1 Byte
    pub move_type: MoveFlag, // 1 Byte
}

impl TTForwardMove {
    pub const NULL_MOVE: Self = TTForwardMove {
        start_sq: 0,
        end_sq: 0,
        move_type: MoveFlag::NULL,
    };

    #[inline(always)]
    pub const fn packForwardMove(forwardMove: ForwardMove) -> Self {
        Self {
            start_sq: forwardMove.start_sq,
            end_sq: forwardMove.end_sq,
            move_type: forwardMove.move_type,
        }
    }

    #[inline(always)]
    pub const fn unpackForwardMove(self) -> ForwardMove {
        ForwardMove {
            pv_score: 0,
            start_sq: self.start_sq,
            end_sq: self.end_sq,
            move_type: self.move_type,
        }
    }
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
    pub key: u64,       // 8 bytes
    pub tt_forward_move: TTForwardMove // 3 bytes
    pub score: i16,     // 2 bytes
    pub depth: i8,      // 1 byte
    pub flag: HashFlag,  // 1 byte
}

impl TTEntry {
    #[inline(always)]
    pub const fn empty() -> Self {
        Self {
            key: 0,
            tt_forward_move: TTForwardMove::NULL_MOVE,
            score: 0,
            depth: 0,
            flag: HashFlag::EMPTY,
        }
    }
}

impl Default for TTEntry {
    #[inline(always)]
    fn default() -> Self {
        Self::empty()
    }
}

// Condon-Thompson Bucket using 100% stable AtomicU64 primitives.
// current implement
#[derive(Clone)]
#[repr(C, align(128))]
pub struct TtBucket {
    // Slots 0 & 4: Depth Preferred
    // Slots 5 through 7: Always Replace
    pub slots: [TTEntry; 8],

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
        
        // Round down to the nearest power of two to guarantee the table fits safely 
        // within limits while preserving lightning-fast bitwise masking operations.
        let mut final_count = count.next_power_of_two();
        if final_count > count && final_count > 1 {
            final_count >>= 1;
        }
        final_count = std::cmp::max(1, final_count);

        let buckets = vec![TtBucket { slots: [TTEntry::empty(); 8] }; final_count];

        Self {
            buckets,
            mask: final_count - 1,
        }
    }

    #[inline(always)]
    pub fn probe(&self, key: u64, ply: i16) -> TTEntry {
        let index = (key as usize) & self.mask;
        let bucket = &self.buckets[index];
        
        // Scanning slots across the 128-byte cache line block.
        for i in 0..8 {
            let entry = bucket.slots[i];
            
            if entry.key == key && entry.flag != HashFlag::EMPTY {
                let mut returned_entry = entry;

                // --- Scale Mate Scores back to local search ply ---
                if returned_entry.score > MATE_THRESHOLD { 
                    returned_entry.score -= ply; 
                } else if returned_entry.score < -MATE_THRESHOLD { 
                    returned_entry.score += ply; 
                }
                
                return returned_entry;
            }
        }
        
        TTEntry::empty()
    }

    #[inline(always)]
    pub fn store(&mut self, key: u64, score: i16, ply: i16, 
        forward_move: ForwardMove, depth: i8, flag: HashFlag
    ) {
        let index = (key as usize) & self.mask;
        let bucket = &mut self.buckets[index];

        let tt_forward_move = TTForwardMove::packForwardMove(forward_move);

        let mut storage_score = score;
        if storage_score >= MATE_THRESHOLD {
            storage_score += ply;
        } else if storage_score <= -MATE_THRESHOLD {
            storage_score -= ply;
        }

        let incoming_depth = depth;

        // =========================================================================
        // PASS 1: Exact Key Match Overwrite Check
        // =========================================================================
        for i in 0..8 {
            let slot = &mut bucket.slots[i];
            if slot.key == key {
                // Overwrite existing data if the new search is as deep or deeper
                if slot.flag == HashFlag::EMPTY || incoming_depth >= slot.depth {
                    slot.tt_forward_move = tt_forward_move;
                    slot.score = storage_score;
                    slot.depth = incoming_depth;
                    slot.flag = flag;
                }
                return;
            }
        }

        // =========================================================================
        // PASS 2: Depth-Preferred Protection (Slots 0 and 1)
        // =========================================================================
        for i in 0..2 {
            let slot = &mut bucket.slots[i];
            if slot.flag == HashFlag::EMPTY || incoming_depth >= slot.depth {
                *slot = TTEntry { key, tt_forward_move, score: storage_score, depth: incoming_depth, flag };
                return;
            }
        }

        // =========================================================================
        // PASS 3: Always-Replace Churn Fallback (Slots 2 through 7)
        // =========================================================================
        // We map the remaining 6 slots. key % 6 yields 0..5, which maps perfectly to indices 2..7.
        let replace_index = 2 + (key % 6) as usize; 
        bucket.slots[replace_index] = TTEntry { key, tt_forward_move, score: storage_score, depth: incoming_depth, flag };
    }
}